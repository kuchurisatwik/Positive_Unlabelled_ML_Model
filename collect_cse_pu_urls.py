from __future__ import annotations

import argparse
import bz2
import concurrent.futures
import csv
import io
import json
import os
import re
import subprocess
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, TypeVar
from urllib.parse import urlparse

import pandas as pd

try:
    import psutil
except Exception:  # pragma: no cover - optional until requirements are installed
    psutil = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - dependency fallback
    tqdm = None

try:
    import dns.resolver as dns_resolver
except Exception:  # pragma: no cover - dependency fallback
    dns_resolver = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract.feature_extractor import (  # noqa: E402
    UrlContext,
    brand_tokens,
    compute_lexical_features,
    ensure_url,
    normalize_hostname,
)

CSE_FILE = Path(os.environ.get("PIPELINE_CSE_FILE", str(SCRIPT_DIR / "input" / "cse" / "cse_list.csv")))
DEFAULT_LOCAL_INPUT = SCRIPT_DIR / "input" / "dataset"
DEFAULT_OUTPUT = SCRIPT_DIR / "input" / "pu_candidates" / "cse_pu_candidates.csv"
SUPPORTED_INPUT_SUFFIXES = {".csv", ".xlsx", ".xls"}
SKIPPED_INPUT_NAME_PARTS = ("audit", "dns_failed", "failed_urls")

DETECTED_COLUMN = "Identified Phishing/Suspected Domain Name"
TARGET_COLUMN = "Corresponding CSE Domain Name"
CSE_NAME_COLUMN = "Critical Sector Entity Name"
LABEL_COLUMN = "Phishing/Suspected Domains (i.e. Class Label)"
DNS_ACTIVE_SOURCE = "dns_active_filter"
DEFAULT_MAX_ACTIVE_URLS = 5000
T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class CseRecord:
    domain: str
    name: str = ""


@dataclass
class CpuGate:
    threshold_percent: float
    sample_seconds: float
    check_interval_seconds: float
    max_wait_seconds: float
    wait_events: int = 0
    waited_seconds: float = 0.0
    checks: int = 0
    last_cpu_percent: float = 0.0
    max_seen_cpu_percent: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def enabled(self) -> bool:
        return self.threshold_percent < 100.0

    def _cpu_percent(self) -> float:
        if psutil is not None:
            return float(psutil.cpu_percent(interval=self.sample_seconds))
        if hasattr(os, "getloadavg"):
            load_1m = os.getloadavg()[0]
            cpu_count = max(1, os.cpu_count() or 1)
            return min(100.0, max(0.0, (load_1m / cpu_count) * 100.0))
        raise RuntimeError(
            "Strict CPU gate requires psutil on this platform. "
            "Run `pip install -r requirements.txt`, or disable the gate with `--cpu-gate-threshold 100`."
        )

    def wait(self, label: str) -> None:
        if not self.enabled:
            return

        started = time.monotonic()
        waited = False
        while True:
            cpu_percent = self._cpu_percent()
            with self._lock:
                self.checks += 1
                self.last_cpu_percent = cpu_percent
                self.max_seen_cpu_percent = max(self.max_seen_cpu_percent, cpu_percent)

            if cpu_percent <= self.threshold_percent:
                if waited:
                    with self._lock:
                        self.waited_seconds += time.monotonic() - started
                return

            waited = True
            if self.max_wait_seconds > 0 and time.monotonic() - started >= self.max_wait_seconds:
                raise RuntimeError(
                    f"CPU stayed above {self.threshold_percent:.1f}% before {label} "
                    f"for {self.max_wait_seconds:.1f}s."
                )

            with self._lock:
                self.wait_events += 1
            time.sleep(self.check_interval_seconds)

    def summary(self) -> dict[str, float | int | str]:
        with self._lock:
            return {
                "enabled": int(self.enabled),
                "threshold_percent": float(self.threshold_percent),
                "checks": int(self.checks),
                "wait_events": int(self.wait_events),
                "waited_seconds": round(float(self.waited_seconds), 3),
                "last_cpu_percent": round(float(self.last_cpu_percent), 3),
                "max_seen_cpu_percent": round(float(self.max_seen_cpu_percent), 3),
            }


@dataclass
class DnsActiveChecker:
    timeout_seconds: float
    lifetime_seconds: float
    record_types: tuple[str, ...]
    cache_enabled: bool = True
    socket_fallback: bool = False
    checks: int = 0
    cache_hits: int = 0
    active: int = 0
    inactive: int = 0
    errors: int = 0
    _cache: dict[str, bool] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def is_active(self, value: str) -> bool:
        host = normalize_hostname(value)
        if not host:
            return False

        if self.cache_enabled:
            with self._lock:
                cached = self._cache.get(host)
                if cached is not None:
                    self.cache_hits += 1
                    return cached

        is_active = self._resolve(host)
        with self._lock:
            self.checks += 1
            if is_active:
                self.active += 1
            else:
                self.inactive += 1
            if self.cache_enabled:
                self._cache[host] = is_active
        return is_active

    def _resolve(self, host: str) -> bool:
        if dns_resolver is not None:
            resolver = dns_resolver.Resolver()
            resolver.timeout = self.timeout_seconds
            resolver.lifetime = self.lifetime_seconds
            for record_type in self.record_types:
                try:
                    answers = resolver.resolve(host, record_type, raise_on_no_answer=False)
                    if any(True for _ in answers):
                        return True
                except Exception:
                    continue

        if self.socket_fallback or dns_resolver is None:
            previous_timeout = socket.getdefaulttimeout()
            try:
                socket.setdefaulttimeout(self.timeout_seconds)
                return bool(socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP))
            except Exception:
                with self._lock:
                    self.errors += 1
                return False
            finally:
                socket.setdefaulttimeout(previous_timeout)

        return False

    def summary(self) -> dict[str, int | float | str]:
        with self._lock:
            return {
                "checks": int(self.checks),
                "cache_hits": int(self.cache_hits),
                "active": int(self.active),
                "inactive": int(self.inactive),
                "errors": int(self.errors),
                "cache_size": int(len(self._cache)),
                "timeout_seconds": float(self.timeout_seconds),
                "lifetime_seconds": float(self.lifetime_seconds),
                "record_types": ",".join(self.record_types),
                "socket_fallback": int(self.socket_fallback),
            }


def _cell(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "nat", "none", "null"}:
        return ""
    return text


def default_dnstwist_command() -> str:
    executable_name = "dnstwist.exe" if os.name == "nt" else "dnstwist"
    candidate = Path(sys.executable).with_name(executable_name)
    if candidate.exists():
        return str(candidate)
    return "dnstwist"


def make_cpu_gate(args: argparse.Namespace) -> CpuGate:
    if not 0 < args.cpu_gate_threshold <= 100:
        raise ValueError("--cpu-gate-threshold must be between 0 and 100.")
    if args.max_active_urls < 0:
        raise ValueError("--max-active-urls must be 0 or greater.")
    return CpuGate(
        threshold_percent=float(args.cpu_gate_threshold),
        sample_seconds=max(0.05, float(args.cpu_gate_sample_seconds)),
        check_interval_seconds=max(0.05, float(args.cpu_gate_check_interval)),
        max_wait_seconds=max(0.0, float(args.cpu_gate_max_wait_seconds)),
    )


def make_dns_checker(args: argparse.Namespace) -> DnsActiveChecker:
    record_types = tuple(item.upper() for item in _parse_repeated_values([args.dns_record_types]) if item)
    if not record_types:
        raise ValueError("--dns-record-types must contain at least one DNS record type.")
    return DnsActiveChecker(
        timeout_seconds=max(0.1, float(args.dns_timeout)),
        lifetime_seconds=max(0.1, float(args.dns_lifetime)),
        record_types=record_types,
        cache_enabled=not bool(args.no_dns_cache),
        socket_fallback=bool(args.dns_socket_fallback),
    )


def run_parallel_gated(
    items: Iterable[T],
    worker: Callable[[T], R],
    max_workers: int,
    cpu_gate: CpuGate | None,
    label: str,
    show_progress: bool = True,
) -> list[R]:
    item_list = list(items)
    if not item_list:
        return []

    progress = None
    if show_progress and tqdm is not None and len(item_list) > 1:
        progress = tqdm(total=len(item_list), desc=label, unit="item", leave=False)

    workers = max(1, int(max_workers))
    try:
        if workers <= 1 or len(item_list) == 1:
            results: list[R] = []
            for index, item in enumerate(item_list, start=1):
                if cpu_gate is not None:
                    cpu_gate.wait(f"{label} {index}/{len(item_list)}")
                results.append(worker(item))
                if progress is not None:
                    progress.update(1)
            return results

        indexed_results: list[tuple[int, R]] = []
        item_iter = iter(enumerate(item_list))

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            pending: dict[concurrent.futures.Future[R], int] = {}

            def submit_next() -> bool:
                try:
                    index, item = next(item_iter)
                except StopIteration:
                    return False
                if cpu_gate is not None:
                    cpu_gate.wait(f"{label} {index + 1}/{len(item_list)}")
                pending[executor.submit(worker, item)] = index
                return True

            for _ in range(min(workers, len(item_list))):
                submit_next()

            while pending:
                done, _ = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    index = pending.pop(future)
                    indexed_results.append((index, future.result()))
                    if progress is not None:
                        progress.update(1)
                    submit_next()

        return [result for _, result in sorted(indexed_results, key=lambda item: item[0])]
    finally:
        if progress is not None:
            progress.close()


def _first(row: pd.Series, *columns: str) -> str:
    for column in columns:
        if column in row.index:
            value = _cell(row[column])
            if value:
                return value
    return ""


def _split_domain_cell(value: str) -> list[str]:
    text = _cell(value)
    if not text:
        return []
    parts = re.split(r"[\s,;|]+", text)
    domains: list[str] = []
    for part in parts:
        cleaned = part.strip("()[]{}'\"")
        host = normalize_hostname(cleaned)
        if host:
            domains.append(host)
    return domains


def load_cse_records() -> list[CseRecord]:
    if not CSE_FILE.exists():
        raise FileNotFoundError(f"CSE list not found: {CSE_FILE}")

    records: dict[str, CseRecord] = {}
    df = pd.read_excel(CSE_FILE) if CSE_FILE.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(CSE_FILE)
    for _, row in df.iterrows():
        domain_text = _first(row, "domains", "domain", "Legitimate Domains", "Public URL", "CSE Domain", "Domain", "URL")
        name = _first(row, "cse", "CSE", "Cooresponding CSE", "Corresponding CSE", "Application Name", "Critical Sector Entity Name")
        for domain in _split_domain_cell(domain_text):
            if domain and domain not in records:
                records[domain] = CseRecord(domain=domain, name=name)
    return sorted(records.values(), key=lambda item: item.domain)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _is_candidate_input_file(path: Path) -> bool:
    name = path.name.lower()
    return (
        path.is_file()
        and not name.startswith("~$")
        and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
        and not any(part in name for part in SKIPPED_INPUT_NAME_PARTS)
    )


def iter_local_input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if _is_candidate_input_file(path) else []
    if not path.exists():
        return []
    return sorted(item for item in path.iterdir() if _is_candidate_input_file(item))


def _candidate_series(df: pd.DataFrame) -> pd.Series:
    preferred = (
        DETECTED_COLUMN,
        "Identified Phishing Domain Name",
        "Identified Suspected Domain Name",
        "domain",
        "Domain",
        "domain_name",
        "url",
        "URL",
    )
    for column in preferred:
        if column in df.columns:
            return df[column]
    for column in df.columns:
        lowered = str(column).lower()
        if "domain" in lowered or "url" in lowered:
            return df[column]
    raise ValueError(f"No domain/url column found. Columns: {list(df.columns)}")


def _url_alnum(value: str) -> str:
    parsed = urlparse(ensure_url(value))
    return re.sub(r"[^a-z0-9]+", "", f"{parsed.hostname or ''} {parsed.path}".lower())


def best_cse_match(candidate: str, records: list[CseRecord]) -> tuple[CseRecord | None, float]:
    host = normalize_hostname(candidate)
    if not host:
        return None, 0.0

    domains = [record.domain for record in records]
    record_by_domain = {record.domain: record for record in records}
    context = UrlContext(url=ensure_url(candidate), detected_domain=host)
    lexical = compute_lexical_features(context, domains)
    best_domain = lexical.get("matched_brand_domain") or ""
    best_score = max(
        float(lexical.get("lexical_similarity_score") or 0.0),
        float(lexical.get("homoglyph_similarity_score") or 0.0),
    )
    best_record = record_by_domain.get(best_domain)

    candidate_alnum = _url_alnum(candidate)
    for record in records:
        tokens = brand_tokens(record.domain, record.name)
        if tokens and any(token in candidate_alnum for token in tokens):
            token_score = 0.86
            if token_score > best_score:
                best_score = token_score
                best_record = record

    return best_record, best_score


def _output_row(
    url: str,
    record: CseRecord,
    label: str,
    source: str,
    score: float,
    source_file: str = "",
) -> dict[str, str]:
    today = date.today().strftime("%d-%m-%Y")
    return {
        "Application_ID": "",
        "Source of detection": source,
        DETECTED_COLUMN: ensure_url(url),
        TARGET_COLUMN: record.domain,
        CSE_NAME_COLUMN: record.name,
        LABEL_COLUMN: label,
        "Date of detection (DD-MM-YYYY)": today,
        "Time of detection (HH-MM-SS)": "",
        "Remarks": f"cse_match_score={score:.6f};source_file={source_file}",
    }


def collect_matched_rows(
    values: Iterable[str],
    records: list[CseRecord],
    threshold: float,
    label: str,
    source: str,
    source_file: str,
    max_workers: int,
    cpu_gate: CpuGate | None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    candidates = [_cell(value) for value in values]
    candidates = [value for value in candidates if value]

    def worker(value: str) -> tuple[str, CseRecord | None, float]:
        match, score = best_cse_match(value, records)
        return value, match, score

    rows: list[dict[str, str]] = []
    skipped = 0
    seen: set[tuple[str, str]] = set()

    for value, match, score in run_parallel_gated(candidates, worker, max_workers, cpu_gate, f"{source}_match"):
        if match is None or score < threshold:
            skipped += 1
            continue
        key = (normalize_hostname(value), match.domain)
        if key in seen:
            continue
        seen.add(key)
        rows.append(_output_row(value, match, label, source, score, source_file))

    return rows, {"loaded": len(candidates), "matched": len(rows), "skipped": skipped}


def collect_local_unlabeled_file(
    path: Path,
    records: list[CseRecord],
    threshold: float,
    max_workers: int,
    cpu_gate: CpuGate | None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    df = _read_table(path)
    series = _candidate_series(df)
    return collect_matched_rows(
        values=series.dropna().astype(str),
        records=records,
        threshold=threshold,
        label="Unlabeled",
        source="local_lexical_candidates",
        source_file=path.name,
        max_workers=max_workers,
        cpu_gate=cpu_gate,
    )


def collect_local_unlabeled(
    path: Path,
    records: list[CseRecord],
    threshold: float,
    max_workers: int,
    cpu_gate: CpuGate | None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    input_files = iter_local_input_files(path)
    rows: list[dict[str, str]] = []
    summary = {"files": len(input_files), "loaded": 0, "matched": 0, "skipped": 0}
    for input_file in input_files:
        file_rows, file_summary = collect_local_unlabeled_file(input_file, records, threshold, max_workers, cpu_gate)
        rows.extend(file_rows)
        summary["loaded"] += file_summary["loaded"]
        summary["matched"] += file_summary["matched"]
        summary["skipped"] += file_summary["skipped"]
    return rows, summary


def _read_feed_file(path: Path) -> list[str]:
    raw = path.read_bytes()
    if path.suffix.lower() == ".bz2":
        raw = bz2.decompress(raw)
    text = raw.decode("utf-8", errors="replace")

    if path.suffix.lower() in {".json", ".bz2"}:
        try:
            data = json.loads(text)
            if isinstance(data, list):
                urls = []
                for item in data:
                    if isinstance(item, str):
                        urls.append(item)
                    elif isinstance(item, dict) and item.get("url"):
                        urls.append(str(item["url"]))
                if urls:
                    return urls
        except json.JSONDecodeError:
            pass

    if path.suffix.lower() == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        urls = []
        for row in reader:
            value = row.get("url") or row.get("URL") or next(iter(row.values()), "")
            if value:
                urls.append(str(value))
        return urls

    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def _download_text(url: str, timeout: int = 40) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_openphish() -> list[str]:
    text = _download_text("https://openphish.com/feed.txt")
    return [line.strip() for line in text.splitlines() if line.strip()]


def fetch_phishtank() -> list[str]:
    key = os.environ.get("PHISHTANK_APP_KEY", "").strip()
    urls = []
    if key:
        urls.append(f"https://data.phishtank.com/data/{key}/online-valid.csv")
    urls.extend(
        [
            "https://data.phishtank.com/data/online-valid.csv",
            "http://data.phishtank.com/data/online-valid.csv",
        ]
    )
    last_error = ""
    for url in urls:
        try:
            text = _download_text(url)
            reader = csv.DictReader(io.StringIO(text))
            return [row["url"].strip() for row in reader if row.get("url")]
        except (urllib.error.URLError, TimeoutError, KeyError, csv.Error) as exc:
            last_error = f"{url}: {type(exc).__name__}"
    raise RuntimeError(f"Could not fetch PhishTank feed. Last error: {last_error}")


def collect_feed_positives(
    feed_urls: list[str],
    records: list[CseRecord],
    threshold: float,
    source: str,
    max_workers: int,
    cpu_gate: CpuGate | None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    return collect_matched_rows(
        values=feed_urls,
        records=records,
        threshold=threshold,
        label="Phishing",
        source=source,
        source_file="",
        max_workers=max_workers,
        cpu_gate=cpu_gate,
    )


def _parse_repeated_values(values: list[str] | None) -> list[str]:
    parsed: list[str] = []
    for value in values or []:
        parsed.extend(item.strip() for item in value.split(",") if item.strip())
    return parsed


def _dnstwist_domain_from_row(row: dict[str, object]) -> str:
    for key in ("domain-name", "domain", "hostname", "url", "URL"):
        value = row.get(key)
        host = normalize_hostname(value)
        if host:
            return host
    return ""


def _parse_dnstwist_json(text: str) -> list[dict[str, object]]:
    stripped = text.strip()
    if not stripped:
        return []

    start = stripped.find("[")
    end = stripped.rfind("]")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]

    data = json.loads(stripped)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def run_dnstwist(
    domain: str,
    command: str,
    timeout: int,
) -> list[str]:
    completed = subprocess.run(
        [command, "--format", "json", "--registered", domain],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(stderr or f"dnstwist exited with code {completed.returncode}")

    urls: list[str] = []
    for row in _parse_dnstwist_json(completed.stdout):
        host = _dnstwist_domain_from_row(row)
        if not host:
            continue
        if host == domain or host.endswith(f".{domain}"):
            continue
        urls.append(ensure_url(host))
    return urls


def filter_dns_active_urls(
    urls: list[str],
    max_workers: int,
    cpu_gate: CpuGate | None,
    dns_checker: DnsActiveChecker,
) -> tuple[list[str], int]:
    unique_urls = list(dict.fromkeys(urls))
    if not unique_urls:
        return [], 0

    active_hosts: set[str] = set()
    inactive = 0

    def worker(url: str) -> tuple[str, bool]:
        try:
            return url, dns_checker.is_active(url)
        except Exception:
            return url, False

    for url, is_active in run_parallel_gated(unique_urls, worker, max_workers, cpu_gate, "dns_active_check"):
        if is_active:
            active_hosts.add(normalize_hostname(url))
        else:
            inactive += 1

    return [url for url in unique_urls if normalize_hostname(url) in active_hosts], inactive


def filter_dns_active_rows(
    rows: list[dict[str, str]],
    max_workers: int,
    cpu_gate: CpuGate | None,
    dns_checker: DnsActiveChecker,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    urls = [row[DETECTED_COLUMN] for row in rows]
    active_urls, _ = filter_dns_active_urls(urls, max_workers, cpu_gate, dns_checker)
    active_hosts = {normalize_hostname(url) for url in active_urls}
    filtered = [row for row in rows if normalize_hostname(row[DETECTED_COLUMN]) in active_hosts]
    return filtered, {"loaded": len(rows), "matched": len(filtered), "skipped": len(rows) - len(filtered)}


def _dnstwist_source_domains(records: list[CseRecord], requested_domains: list[str], limit: int) -> list[str]:
    if requested_domains:
        domains = [normalize_hostname(value) for value in requested_domains]
    else:
        domains = [record.domain for record in records]
    unique_domains = [domain for domain in dict.fromkeys(domains) if domain]
    if limit > 0:
        return unique_domains[:limit]
    return unique_domains


def collect_dnstwist_candidates(
    records: list[CseRecord],
    threshold: float,
    command: str,
    timeout: int,
    dnstwist_workers: int,
    dns_workers: int,
    match_workers: int,
    requested_domains: list[str],
    limit: int,
    label: str,
    cpu_gate: CpuGate | None,
    dns_checker: DnsActiveChecker,
    max_rows: int,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    domains = _dnstwist_source_domains(records, requested_domains, limit)

    def worker(domain: str) -> tuple[str, list[str], str]:
        try:
            return domain, run_dnstwist(domain, command, timeout), ""
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"dnstwist command not found: {command}. Install it with `pip install dnstwist` "
                "or pass --dnstwist-command."
            ) from exc
        except (subprocess.TimeoutExpired, json.JSONDecodeError, RuntimeError) as exc:
            return domain, [], f"{type(exc).__name__}: {exc}"

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    failed_domains = 0
    processed_domains = 0
    generated_count = 0
    dns_active_count = 0
    dns_inactive = 0
    lexical_skipped = 0
    capped = 0

    workers = max(1, int(dnstwist_workers))
    domain_iter = iter(enumerate(domains))
    progress = tqdm(total=len(domains), desc="dnstwist_domain", unit="domain", leave=False) if tqdm is not None and len(domains) > 1 else None

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            pending: dict[concurrent.futures.Future[tuple[str, list[str], str]], int] = {}

            def submit_next() -> bool:
                if max_rows > 0 and len(rows) >= max_rows:
                    return False
                try:
                    index, domain = next(domain_iter)
                except StopIteration:
                    return False
                if cpu_gate is not None:
                    cpu_gate.wait(f"dnstwist_domain {index + 1}/{len(domains)}")
                pending[executor.submit(worker, domain)] = index
                return True

            for _ in range(min(workers, len(domains))):
                submit_next()

            while pending:
                done, _ = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    pending.pop(future)
                    processed_domains += 1
                    if progress is not None:
                        progress.update(1)

                    domain, generated_urls, error = future.result()
                    if error:
                        failed_domains += 1
                        print(f"[warn] dnstwist failed for {domain}: {error}", file=sys.stderr)
                    else:
                        generated_count += len(generated_urls)
                        active_urls, inactive = filter_dns_active_urls(generated_urls, dns_workers, cpu_gate, dns_checker)
                        dns_active_count += len(active_urls)
                        dns_inactive += inactive
                        match_rows, match_summary = collect_matched_rows(
                            values=active_urls,
                            records=records,
                            threshold=threshold,
                            label=label,
                            source="dnstwist_dns_active",
                            source_file="dnstwist",
                            max_workers=match_workers,
                            cpu_gate=cpu_gate,
                        )
                        lexical_skipped += int(match_summary["skipped"])
                        for row in match_rows:
                            key = (normalize_hostname(row[DETECTED_COLUMN]), normalize_hostname(row[TARGET_COLUMN]))
                            if key in seen:
                                continue
                            seen.add(key)
                            if max_rows > 0 and len(rows) >= max_rows:
                                capped = 1
                                break
                            rows.append(row)

                    if max_rows > 0 and len(rows) >= max_rows:
                        capped = 1
                        for pending_future in pending:
                            pending_future.cancel()
                        pending.clear()
                        break
                    submit_next()

                if capped:
                    break
    finally:
        if progress is not None:
            progress.close()

    return rows, {
        "domains": len(domains),
        "domains_processed": processed_domains,
        "domain_failures": failed_domains,
        "loaded": generated_count,
        "dns_active": dns_active_count,
        "matched": len(rows),
        "skipped": dns_inactive + lexical_skipped,
        "dns_inactive": dns_inactive,
        "lexical_skipped": lexical_skipped,
        "max_rows": int(max_rows),
        "capped": capped,
    }


def dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    priority = {"phishing": 2, "unlabeled": 1}
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (normalize_hostname(row[DETECTED_COLUMN]), normalize_hostname(row[TARGET_COLUMN]))
        label = row[LABEL_COLUMN].lower()
        existing = by_key.get(key)
        if existing is None or priority.get(label, 0) > priority.get(existing[LABEL_COLUMN].lower(), 0):
            by_key[key] = row
    return list(by_key.values())


def limit_rows(rows: list[dict[str, str]], limit: int) -> tuple[list[dict[str, str]], dict[str, int]]:
    if limit <= 0 or len(rows) <= limit:
        return rows, {"limit": int(limit), "loaded": len(rows), "matched": len(rows), "skipped": 0}
    return rows[:limit], {
        "limit": int(limit),
        "loaded": len(rows),
        "matched": int(limit),
        "skipped": len(rows) - int(limit),
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "Application_ID",
        "Source of detection",
        DETECTED_COLUMN,
        TARGET_COLUMN,
        CSE_NAME_COLUMN,
        LABEL_COLUMN,
        "Date of detection (DD-MM-YYYY)",
        "Time of detection (HH-MM-SS)",
        "Remarks",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CSE-only PU candidate URL file.")
    parser.add_argument("--input", default=str(DEFAULT_LOCAL_INPUT), help="Local dataset file or folder containing CSV/XLSX candidates.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV for run_pipeline.py.")
    parser.add_argument("--local-threshold", type=float, default=0.75, help="CSE match threshold for local unlabeled rows.")
    parser.add_argument("--feed-threshold", type=float, default=0.82, help="CSE match threshold for phishing-feed rows.")
    parser.add_argument("--dnstwist-threshold", type=float, default=0.75, help="CSE match threshold for dnstwist rows.")
    parser.add_argument("--local-workers", type=int, default=4, help="Parallel workers for local candidate lexical matching.")
    parser.add_argument("--feed-workers", type=int, default=8, help="Parallel workers for OpenPhish/PhishTank lexical matching.")
    parser.add_argument("--feed-fetch-workers", type=int, default=2, help="Parallel workers for live feed downloads.")
    parser.add_argument("--no-local", action="store_true", help="Do not include local unlabeled candidate rows.")
    parser.add_argument("--fetch-feeds", action="store_true", help="Download OpenPhish and PhishTank live feeds.")
    parser.add_argument("--openphish-file", default="", help="Optional saved OpenPhish feed.txt path.")
    parser.add_argument("--phishtank-file", default="", help="Optional saved PhishTank CSV/JSON/BZ2 path.")
    parser.add_argument("--dnstwist", action="store_true", help="Generate DNS-active dnstwist variants from CSE domains.")
    parser.add_argument(
        "--dnstwist-domain",
        action="append",
        default=[],
        help="CSE domain(s) to fuzz with dnstwist. May be repeated or comma-separated. Defaults to all CSE domains.",
    )
    parser.add_argument("--dnstwist-limit", type=int, default=0, help="Max CSE domains to fuzz when --dnstwist-domain is omitted. 0 means no limit.")
    parser.add_argument("--dnstwist-command", default=default_dnstwist_command(), help="dnstwist executable or script path.")
    parser.add_argument("--dnstwist-timeout", type=int, default=120, help="Timeout in seconds for each dnstwist domain run.")
    parser.add_argument("--dnstwist-workers", type=int, default=2, help="Parallel dnstwist domain workers.")
    parser.add_argument("--dnstwist-dns-workers", type=int, default=20, help="Parallel DNS checks for generated dnstwist URLs.")
    parser.add_argument("--dnstwist-match-workers", type=int, default=8, help="Parallel lexical matching workers for DNS-active dnstwist URLs.")
    parser.add_argument(
        "--dnstwist-label",
        choices=("Unlabeled", "Phishing"),
        default="Unlabeled",
        help="Label for dnstwist rows. Use Phishing only if you have independently verified the generated URLs.",
    )
    parser.add_argument(
        "--require-dns-active",
        action="store_true",
        help="After all sources are collected, keep only rows whose detected URL passes DNS lookup.",
    )
    parser.add_argument(
        "--max-active-urls",
        type=int,
        default=DEFAULT_MAX_ACTIVE_URLS,
        help="Maximum final rows to write after DNS-active filtering. Use 0 to disable the cap.",
    )
    parser.add_argument("--dns-workers", type=int, default=20, help="Parallel DNS checks for --require-dns-active.")
    parser.add_argument("--dns-timeout", type=float, default=1.0, help="Per-DNS-query timeout in seconds for fast active checks.")
    parser.add_argument("--dns-lifetime", type=float, default=1.5, help="Total resolver lifetime in seconds for each DNS record type.")
    parser.add_argument("--dns-record-types", default="A,AAAA", help="Comma-separated DNS record types used for active checks.")
    parser.add_argument("--no-dns-cache", action="store_true", help="Disable hostname DNS active-result cache.")
    parser.add_argument(
        "--dns-socket-fallback",
        action="store_true",
        help="Use socket.getaddrinfo fallback when dnspython checks fail. Slower; off by default.",
    )
    parser.add_argument(
        "--cpu-gate-threshold",
        type=float,
        default=85.0,
        help="Strict system CPU percent threshold. New parallel work waits while CPU is above this value. Use 100 to disable.",
    )
    parser.add_argument("--cpu-gate-sample-seconds", type=float, default=0.25, help="CPU sampling window for the strict CPU gate.")
    parser.add_argument("--cpu-gate-check-interval", type=float, default=0.75, help="Seconds to wait between CPU gate checks.")
    parser.add_argument(
        "--cpu-gate-max-wait-seconds",
        type=float,
        default=600.0,
        help="Fail if CPU remains above the threshold before starting new work for this many seconds. Use 0 to wait forever.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cpu_gate = make_cpu_gate(args)
    dns_checker = make_dns_checker(args)
    records = load_cse_records()
    if not records:
        raise RuntimeError(f"No CSE records found in {CSE_FILE}.")

    rows: list[dict[str, str]] = []
    summaries: dict[str, dict[str, int | float | str]] = {}

    if not args.no_local:
        local_rows, summaries["local_unlabeled"] = collect_local_unlabeled(
            Path(args.input),
            records,
            args.local_threshold,
            args.local_workers,
            cpu_gate,
        )
        rows.extend(local_rows)

    if args.openphish_file:
        feed_rows, summaries["openphish_file"] = collect_feed_positives(
            _read_feed_file(Path(args.openphish_file)),
            records,
            args.feed_threshold,
            "openphish_file",
            args.feed_workers,
            cpu_gate,
        )
        rows.extend(feed_rows)

    if args.phishtank_file:
        feed_rows, summaries["phishtank_file"] = collect_feed_positives(
            _read_feed_file(Path(args.phishtank_file)),
            records,
            args.feed_threshold,
            "phishtank_file",
            args.feed_workers,
            cpu_gate,
        )
        rows.extend(feed_rows)

    if args.fetch_feeds:
        live_sources = [("openphish_live", fetch_openphish), ("phishtank_live", fetch_phishtank)]

        def fetch_worker(source_and_fetcher: tuple[str, Callable[[], list[str]]]) -> tuple[str, list[str], str]:
            source, fetcher = source_and_fetcher
            try:
                return source, fetcher(), ""
            except Exception as exc:
                return source, [], f"{type(exc).__name__}: {exc}"

        for source, feed_urls, error in run_parallel_gated(
            live_sources,
            fetch_worker,
            args.feed_fetch_workers,
            cpu_gate,
            "live_feed_fetch",
        ):
            if error:
                summaries[source] = {"loaded": 0, "matched": 0, "skipped": 0}
                print(f"[warn] {source} failed: {error}", file=sys.stderr)
                continue
            feed_rows, summaries[source] = collect_feed_positives(
                feed_urls,
                records,
                args.feed_threshold,
                source,
                args.feed_workers,
                cpu_gate,
            )
            rows.extend(feed_rows)

    if args.dnstwist:
        dnstwist_max_rows = args.max_active_urls
        if args.max_active_urls > 0 and rows:
            dnstwist_max_rows = max(0, args.max_active_urls - len(dedupe_rows(rows)))
        if args.max_active_urls > 0 and dnstwist_max_rows <= 0:
            summaries["dnstwist_dns_active"] = {
                "domains": 0,
                "domains_processed": 0,
                "domain_failures": 0,
                "loaded": 0,
                "dns_active": 0,
                "matched": 0,
                "skipped": 0,
                "dns_inactive": 0,
                "lexical_skipped": 0,
                "max_rows": 0,
                "capped": 1,
            }
        else:
            dnstwist_rows, summaries["dnstwist_dns_active"] = collect_dnstwist_candidates(
                records=records,
                threshold=args.dnstwist_threshold,
                command=args.dnstwist_command,
                timeout=args.dnstwist_timeout,
                dnstwist_workers=args.dnstwist_workers,
                dns_workers=args.dnstwist_dns_workers,
                match_workers=args.dnstwist_match_workers,
                requested_domains=_parse_repeated_values(args.dnstwist_domain),
                limit=args.dnstwist_limit,
                label=args.dnstwist_label,
                cpu_gate=cpu_gate,
                dns_checker=dns_checker,
                max_rows=dnstwist_max_rows,
            )
            rows.extend(dnstwist_rows)

    final_rows = dedupe_rows(rows)
    if args.require_dns_active:
        final_rows, summaries[DNS_ACTIVE_SOURCE] = filter_dns_active_rows(
            final_rows,
            args.dns_workers,
            cpu_gate,
            dns_checker,
        )
    final_rows, summaries["max_active_urls"] = limit_rows(final_rows, args.max_active_urls)
    write_rows(Path(args.output), final_rows)
    summaries["cpu_gate"] = cpu_gate.summary()
    summaries["dns_active_cache"] = dns_checker.summary()

    labels = {}
    for row in final_rows:
        labels[row[LABEL_COLUMN]] = labels.get(row[LABEL_COLUMN], 0) + 1
    print(f"Loaded CSE records: {len(records)}")
    for name, summary in summaries.items():
        print(f"{name}: {summary}")
    print(f"Output rows: {len(final_rows)}")
    print(f"Labels: {labels}")
    print(f"Written: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
