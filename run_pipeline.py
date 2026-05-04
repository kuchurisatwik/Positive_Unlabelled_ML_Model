from __future__ import annotations

import asyncio
import csv
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import socket
from urllib.parse import urlparse

import aiohttp
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from extract.feature_extractor import (
    DnsResult,
    FetchResult,
    HostingResult,
    IMAGE_PHASH_AVAILABLE,
    TlsResult,
    UrlContext,
    DEFAULT_DNS_LIFETIME,
    DEFAULT_DNS_RETRIES,
    DEFAULT_DNS_TIMEOUT,
    brand_tokens,
    compute_lexical_features,
    dom_hash,
    ensure_url,
    extract_features_from_prefetch,
    favicon_hash,
    finalize_feature_rows,
    find_favicon_url,
    is_official_domain,
    normalize_hostname,
    probe_tls,
    registered_domain,
)
from extract.rdap_whois import RDAPClient

def simple_dns_check(domain: str) -> DnsResult:
    try:
        ip = socket.gethostbyname(domain)
        return DnsResult(
            dns_check_pass=1,
            dns_resolves_to_ip=1,
            resolved_ips=ip,
            cname_exists=0,
            cname_chain_length=0,
            dns_error="",
            status="success"
        )
    except Exception as exc:
        return DnsResult(
            dns_check_pass=0,
            dns_resolves_to_ip=0,
            dns_error=type(exc).__name__,
            status="error"
        )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

DEFAULT_DATASET_INPUT_DIR = os.path.join(SCRIPT_DIR, "input", "dataset")
DEFAULT_INPUT_DIR = DEFAULT_DATASET_INPUT_DIR

INPUT_DIR = os.environ.get("PIPELINE_INPUT_DIR", DEFAULT_INPUT_DIR)
OUTPUT_DIR = os.environ.get("PIPELINE_OUTPUT_DIR", os.path.join(SCRIPT_DIR, "output"))
OUTPUT_FILE = os.environ.get("PIPELINE_OUTPUT_FILE", os.path.join(OUTPUT_DIR, "enriched_features_v2.csv"))
OUTPUT_AUDIT_FILE = os.environ.get(
    "PIPELINE_OUTPUT_AUDIT_FILE",
    os.path.join(OUTPUT_DIR, "enriched_features_v2_audit.csv"),
)
OUTPUT_DNS_FAILED_FILE = os.environ.get(
    "PIPELINE_OUTPUT_DNS_FAILED_FILE",
    os.path.join(OUTPUT_DIR, "dns_failed_urls.csv"),
)

CSE_FILE = os.environ.get("PIPELINE_CSE_FILE", os.path.join(SCRIPT_DIR, "input", "cse", "cse_list.csv"))
DATA_DIR = os.path.join(SCRIPT_DIR, "input", "data")
FAVICON_HASH_ALGORITHM = "phash64"
DOM_HASH_ALGORITHM = "simhash64"

def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _nonnegative_int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


DEFAULT_IO_CONCURRENCY = _int_env("PIPELINE_CONCURRENCY", 12)
DNS_CONCURRENCY = _int_env("PIPELINE_DNS_CONCURRENCY", max(32, DEFAULT_IO_CONCURRENCY * 2))
HTTP_CONCURRENCY = _int_env("PIPELINE_HTTP_CONCURRENCY", DEFAULT_IO_CONCURRENCY)
RDAP_CONCURRENCY = _int_env("PIPELINE_RDAP_CONCURRENCY", max(20, DEFAULT_IO_CONCURRENCY))
WHOIS_CONCURRENCY = _int_env("PIPELINE_WHOIS_CONCURRENCY", 2)
WHOIS_DELAY = _float_env("PIPELINE_WHOIS_DELAY", 0.5)
TLS_CONCURRENCY = _int_env("PIPELINE_TLS_CONCURRENCY", max(4, DEFAULT_IO_CONCURRENCY // 2))
ASN_CONCURRENCY = _int_env("PIPELINE_ASN_CONCURRENCY", max(4, DEFAULT_IO_CONCURRENCY // 2))
THREAD_WORKERS = _int_env("PIPELINE_THREAD_WORKERS", min(4, (os.cpu_count() or 1)))
HTTP_TIMEOUT = _int_env("PIPELINE_HTTP_TIMEOUT", 10)
DNS_TIMEOUT = _float_env("PIPELINE_DNS_TIMEOUT", DEFAULT_DNS_TIMEOUT)
DNS_LIFETIME = _float_env("PIPELINE_DNS_LIFETIME", DEFAULT_DNS_LIFETIME)
DNS_RETRIES = _nonnegative_int_env("PIPELINE_DNS_RETRIES", DEFAULT_DNS_RETRIES)
CSE_MATCH_THRESHOLD = _float_env("PIPELINE_CSE_MATCH_THRESHOLD", 0.75)
REQUIRE_CSE_MATCH = os.environ.get("PIPELINE_REQUIRE_CSE_MATCH", "0").strip().lower() not in {"0", "false", "no"}
FAVICON_MAX_BYTES = 128_000


@dataclass(frozen=True)
class CseRecord:
    domain: str
    name: str = ""

OUTPUT_COLUMNS = [
    "lexical_similarity_score",
    "min_edit_distance_to_brand_domain",
    "homoglyph_similarity_score",
    "brand_token_in_subdomain",
    "brand_token_in_path",
    "brand_not_in_registered_domain",
    "subdomain_depth",
    "dns_check_pass",
    "dns_resolves_to_ip",
    "cname_exists",
    "cname_chain_length",
    "nameserver_reputation_score",
    "passive_dns_first_seen_days",
    "domain_age_days",
    "domain_expiry_days",
    "registrar_reputation_score",
    "asn_reputation_score",
    "ip_seen_with_many_brands",
    "https_enabled",
    "ssl_valid",
    "cert_age_days",
    "ct_log_first_seen_days",
    "url_fetch_success",
    "redirect_count",
    "redirect_cross_domain_count",
    "page_render_success",
    "has_login_form",
    "has_password_input",
    "form_action_external",
    "logo_detected",
    "logo_brand_matches_target_brand",
    "logo_brand_domain_mismatch",
    "favicon_hash_matches_target_brand",
    "favicon_hash_matches_known_phish",
    "html_dom_hash_matches_known_phish",
    "high_lexical_similarity",
    "page_fetch_success",
    "no_login_form",
    "no_brand_visual_claim",
    "brand_token_or_logo_present",
    "visual_brand_domain_mismatch",
    "same_html_hash_domain_count_7d",
    "suspected_inactive",
    "suspected_active_infra",
    "suspected_active_no_phish_evidence",
    "phishing",
    "phishing_campaign",
    "validation_status",
    "validation_reason",
    "extraction_status",
    "dns_status",
    "rdap_status",
    "whois_status",
    "http_status",
    "tls_status",
    "page_status",
    "label",
]

DNS_FAILED_COLUMNS = [
    "url",
    "domain",
    "target_brand_domain",
    "critical_sector_entity_name",
    "source_label",
    "source_file",
    "source_row",
    "dns_status",
    "dns_check_pass",
    "dns_resolves_to_ip",
    "resolved_ips",
    "cname_exists",
    "cname_chain_length",
    "dns_error",
    "validation_status",
    "validation_reason",
]

BOOLEAN_COLUMNS = {
    "brand_not_in_registered_domain",
    "brand_token_in_path",
    "brand_token_in_subdomain",
    "brand_token_or_logo_present",
    "cname_exists",
    "dns_check_pass",
    "dns_resolves_to_ip",
    "favicon_hash_matches_known_phish",
    "favicon_hash_matches_target_brand",
    "form_action_external",
    "has_login_form",
    "has_password_input",
    "high_lexical_similarity",
    "html_dom_hash_matches_known_phish",
    "https_enabled",
    "ip_seen_with_many_brands",
    "logo_brand_domain_mismatch",
    "logo_brand_matches_target_brand",
    "logo_detected",
    "no_brand_visual_claim",
    "no_login_form",
    "page_fetch_success",
    "page_render_success",
    "phishing",
    "phishing_campaign",
    "ssl_valid",
    "suspected_active_infra",
    "suspected_active_no_phish_evidence",
    "suspected_inactive",
    "url_fetch_success",
    "visual_brand_domain_mismatch",
}
STRING_COLUMNS = {
    "dns_status",
    "extraction_status",
    "http_status",
    "page_status",
    "rdap_status",
    "tls_status",
    "validation_reason",
    "validation_status",
    "whois_status",
}
NUMERIC_COLUMNS = set(OUTPUT_COLUMNS) - BOOLEAN_COLUMNS - STRING_COLUMNS


def _cell(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "nat", "none", "null"}:
        return ""
    return text


def normalize_domain(domain: str) -> str:
    return normalize_hostname(domain)


def _split_domain_cell(value: str) -> list[str]:
    text = _cell(value)
    if not text:
        return []
    parts = re.split(r"[\s,;|]+", text)
    domains = []
    for part in parts:
        cleaned = part.strip("()[]{}'\"")
        host = normalize_hostname(cleaned)
        if host:
            domains.append(host)
    return domains


def extract_domain_column(df: pd.DataFrame) -> pd.Series:
    for col in df.columns:
        col_clean = col.lower()
        if "domain" in col_clean or "url" in col_clean:
            return df[col]
    raise ValueError(f"No usable column found. Columns: {list(df.columns)}")


def _cse_domain_value(row: pd.Series) -> str:
    return _first(row, "domains", "domain", "Legitimate Domains", "Public URL", "CSE Domain", "Domain", "URL")


def _cse_name_value(row: pd.Series) -> str:
    return _first(row, "cse", "CSE", "Cooresponding CSE", "Application Name", "Critical Sector Entity Name")


def load_cse_records() -> list[CseRecord]:
    log.info("Loading CSE list: %s", CSE_FILE)
    if not os.path.exists(CSE_FILE):
        raise FileNotFoundError(f"CSE list not found: {CSE_FILE}")

    if CSE_FILE.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(CSE_FILE)
    else:
        df = pd.read_csv(CSE_FILE)

    records: dict[str, CseRecord] = {}
    for _, row in df.iterrows():
        name = _cse_name_value(row)
        for domain in _split_domain_cell(_cse_domain_value(row)):
            if domain and domain not in records:
                records[domain] = CseRecord(domain=domain, name=name)

    deduped = sorted(records.values(), key=lambda record: record.domain)
    log.info("CSE records loaded: %d", len(deduped))
    return deduped


def load_cse_domains(records: list[CseRecord]) -> list[str]:
    return [record.domain for record in records]


def _url_alnum(value: str) -> str:
    parsed = urlparse(ensure_url(value))
    return re.sub(r"[^a-z0-9]+", "", f"{parsed.hostname or ''} {parsed.path}".lower())


def best_cse_match(context: UrlContext, records: list[CseRecord]) -> tuple[CseRecord | None, float]:
    domains = [record.domain for record in records]
    record_by_domain = {record.domain: record for record in records}
    lexical = compute_lexical_features(context, domains)
    best_domain = lexical.get("matched_brand_domain") or ""
    best_score = max(
        float(lexical.get("lexical_similarity_score") or 0.0),
        float(lexical.get("homoglyph_similarity_score") or 0.0),
    )
    best_record = record_by_domain.get(best_domain)

    candidate_alnum = _url_alnum(context.url or context.detected_domain)
    for record in records:
        tokens = brand_tokens(record.domain, record.name)
        if tokens and any(token in candidate_alnum for token in tokens):
            token_score = 0.86
            if token_score > best_score:
                best_score = token_score
                best_record = record

    return best_record, best_score


def apply_cse_matching(contexts: list[UrlContext], records: list[CseRecord]) -> tuple[list[UrlContext], list[dict]]:
    matched_contexts: list[UrlContext] = []
    unmatched_rows: list[dict] = []

    for context in contexts:
        match, score = best_cse_match(context, records)
        if match is not None and score >= CSE_MATCH_THRESHOLD:
            if not normalize_hostname(context.target_domain):
                context.target_domain = match.domain
            if not context.cse_name:
                context.cse_name = match.name
            context.remarks = (
                f"{context.remarks};pipeline_cse_match={match.domain};pipeline_cse_match_score={score:.6f}"
                if context.remarks
                else f"pipeline_cse_match={match.domain};pipeline_cse_match_score={score:.6f}"
            )
            matched_contexts.append(context)
            continue

        unmatched_rows.append(
            {
                "url": context.url,
                "domain": context.detected_domain,
                "target_brand_domain": normalize_hostname(context.target_domain),
                "critical_sector_entity_name": context.cse_name,
                "source_label": context.source_label,
                "source_file": context.source_file,
                "source_row": context.source_row,
                "validation_status": "filtered_below_cse_match_threshold",
                "validation_reason": f"CSE match score {score:.6f} < {CSE_MATCH_THRESHOLD}",
            }
        )
        if not REQUIRE_CSE_MATCH:
            matched_contexts.append(context)

    return matched_contexts, unmatched_rows


def _first(row: pd.Series, *columns: str) -> str:
    for column in columns:
        if column in row.index:
            value = _cell(row[column])
            if value:
                return value
    return ""


def _infer_source_label(row: pd.Series, source_file: str) -> str:
    explicit = _first(row, "Phishing/Suspected Domains (i.e. Class Label)", "Class Label", "label", "source_label")
    if explicit:
        return explicit

    source_text = " ".join(
        [
            source_file,
            _first(row, "Source of detection", "source_of_detection", "source"),
        ]
    ).lower()
    if "openphish" in source_text or "phishtank" in source_text:
        return "Phishing"
    if "dnstwist" in source_text:
        return "Unlabeled"
    return ""


def _row_to_context(row: pd.Series, source_file: str, source_row: int) -> UrlContext | None:
    detected = _first(
        row,
        "Identified Phishing/Suspected Domain Name",
        "Identified Phishing Domain Name",
        "Identified Suspected Domain Name",
        "Domain Name",
        "domain",
        "url",
    )
    detected_host = normalize_hostname(detected)
    if not detected_host:
        return None

    return UrlContext(
        url=ensure_url(detected),
        detected_domain=detected_host,
        target_domain=_first(row, "Corresponding CSE Domain Name", "CSE Domain", "Legitimate Domain"),
        cse_name=_first(row, "Critical Sector Entity Name", "Cooresponding CSE", "Corresponding CSE", "Application Name", "cse"),
        source_label=_infer_source_label(row, source_file),
        registrar_name=_first(row, "Registrar Name"),
        registrant_name=_first(row, "Registrant Name or Registrant Organisation"),
        registrant_country=_first(row, "Registrant Country"),
        nameservers=_first(row, "Name Servers"),
        hosting_ip=_first(row, "Hosting IP"),
        hosting_isp=_first(row, "Hosting ISP"),
        hosting_country=_first(row, "Hosting Country"),
        dns_records=_first(row, "DNS Records (if any)", "DNS Records"),
        evidence_file=_first(row, "Evidence file name"),
        detection_date=_first(row, "Date of detection (DD-MM-YYYY)", "Detection Date"),
        detection_time=_first(row, "Time of detection (HH-MM-SS)", "Detection Time"),
        sandbox_verdict=_first(row, "Sandbox Verdict"),
        sandbox_reason=_first(row, "Sandbox Reason"),
        application_id=_first(row, "Application_ID"),
        source_of_detection=_first(row, "Source of detection"),
        remarks=_first(row, "Remarks"),
        source_file=source_file,
        source_row=source_row,
    )


def _label_priority(label: str) -> int:
    text = label.lower()
    if "phish" in text:
        return 3
    if "suspect" in text:
        return 2
    if "legit" in text:
        return 1
    return 0


def _merge_context(existing: UrlContext, incoming: UrlContext) -> UrlContext:
    if _label_priority(incoming.source_label) > _label_priority(existing.source_label):
        existing.source_label = incoming.source_label

    for field in existing.__dataclass_fields__:
        if field in {"url", "detected_domain", "source_label"}:
            continue
        if getattr(existing, field) or not getattr(incoming, field):
            continue
        setattr(existing, field, getattr(incoming, field))

    if incoming.detection_date and incoming.detection_date >= existing.detection_date:
        for field in (
            "detection_date",
            "detection_time",
            "evidence_file",
            "sandbox_verdict",
            "sandbox_reason",
            "source_file",
            "source_row",
        ):
            setattr(existing, field, getattr(incoming, field))
    return existing


def load_input_contexts(input_dir: str) -> list[UrlContext]:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(
        file_name
        for file_name in os.listdir(input_dir)
        if file_name.lower().endswith((".csv", ".xlsx", ".txt"))
    )
    if not files:
        raise FileNotFoundError(f"No .csv, .xlsx, or .txt files found in {input_dir}")

    contexts_by_key = {}
    raw_rows = 0
    for file_name in files:
        path = os.path.join(input_dir, file_name)
        try:
            if file_name.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    values = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
                df = pd.DataFrame({"url": values})
            elif file_name.lower().endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
        except Exception as exc:
            log.warning("Failed to read %s: %s", file_name, exc)
            continue

        for row_index, row in df.iterrows():
            raw_rows += 1
            context = _row_to_context(row, file_name, int(row_index) + 2)
            if context is None:
                continue
            key = (context.detected_domain, normalize_hostname(context.target_domain))
            if key in contexts_by_key:
                contexts_by_key[key] = _merge_context(contexts_by_key[key], context)
            else:
                contexts_by_key[key] = context

        log.info("Loaded %d rows from %s", len(df), file_name)

    contexts = list(contexts_by_key.values())
    log.info("Input contexts: %d unique URL/target pairs from %d raw rows", len(contexts), raw_rows)
    return contexts


def _extract_ips(value: str) -> list[str]:
    ips = []
    for match in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", value or ""):
        parts = match.split(".")
        if all(part.isdigit() and 0 <= int(part) <= 255 for part in parts):
            ips.append(match)
    return ips


def build_ip_brand_counts(contexts: list[UrlContext]) -> dict[str, int]:
    ip_to_brands = defaultdict(set)
    for context in contexts:
        brand = context.cse_name or normalize_hostname(context.target_domain) or "unknown"
        for ip in _extract_ips(context.hosting_ip + " " + context.dns_records):
            ip_to_brands[ip].add(brand)
    return {ip: len(brands) for ip, brands in ip_to_brands.items()}


def update_ip_brand_counts_from_rows(rows: list[dict]) -> dict[str, int]:
    """Build IP reuse counts from live DNS results, grouped by target brand."""
    ip_to_brands = defaultdict(set)
    for row in rows:
        brand = row.get("target_brand_domain") or row.get("critical_sector_entity_name") or "unknown"
        for ip in _extract_ips(str(row.get("resolved_ips", ""))):
            ip_to_brands[ip].add(str(brand))

    counts = {ip: len(brands) for ip, brands in ip_to_brands.items()}
    for row in rows:
        ips = _extract_ips(str(row.get("resolved_ips", "")))
        row["ip_seen_with_many_brands"] = int(any(counts.get(ip, 0) > 1 for ip in ips)) if ips else -1
    return counts


def find_official_match(candidate: str, official_domains: list[str] | set[str]) -> str:
    candidate_host = normalize_hostname(candidate)
    for official in official_domains:
        official_host = normalize_hostname(official)
        if not official_host:
            continue
        if candidate_host == official_host:
            return official_host
    return ""


def partition_official_contexts(
    contexts: list[UrlContext],
    cse_domains: list[str],
) -> tuple[list[UrlContext], list[dict]]:
    official_domains = sorted(
        {domain for domain in cse_domains if domain}
        | {normalize_hostname(ctx.target_domain) for ctx in contexts if ctx.target_domain}
    )
    candidates = []
    filtered = []
    for context in contexts:
        exact_official = bool(find_official_match(context.detected_domain, official_domains))
        if exact_official:
            filtered.append(
                {
                    "url": context.url,
                    "domain": context.detected_domain,
                    "target_brand_domain": normalize_hostname(context.target_domain),
                    "matched_official_domain": find_official_match(context.detected_domain, official_domains),
                    "source_label": context.source_label,
                    "source_file": context.source_file,
                    "source_row": context.source_row,
                    "validation_status": "filtered_official_domain",
                    "validation_reason": "exact official domain",
                }
            )
        else:
            candidates.append(context)
    return candidates, filtered


def _dns_failed_row(context: UrlContext, dns_result: DnsResult) -> dict:
    return {
        "url": context.url,
        "domain": context.detected_domain,
        "target_brand_domain": normalize_hostname(context.target_domain),
        "critical_sector_entity_name": context.cse_name,
        "source_label": context.source_label,
        "source_file": context.source_file,
        "source_row": context.source_row,
        "dns_status": dns_result.status,
        "dns_check_pass": dns_result.dns_check_pass,
        "dns_resolves_to_ip": dns_result.dns_resolves_to_ip,
        "resolved_ips": dns_result.resolved_ips,
        "cname_exists": dns_result.cname_exists,
        "cname_chain_length": dns_result.cname_chain_length,
        "dns_error": dns_result.dns_error,
        "validation_status": "dns_failed_skipped",
        "validation_reason": "DNS failed before feature extraction; excluded from ML input",
    }


def _dns_failure_bucket(row: dict) -> str:
    error = str(row.get("dns_error") or row.get("dns_status") or "unknown")
    lowered = error.lower()
    if "nxdomain" in lowered:
        return "nxdomain"
    if "timeout" in lowered:
        return "timeout"
    if "nonameservers" in lowered:
        return "no_nameservers"
    if "noanswer" in lowered:
        return "no_answer"
    if "gaierror" in lowered:
        return "socket_gaierror"
    return error[:80] or "unknown"


def _format_counts(counter: Counter, limit: int = 8) -> str:
    return "; ".join(f"{name}={count}" for name, count in counter.most_common(limit)) or "none"


def prefilter_dns_active_contexts(contexts: list[UrlContext]) -> tuple[list[UrlContext], list[dict]]:
    """Keep DNS-active candidates and write DNS-failed candidates to a review file."""
    if not contexts:
        return [], []

    workers = min(DNS_CONCURRENCY, max(1, len(contexts)))
    results: list[tuple[int, UrlContext, DnsResult]] = []
    log.info(
        "DNS precheck settings: workers=%d timeout=%.1fs lifetime=%.1fs retries=%d",
        workers,
        DNS_TIMEOUT,
        DNS_LIFETIME,
        DNS_RETRIES,
    )

    def check(context: UrlContext) -> DnsResult:
        return simple_dns_check(context.detected_domain)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {
            executor.submit(check, context): (index, context)
            for index, context in enumerate(contexts)
        }
        with tqdm(total=len(contexts), desc="DNS precheck", unit="url") as pbar:
            for future in as_completed(future_to_item):
                index, context = future_to_item[future]
                try:
                    dns_result = future.result()
                except Exception as exc:
                    dns_result = DnsResult(
                        dns_error=type(exc).__name__,
                        status="error",
                    )
                results.append((index, context, dns_result))
                pbar.update(1)

    active_contexts: list[UrlContext] = []
    dns_failed_rows: list[dict] = []
    source_counts: dict[str, Counter] = defaultdict(Counter)
    for _, context, dns_result in sorted(results, key=lambda item: item[0]):
        if int(dns_result.dns_check_pass or 0) == 1:
            active_contexts.append(context)
            source_counts[context.source_file or "unknown"]["active"] += 1
        else:
            failed_row = _dns_failed_row(context, dns_result)
            dns_failed_rows.append(failed_row)
            source_counts[context.source_file or "unknown"]["skipped"] += 1

    if source_counts:
        summary = [
            f"{source}: active={counts.get('active', 0)} skipped={counts.get('skipped', 0)}"
            for source, counts in sorted(source_counts.items())
        ]
        log.info("DNS precheck by source: %s", "; ".join(summary))
    if dns_failed_rows:
        bucket_counts = Counter(_dns_failure_bucket(row) for row in dns_failed_rows)
        exact_counts = Counter(row.get("dns_error") or row.get("dns_status") or "unknown" for row in dns_failed_rows)
        log.info("DNS precheck failure buckets: %s", _format_counts(bucket_counts))
        log.info("DNS precheck top errors: %s", _format_counts(exact_counts, limit=5))

    return active_contexts, dns_failed_rows


def write_csv(path: str, rows: list[dict], columns: list[str] | None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if columns is None:
        columns = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key in seen:
                    continue
                columns.append(key)
                seen.add(key)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(enforce_schema(row) if columns == OUTPUT_COLUMNS else row)


def _load_hash_set(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as file:
        return {line.strip().lower() for line in file if line.strip()}


def load_known_hashes() -> tuple[dict[str, set[str]], set[str], set[str]]:
    phish_favicons = _load_hash_set(os.path.join(DATA_DIR, "known_phish_favicon_hashes.txt"))
    phish_doms = _load_hash_set(os.path.join(DATA_DIR, "known_phish_dom_hashes.txt"))
    brand_favicons = defaultdict(set)
    brand_path = os.path.join(DATA_DIR, "brand_favicon_hashes.csv")
    if os.path.exists(brand_path):
        with open(brand_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                raw_domain = row.get("domain", "")
                domain = registered_domain(raw_domain)
                host = normalize_hostname(raw_domain)
                fav_hash = (row.get("favicon_hash") or "").strip().lower()
                if domain and fav_hash:
                    brand_favicons[domain].add(fav_hash)
                if host and fav_hash:
                    brand_favicons[host].add(fav_hash)
    return dict(brand_favicons), phish_favicons, phish_doms


def load_ip_asn_cache() -> dict[str, str]:
    path = os.path.join(DATA_DIR, "ip_asn_cache.csv")
    if not os.path.exists(path):
        return {}
    cache = {}
    with open(path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ip = (row.get("ip") or "").strip()
            org = (
                row.get("asn_org")
                or row.get("organization")
                or row.get("org")
                or row.get("network_name")
                or row.get("name")
                or ""
            ).strip()
            if ip and org:
                cache[ip] = org
    return cache


def _reference_file_uses_algorithm(path: str, expected_algorithm: str, hash_column: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if "hash_algorithm" not in (reader.fieldnames or []):
                return False
            return any(
                row.get("hash_algorithm") == expected_algorithm
                and bool((row.get(hash_column) or "").strip())
                for row in reader
            )
    except Exception:
        return False


def cse_reference_files_exist(data_dir: str = DATA_DIR) -> bool:
    return _reference_file_uses_algorithm(
        os.path.join(data_dir, "brand_favicon_hashes.csv"),
        FAVICON_HASH_ALGORITHM,
        "favicon_hash",
    ) and _reference_file_uses_algorithm(
        os.path.join(data_dir, "brand_dom_hashes.csv"),
        DOM_HASH_ALGORITHM,
        "html_dom_hash",
    )


async def extract_cse_reference_hashes(
    cse_domains: list[str],
    data_dir: str = DATA_DIR,
    force: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Fetch official CSE pages once and write brand favicon/DOM references."""
    if not IMAGE_PHASH_AVAILABLE:
        raise RuntimeError(
            "Pillow is required to compute phash64 favicon hashes. "
            "Install dependencies with: pip install -r requirements.txt"
        )

    os.makedirs(data_dir, exist_ok=True)
    favicon_path = os.path.join(data_dir, "brand_favicon_hashes.csv")
    dom_path = os.path.join(data_dir, "brand_dom_hashes.csv")
    if not force and cse_reference_files_exist(data_dir):
        return [], []

    domains = sorted({normalize_hostname(domain) for domain in cse_domains if normalize_hostname(domain)})
    favicon_rows = []
    dom_rows = []
    http_sem = asyncio.Semaphore(min(HTTP_CONCURRENCY, 25))
    connector = aiohttp.TCPConnector(limit=min(HTTP_CONCURRENCY, 25), ttl_dns_cache=300)

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT, connect=5),
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
    ) as session:
        async def process_domain(domain: str) -> tuple[dict, dict]:
            fetch = await fetch_html(session, f"https://{domain}", http_sem)
            if not fetch.fetch_success:
                fetch = await fetch_html(session, f"http://{domain}", http_sem)
            fav_url = find_favicon_url(fetch.final_url or f"https://{domain}", fetch.html)
            fav_bytes = await fetch_binary(session, fav_url, http_sem)
            fav_hash = favicon_hash(fav_bytes, "")
            page_hash = await asyncio.to_thread(dom_hash, fetch.html)
            status = fetch.status if fetch.fetch_success else "fetch_failed"
            return (
                {
                    "domain": domain,
                    "favicon_hash": fav_hash,
                    "hash_algorithm": FAVICON_HASH_ALGORITHM,
                    "status": "success" if fav_hash else ("image_hash_failed" if fetch.fetch_success else status),
                    "final_url": fetch.final_url or "unknown",
                },
                {
                    "domain": domain,
                    "html_dom_hash": page_hash,
                    "hash_algorithm": DOM_HASH_ALGORITHM,
                    "status": "success" if page_hash else status,
                    "final_url": fetch.final_url or "unknown",
                },
            )

        results = await asyncio.gather(*(process_domain(domain) for domain in domains))

    for fav_row, dom_row in results:
        favicon_rows.append(fav_row)
        dom_rows.append(dom_row)

    write_csv(favicon_path, favicon_rows, ["domain", "favicon_hash", "hash_algorithm", "status", "final_url"])
    write_csv(dom_path, dom_rows, ["domain", "html_dom_hash", "hash_algorithm", "status", "final_url"])
    return favicon_rows, dom_rows


def ensure_cse_reference_hashes(cse_domains: list[str]) -> None:
    if cse_reference_files_exist(DATA_DIR):
        return
    log.info("CSE hash reference files missing or not phash64/simhash64; extracting official references...")
    favicon_rows, dom_rows = asyncio.run(extract_cse_reference_hashes(cse_domains))
    log.info("CSE references written: %d favicon rows, %d DOM rows", len(favicon_rows), len(dom_rows))


def _cross_domain_redirect_count(urls: list[str]) -> int:
    count = 0
    previous = ""
    for url in urls:
        current = registered_domain(url)
        if previous and current and current != previous:
            count += 1
        previous = current
    return count


def _origin(url: str) -> str:
    parsed = urlparse(ensure_url(url))
    if not parsed.hostname:
        return "unknown"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def _stage_status(status: str) -> str:
    return status or "unknown"


def _first_resolved_ip(dns_result: DnsResult) -> str:
    for ip in (dns_result.resolved_ips or "").split(";"):
        clean = ip.strip()
        if clean:
            return clean
    return ""


async def lookup_ip_asn(
    session: aiohttp.ClientSession,
    ip: str,
    ip_asn_cache: dict[str, str] | None = None,
    asn_sem: asyncio.Semaphore | None = None,
) -> HostingResult:
    if not ip:
        return HostingResult(status="unknown")
    cache = ip_asn_cache or {}
    if ip in cache:
        return HostingResult(asn_org=cache[ip], status="cache")

    sem = asn_sem or asyncio.Semaphore(1_000_000)
    try:
        async with sem:
            async with session.get(
                f"https://rdap.org/ip/{ip}",
                timeout=aiohttp.ClientTimeout(total=8),
                headers={"Accept": "application/rdap+json"},
            ) as resp:
                if resp.status != 200:
                    return HostingResult(status=f"http_{resp.status}", error=f"HTTP {resp.status}")
                data = await resp.json(content_type=None)
    except Exception as exc:
        return HostingResult(status="error", error=type(exc).__name__)

    org = data.get("name") or data.get("handle") or data.get("port43") or data.get("country") or ""
    for entity in data.get("entities", []):
        roles = [str(role).lower() for role in entity.get("roles", [])]
        if roles and not ({"registrant", "administrative", "technical", "abuse"} & set(roles)):
            continue
        vcard = entity.get("vcardArray", [])
        if len(vcard) >= 2:
            for entry in vcard[1]:
                if entry[0] in {"fn", "org"} and entry[3]:
                    org = str(entry[3])
                    break
        if org:
            break
    return HostingResult(asn_org=str(org), status="success" if org else "not_found")


def enforce_schema(row: dict) -> dict:
    """Return a row with every output column present and safe defaults."""
    enforced = {}
    for column in OUTPUT_COLUMNS:
        value = row.get(column)
        if value is None or value == "":
            if column in BOOLEAN_COLUMNS:
                value = False
            elif column in STRING_COLUMNS:
                value = "unknown"
            else:
                value = -1
        enforced[column] = value
    return enforced


def _empty_feature_row(status: str) -> dict:
    row = enforce_schema({})
    row["extraction_status"] = status
    row["validation_status"] = "extraction_failed"
    row["validation_reason"] = "row-level extraction failed"
    row["label"] = 0
    return row


def order_indexed_rows(indexed_results: list[tuple[int, dict]]) -> list[dict]:
    return [row for _, row in sorted(indexed_results, key=lambda item: item[0])]


async def fetch_html(
    session: aiohttp.ClientSession,
    url: str,
    http_sem: asyncio.Semaphore | None = None,
) -> FetchResult:
    sem = http_sem or asyncio.Semaphore(1_000_000)
    try:
        async with sem:
            async with session.get(
                ensure_url(url),
                timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT),
                allow_redirects=True,
                ssl=False,
            ) as resp:
                html = await resp.text(errors="replace")
                history_urls = [str(item.url) for item in resp.history] + [str(resp.url)]
                ok = 200 <= resp.status < 400
                return FetchResult(
                    html=html,
                    final_url=str(resp.url),
                    status_code=resp.status,
                    fetch_success=int(ok and bool(html.strip())),
                    redirect_count=len(resp.history),
                    redirect_cross_domain_count=_cross_domain_redirect_count(history_urls),
                    content_type=resp.headers.get("content-type", ""),
                    status="success" if ok else f"http_{resp.status}",
                )
    except Exception as exc:
        return FetchResult(final_url=ensure_url(url), error=type(exc).__name__, status="error")


async def fetch_binary(
    session: aiohttp.ClientSession,
    url: str,
    http_sem: asyncio.Semaphore | None = None,
    max_bytes: int = FAVICON_MAX_BYTES,
) -> bytes:
    if not url:
        return b""
    sem = http_sem or asyncio.Semaphore(1_000_000)
    try:
        async with sem:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=5),
                allow_redirects=True,
                ssl=False,
            ) as resp:
                if resp.status != 200:
                    return b""
                data = await resp.content.read(max_bytes + 1)
                return data[:max_bytes]
    except Exception:
        return b""


class AsyncFeatureCache:
    """Per-run async stage cache for network and blocking feature sources."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        rdap_client: RDAPClient,
        dns_sem: asyncio.Semaphore,
        http_sem: asyncio.Semaphore,
        tls_sem: asyncio.Semaphore,
        asn_sem: asyncio.Semaphore | None = None,
        ip_asn_cache: dict[str, str] | None = None,
    ) -> None:
        self.session = session
        self.rdap_client = rdap_client
        self.dns_sem = dns_sem
        self.http_sem = http_sem
        self.tls_sem = tls_sem
        self.asn_sem = asn_sem or asyncio.Semaphore(ASN_CONCURRENCY)
        self.ip_asn_cache = ip_asn_cache or {}
        self.dns_tasks = {}
        self.rdap_tasks = {}
        self.tls_tasks = {}
        self.asn_tasks = {}
        self.http_tasks = {}
        self.favicon_tasks = {}
        self.dom_hash_tasks = {}

    def dns(self, context: UrlContext) -> asyncio.Task[DnsResult]:
        key = normalize_hostname(context.detected_domain)
        if key not in self.dns_tasks:
            self.dns_tasks[key] = asyncio.create_task(self._dns(context))
        return self.dns_tasks[key]

    async def _dns(self, context: UrlContext) -> DnsResult:
        try:
            async with self.dns_sem:
                return await asyncio.to_thread(
                    simple_dns_check,
                    context.detected_domain
                )
        except Exception as exc:
            return DnsResult(dns_error=type(exc).__name__, status="error")

    def registration(self, context: UrlContext) -> asyncio.Task:
        key = registered_domain(context.detected_domain) or normalize_hostname(context.detected_domain)
        if key not in self.rdap_tasks:
            self.rdap_tasks[key] = asyncio.create_task(self.rdap_client.lookup(key))
        return self.rdap_tasks[key]

    def tls(self, context: UrlContext) -> asyncio.Task[TlsResult]:
        key = normalize_hostname(context.detected_domain)
        if key not in self.tls_tasks:
            self.tls_tasks[key] = asyncio.create_task(self._tls(context))
        return self.tls_tasks[key]

    async def _tls(self, context: UrlContext) -> TlsResult:
        try:
            async with self.tls_sem:
                return await asyncio.to_thread(
                    probe_tls,
                    context.detected_domain,
                    pd.Timestamp.utcnow().date(),
                )
        except Exception as exc:
            return TlsResult(tls_error=type(exc).__name__, status="error")

    def http(self, url: str) -> asyncio.Task[FetchResult]:
        key = ensure_url(url)
        if key not in self.http_tasks:
            self.http_tasks[key] = asyncio.create_task(fetch_html(self.session, key, self.http_sem))
        return self.http_tasks[key]

    def favicon(self, page_url: str, html: str) -> asyncio.Task[tuple[str, bytes]]:
        origin = _origin(page_url)
        if origin not in self.favicon_tasks:
            favicon_url = find_favicon_url(page_url, html)
            self.favicon_tasks[origin] = asyncio.create_task(self._favicon(favicon_url))
        return self.favicon_tasks[origin]

    async def _favicon(self, favicon_url: str) -> tuple[str, bytes]:
        return favicon_url, await fetch_binary(self.session, favicon_url, self.http_sem)

    def html_dom_hash(self, final_url: str, html: str) -> asyncio.Task[str]:
        key = ensure_url(final_url)
        if key not in self.dom_hash_tasks:
            self.dom_hash_tasks[key] = asyncio.create_task(asyncio.to_thread(dom_hash, html))
        return self.dom_hash_tasks[key]

    def asn(self, dns_result: DnsResult) -> asyncio.Task[HostingResult]:
        ip = _first_resolved_ip(dns_result)
        if not ip:
            return asyncio.create_task(_return_hosting_unknown())
        if ip not in self.asn_tasks:
            self.asn_tasks[ip] = asyncio.create_task(
                lookup_ip_asn(self.session, ip, self.ip_asn_cache, self.asn_sem)
            )
        return self.asn_tasks[ip]


async def _return_hosting_unknown() -> HostingResult:
    return HostingResult(status="unknown")


async def process_context(
    index: int,
    context: UrlContext,
    cache: AsyncFeatureCache,
    executor: ThreadPoolExecutor,
    cse_domains: list[str],
    ip_brand_counts: dict[str, int],
    known_brand_favicon_hashes: dict[str, set[str]],
    known_phish_favicon_hashes: set[str],
    known_phish_dom_hashes: set[str],
) -> tuple[int, dict]:
    try:
        dns_task = cache.dns(context)
        rdap_task = cache.registration(context)
        tls_task = cache.tls(context)
        fetch = await cache.http(context.url)
        favicon_url, favicon_bytes = await cache.favicon(fetch.final_url or context.url, fetch.html)
        html_hash = await cache.html_dom_hash(fetch.final_url or context.url, fetch.html)
        dns_result = await dns_task
        asn_task = cache.asn(dns_result)
        domain_info, tls_result, hosting_result = await asyncio.gather(rdap_task, tls_task, asn_task)

        loop = asyncio.get_running_loop()
        row = await loop.run_in_executor(
            executor,
            extract_features_from_prefetch,
            context,
            fetch,
            cse_domains,
            domain_info,
            dns_result,
            tls_result,
            hosting_result,
            favicon_bytes,
            favicon_url,
            html_hash,
            ip_brand_counts,
            known_brand_favicon_hashes,
            known_phish_favicon_hashes,
            known_phish_dom_hashes,
        )
        row["rdap_status"] = _stage_status(getattr(domain_info, "rdap_status", "unknown"))
        row["whois_status"] = _stage_status(getattr(domain_info, "whois_status", "unknown"))
        stage_statuses = [
            row.get("dns_status"),
            row.get("rdap_status"),
            row.get("whois_status"),
            row.get("http_status"),
            row.get("tls_status"),
            row.get("page_status"),
        ]
        row["extraction_status"] = "complete" if all(
            status in {"success", "skipped"} for status in stage_statuses
        ) else "partial"
        return index, row
    except Exception as exc:
        log.error("Row extraction failed for %s: %s", context.detected_domain, exc)
        return index, _empty_feature_row("failed")


async def run_pipeline_async(
    contexts: list[UrlContext],
    cse_domains: list[str],
    ip_brand_counts: dict[str, int],
    known_brand_favicon_hashes: dict[str, set[str]],
    known_phish_favicon_hashes: set[str],
    known_phish_dom_hashes: set[str],
) -> list[dict]:
    total = len(contexts)
    log.info(
        (
            "Starting staged v2 pipeline: %d candidates, http=%d dns=%d "
            "dns_timeout=%.1fs dns_lifetime=%.1fs dns_retries=%d "
            "rdap=%d whois=%d tls=%d threads=%d"
        ),
        total,
        HTTP_CONCURRENCY,
        DNS_CONCURRENCY,
        DNS_TIMEOUT,
        DNS_LIFETIME,
        DNS_RETRIES,
        RDAP_CONCURRENCY,
        WHOIS_CONCURRENCY,
        TLS_CONCURRENCY,
        THREAD_WORKERS,
    )

    executor = ThreadPoolExecutor(max_workers=THREAD_WORKERS)
    connector = aiohttp.TCPConnector(
        limit=HTTP_CONCURRENCY + RDAP_CONCURRENCY,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT, connect=5)
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
    ) as session:
        rdap_client = RDAPClient(
            session,
            rdap_concurrency=RDAP_CONCURRENCY,
            whois_concurrency=WHOIS_CONCURRENCY,
            rdap_timeout=10.0,
            whois_timeout=15.0,
            whois_delay=WHOIS_DELAY,
        )
        await rdap_client.init()
        cache = AsyncFeatureCache(
            session=session,
            rdap_client=rdap_client,
            dns_sem=asyncio.Semaphore(DNS_CONCURRENCY),
            http_sem=asyncio.Semaphore(HTTP_CONCURRENCY),
            tls_sem=asyncio.Semaphore(TLS_CONCURRENCY),
            asn_sem=asyncio.Semaphore(ASN_CONCURRENCY),
            ip_asn_cache=load_ip_asn_cache(),
        )

        async def tracked(index: int, context: UrlContext, pbar: tqdm) -> tuple[int, dict]:
            try:
                return await process_context(
                    index,
                    context,
                    cache,
                    executor,
                    cse_domains,
                    ip_brand_counts,
                    known_brand_favicon_hashes,
                    known_phish_favicon_hashes,
                    known_phish_dom_hashes,
                )
            finally:
                pbar.update(1)

        with tqdm(total=total, desc="Processing URLs", unit="url") as pbar:
            tasks = [asyncio.create_task(tracked(index, context, pbar)) for index, context in enumerate(contexts)]
            indexed_results = await asyncio.gather(*tasks)

        rdap_client.shutdown()

    executor.shutdown(wait=False)
    return [row for _, row in sorted(indexed_results, key=lambda item: item[0])]


def main() -> None:
    start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("Pipeline input directory: %s", INPUT_DIR)
    log.info("Pipeline output directory: %s", OUTPUT_DIR)

    cse_records = load_cse_records()
    cse_domains = load_cse_domains(cse_records)
    ensure_cse_reference_hashes(cse_domains)
    contexts = load_input_contexts(INPUT_DIR)
    contexts, cse_unmatched = apply_cse_matching(contexts, cse_records)
    log.info("CSE-matched contexts: %d", len(contexts))
    if cse_unmatched:
        log.info("Rows below CSE match threshold: %d", len(cse_unmatched))
    candidates, official_filtered = partition_official_contexts(contexts, cse_domains)
    log.info("Official-domain rows filtered from final output: %d", len(official_filtered))

    if not candidates:
        write_csv(OUTPUT_FILE, [], OUTPUT_COLUMNS)
        write_csv(OUTPUT_AUDIT_FILE, [], OUTPUT_COLUMNS)
        write_csv(OUTPUT_DNS_FAILED_FILE, cse_unmatched, DNS_FAILED_COLUMNS)
        log.info("No non-official candidate URLs found.")
        return

    candidates, dns_failed_rows = prefilter_dns_active_contexts(candidates)
    write_csv(OUTPUT_DNS_FAILED_FILE, [*cse_unmatched, *dns_failed_rows], DNS_FAILED_COLUMNS)
    log.info("DNS precheck skipped from feature extraction: %d", len(dns_failed_rows))

    if not candidates:
        write_csv(OUTPUT_FILE, [], OUTPUT_COLUMNS)
        write_csv(OUTPUT_AUDIT_FILE, [], OUTPUT_COLUMNS)
        log.info("No DNS-active candidate URLs left for feature extraction.")
        return

    ip_brand_counts = build_ip_brand_counts(candidates)
    known_brand_favicon_hashes, known_phish_favicon_hashes, known_phish_dom_hashes = load_known_hashes()
    raw_rows = asyncio.run(
        run_pipeline_async(
            candidates,
            cse_domains,
            ip_brand_counts,
            known_brand_favicon_hashes,
            known_phish_favicon_hashes,
            known_phish_dom_hashes,
        )
    )

    update_ip_brand_counts_from_rows(raw_rows)
    _, finalized_rows = finalize_feature_rows(raw_rows)
    write_csv(OUTPUT_FILE, finalized_rows, OUTPUT_COLUMNS)
    write_csv(OUTPUT_AUDIT_FILE, finalized_rows, None)

    elapsed = time.time() - start
    print(f"\n{'=' * 50}")
    print("Done!")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Audit:  {OUTPUT_AUDIT_FILE}")
    print(f"DNS failed: {OUTPUT_DNS_FAILED_FILE}")
    print(f"Rows:   {len(finalized_rows)}")
    print(f"Time:   {elapsed / 60:.2f} minutes ({elapsed:.0f}s)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
