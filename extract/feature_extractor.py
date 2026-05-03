# ============================================================
# FEATURE EXTRACTOR V2
# ============================================================
#
# This module extracts evidence for the suspected-vs-phishing dataset.
# Lexical similarity only decides whether a URL is a lookalike candidate;
# the binary label is assigned later from active infrastructure/page evidence.
# ============================================================

from __future__ import annotations

import hashlib
import io
import logging
import math
import re
import socket
import ssl
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urljoin, urlparse

import tldextract
from bs4 import BeautifulSoup

try:
    from PIL import Image
except Exception:  # pragma: no cover - dependency fallback
    Image = None

IMAGE_PHASH_AVAILABLE = Image is not None

try:
    import dns.resolver
except Exception:  # pragma: no cover - dependency fallback
    dns = None

try:
    import Levenshtein
except Exception:  # pragma: no cover - exercised only when dependency is absent
    Levenshtein = None

log = logging.getLogger(__name__)

HIGH_LEXICAL_THRESHOLD = 0.75
CAMPAIGN_DOMAIN_THRESHOLD = 2
MISSING_NUMERIC = -1
TEXT_SIMHASH_HAMMING_THRESHOLD = 3
IMAGE_PHASH_HAMMING_THRESHOLD = 6
_TLD_EXTRACTOR = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=())
DEFAULT_DNS_TIMEOUT = 3.0
DEFAULT_DNS_LIFETIME = 6.0
DEFAULT_DNS_RETRIES = 1

_BRAND_STOPWORDS = {
    "and", "bank", "co", "com", "corp", "corporate", "gov", "http", "https",
    "in", "india", "limited", "login", "ltd", "net", "online", "org",
    "portal", "public", "services", "the", "www",
}

_HOMOGLYPH_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "l",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
        "\u0430": "a",  # Cyrillic
        "\u0435": "e",
        "\u043e": "o",
        "\u0440": "p",
        "\u0441": "c",
        "\u0445": "x",
        "\u0443": "y",
    }
)


@dataclass
class UrlContext:
    url: str
    detected_domain: str
    target_domain: str = ""
    cse_name: str = ""
    source_label: str = ""
    registrar_name: str = ""
    registrant_name: str = ""
    registrant_country: str = ""
    nameservers: str = ""
    hosting_ip: str = ""
    hosting_isp: str = ""
    hosting_country: str = ""
    dns_records: str = ""
    evidence_file: str = ""
    detection_date: str = ""
    detection_time: str = ""
    sandbox_verdict: str = ""
    sandbox_reason: str = ""
    application_id: str = ""
    source_of_detection: str = ""
    remarks: str = ""
    source_file: str = ""
    source_row: int = 0


@dataclass
class FetchResult:
    html: str = ""
    final_url: str = ""
    status_code: int = 0
    fetch_success: int = 0
    redirect_count: int = 0
    redirect_cross_domain_count: int = 0
    content_type: str = ""
    error: str = ""
    status: str = "unknown"


@dataclass
class DnsResult:
    dns_check_pass: int = 0
    dns_resolves_to_ip: int = 0
    resolved_ips: str = ""
    cname_exists: int = 0
    cname_chain_length: int = 0
    dns_error: str = ""
    status: str = "unknown"


@dataclass
class TlsResult:
    https_enabled: int = 0
    ssl_valid: int = 0
    cert_age_days: int = MISSING_NUMERIC
    ct_log_first_seen_days: int = MISSING_NUMERIC
    tls_error: str = ""
    status: str = "unknown"


@dataclass
class HostingResult:
    asn_org: str = ""
    status: str = "unknown"
    error: str = ""


@dataclass
class PageEvidence:
    page_render_success: int = 0
    has_login_form: int = 0
    has_password_input: int = 0
    form_action_external: int = 0
    brand_token_or_logo_present: int = 0
    logo_detected: int = 0
    logo_brand_matches_target_brand: int = 0
    logo_brand_domain_mismatch: int = 0
    visual_brand_domain_mismatch: int = 0
    favicon_hash: str = ""
    favicon_hash_matches_target_brand: int = 0
    favicon_hash_matches_known_phish: int = 0
    html_dom_hash: str = ""
    html_dom_hash_matches_known_phish: int = 0
    same_html_hash_domain_count_7d: int = 0
    status: str = "unknown"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "nat", "none", "null"}:
        return ""
    return text


def normalize_hostname(value: Any) -> str:
    text = _clean_text(value).lower()
    if not text:
        return ""
    if "://" not in text:
        text = "http://" + text
    parsed = urlparse(text)
    host = parsed.hostname or parsed.netloc or parsed.path.split("/")[0]
    host = host.strip(".").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def ensure_url(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    if text.startswith(("http://", "https://")):
        return text
    return f"http://{text}"


def registered_domain(host: Any) -> str:
    clean = normalize_hostname(host)
    if not clean:
        return ""
    ext = _TLD_EXTRACTOR(clean)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return clean


def subdomain_depth(host: Any) -> int:
    clean = normalize_hostname(host)
    ext = _TLD_EXTRACTOR(clean)
    if not ext.subdomain:
        return 0
    return len([part for part in ext.subdomain.split(".") if part])


def is_official_domain(candidate: Any, official_domains: list[str] | set[str]) -> bool:
    candidate_host = normalize_hostname(candidate)
    if not candidate_host:
        return False

    for official in official_domains:
        official_host = normalize_hostname(official)
        if not official_host:
            continue
        if candidate_host == official_host or candidate_host.endswith("." + official_host):
            return True
    return False


def parse_observation_date(value: Any) -> date:
    text = _clean_text(value)
    if not text:
        return datetime.now(timezone.utc).date()

    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            pass
    try:
        return pd_to_datetime_date(text)
    except Exception:
        return datetime.now(timezone.utc).date()


def pd_to_datetime_date(value: str) -> date:
    # Isolated helper so pandas is not a hard dependency inside the extractor.
    from pandas import to_datetime

    parsed = to_datetime(value, errors="coerce")
    if getattr(parsed, "date", None) and not getattr(parsed, "isna", lambda: False)():
        return parsed.date()
    raise ValueError(f"Could not parse date: {value}")


def _domain_tokens(domain: str) -> set[str]:
    host = normalize_hostname(domain)
    ext = _TLD_EXTRACTOR(host)
    raw = " ".join(part for part in (ext.subdomain, ext.domain) if part)
    return {
        token
        for token in re.split(r"[^a-z0-9]+", raw.lower())
        if len(token) >= 3 and token not in _BRAND_STOPWORDS
    }


def brand_tokens(target_domain: str, cse_name: str = "") -> set[str]:
    tokens = set(_domain_tokens(target_domain))
    tokens.update(
        token
        for token in re.split(r"[^a-z0-9]+", _clean_text(cse_name).lower())
        if len(token) >= 3 and token not in _BRAND_STOPWORDS
    )
    return tokens


def _levenshtein_distance(left: str, right: str) -> int:
    if Levenshtein is not None:
        return int(Levenshtein.distance(left, right))

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if Levenshtein is not None:
        return float(Levenshtein.ratio(left, right))
    distance = _levenshtein_distance(left, right)
    return 1.0 - (distance / max(len(left), len(right), 1))


def _homoglyph_normalize(value: str) -> str:
    return value.lower().translate(_HOMOGLYPH_TRANSLATION)


def _domain_stem(value: str) -> str:
    host = normalize_hostname(value)
    ext = _TLD_EXTRACTOR(host)
    return re.sub(r"[^a-z0-9]+", "", ext.domain.lower())


def _min_substring_distance(haystack: str, needle: str) -> int:
    if not haystack or not needle:
        return MISSING_NUMERIC
    if needle in haystack:
        return 0
    if len(haystack) <= len(needle):
        return _levenshtein_distance(haystack, needle)
    window_size = len(needle)
    return min(
        _levenshtein_distance(haystack[index:index + window_size], needle)
        for index in range(0, len(haystack) - window_size + 1)
    )


def compute_lexical_features(
    context: UrlContext,
    cse_domains: list[str],
) -> dict[str, Any]:
    host = normalize_hostname(context.detected_domain or context.url)
    reg_domain = registered_domain(host)
    detected_stem = _domain_stem(host)
    detected_alnum = re.sub(r"[^a-z0-9]+", "", host.lower())
    detected_homoglyph_alnum = _homoglyph_normalize(detected_alnum)
    candidates = []
    if context.target_domain:
        candidates.append(context.target_domain)
    candidates.extend(cse_domains)

    best_domain = ""
    best_score = 0.0
    best_distance = MISSING_NUMERIC
    best_homoglyph = 0.0

    seen: set[str] = set()
    for candidate in candidates:
        candidate_host = normalize_hostname(candidate)
        if not candidate_host or candidate_host in seen:
            continue
        seen.add(candidate_host)
        candidate_registered = registered_domain(candidate_host)
        candidate_stem = _domain_stem(candidate_host)
        candidate_tokens = brand_tokens(candidate_host, context.cse_name)
        token_match_score = 0.85 if any(
            token in detected_alnum or token in detected_homoglyph_alnum
            for token in candidate_tokens
        ) else 0.0
        score = max(
            _similarity(reg_domain, candidate_registered),
            _similarity(detected_stem, candidate_stem),
            token_match_score,
        )
        candidate_token_distances = []
        for token in candidate_tokens:
            candidate_token_distances.append(_min_substring_distance(detected_alnum, token))
            candidate_token_distances.append(_min_substring_distance(detected_homoglyph_alnum, token))
        distance_candidates = [
            _levenshtein_distance(reg_domain, candidate_registered),
            _levenshtein_distance(detected_stem, candidate_stem),
            *[distance for distance in candidate_token_distances if distance != MISSING_NUMERIC],
        ]
        distance = min(distance_candidates)
        homoglyph_score = max(
            _similarity(_homoglyph_normalize(reg_domain), _homoglyph_normalize(candidate_registered)),
            _similarity(_homoglyph_normalize(detected_stem), _homoglyph_normalize(candidate_stem)),
            token_match_score,
        )
        if score > best_score:
            best_domain = candidate_host
            best_score = score
            best_distance = distance
            best_homoglyph = homoglyph_score

    tokens = brand_tokens(best_domain or context.target_domain, context.cse_name)
    parsed = urlparse(ensure_url(context.url))
    detected_subdomain = _TLD_EXTRACTOR(host).subdomain.lower()
    path = parsed.path.lower()
    registered = registered_domain(host).lower()
    token_in_subdomain = int(any(token in detected_subdomain for token in tokens))
    token_in_path = int(any(token in path for token in tokens))
    token_in_registered = any(token in registered for token in tokens)

    return {
        "lexical_similarity_score": round(best_score, 6),
        "min_edit_distance_to_brand_domain": best_distance,
        "homoglyph_similarity_score": round(best_homoglyph, 6),
        "brand_token_in_subdomain": token_in_subdomain,
        "brand_token_in_path": token_in_path,
        "brand_not_in_registered_domain": int(bool(tokens) and not token_in_registered),
        "subdomain_depth": subdomain_depth(host),
        "matched_brand_domain": best_domain,
        "registered_domain": registered_domain(host),
    }


def _extract_ips(*values: str) -> list[str]:
    ips: set[str] = set()
    for value in values:
        for match in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", _clean_text(value)):
            parts = match.split(".")
            if all(part.isdigit() and 0 <= int(part) <= 255 for part in parts):
                ips.add(match)
    return sorted(ips)


def _positive_float(value: float | None, default: float) -> float:
    try:
        parsed = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return max(0.1, parsed)


def _positive_int(value: int | None, default: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _contains_timeout(errors: list[str]) -> bool:
    return any("timeout" in error.lower() for error in errors)


def lookup_dns(
    domain: str,
    hosting_ip: str = "",
    dns_records: str = "",
    timeout: float | None = None,
    lifetime: float | None = None,
    retries: int | None = None,
) -> DnsResult:
    host = normalize_hostname(domain)
    source_ips = _extract_ips(hosting_ip, dns_records)
    aliases: set[str] = set()
    ips: set[str] = set(source_ips)
    dns_errors: list[str] = []
    timeout = _positive_float(timeout, DEFAULT_DNS_TIMEOUT)
    lifetime = _positive_float(lifetime, DEFAULT_DNS_LIFETIME)
    retries = _positive_int(retries, DEFAULT_DNS_RETRIES)

    if dns is not None:
        for attempt in range(retries + 1):
            attempt_errors: list[str] = []
            resolver = dns.resolver.Resolver()
            resolver.lifetime = lifetime * (attempt + 1)
            resolver.timeout = timeout * (attempt + 1)

            for record_type in ("A", "AAAA"):
                try:
                    for answer in resolver.resolve(host, record_type):
                        ips.add(answer.to_text().strip())
                except Exception as exc:
                    attempt_errors.append(f"{record_type}:{type(exc).__name__}")

            cname_target = host
            seen_cnames: set[str] = set()
            for _ in range(10):
                try:
                    answers = resolver.resolve(cname_target, "CNAME")
                except Exception as exc:
                    attempt_errors.append(f"CNAME:{type(exc).__name__}")
                    break
                next_targets = [answer.target.to_text().strip(".").lower() for answer in answers]
                if not next_targets:
                    break
                next_target = next_targets[0]
                if next_target in seen_cnames:
                    break
                seen_cnames.add(next_target)
                aliases.add(next_target)
                cname_target = next_target

            dns_errors.extend(attempt_errors)
            if ips or aliases or not _contains_timeout(attempt_errors) or attempt >= retries:
                break
            time.sleep(min(0.2 * (attempt + 1), 1.0))

    if not ips:
        try:
            name, alias_list, address_list = socket.gethostbyname_ex(host)
            aliases.update(alias for alias in alias_list if alias and alias != name)
            ips.update(address_list)
        except Exception as exc:
            dns_errors.append(f"socket:{type(exc).__name__}")
            if not source_ips and "cname" not in _clean_text(dns_records).lower():
                return DnsResult(dns_error=";".join(dns_errors) or type(exc).__name__, status="error")

    record_text = _clean_text(dns_records).lower()
    cname_mentions = record_text.count("cname")
    cname_count = max(len(aliases), cname_mentions)
    has_dns = bool(ips or cname_count or _clean_text(dns_records))

    return DnsResult(
        dns_check_pass=int(has_dns),
        dns_resolves_to_ip=int(bool(ips)),
        resolved_ips=";".join(sorted(ips)),
        cname_exists=int(cname_count > 0),
        cname_chain_length=cname_count,
        dns_error=";".join(dns_errors),
        status="success" if has_dns else "not_found",
    )


def reputation_score(value: str, kind: str) -> int:
    text = _clean_text(value).lower()
    if not text:
        return MISSING_NUMERIC

    high = {
        "registrar": ("markmonitor", "csc", "network solutions", "tucows", "godaddy"),
        "nameserver": ("cloudflare", "akamai", "awsdns", "azure-dns", "googledomains"),
        "asn": ("amazon", "google", "microsoft", "cloudflare", "akamai", "oracle"),
    }.get(kind, ())
    low = {
        "registrar": ("namecheap", "dynadot", "spaceship", "porkbun", "nicenic"),
        "nameserver": ("namecheaphosting", "dnspod", "parking", "sedoparking"),
        "asn": ("bulletproof", "hostinger", "namecheap", "digitalocean"),
    }.get(kind, ())

    if any(token in text for token in high):
        return 8
    if any(token in text for token in low):
        return 4
    return 6


def probe_tls(domain: str, observation: date, timeout: float = 5.0) -> TlsResult:
    host = normalize_hostname(domain)
    if not host:
        return TlsResult(tls_error="missing_host", status="error")

    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
        not_before = cert.get("notBefore")
        cert_age = MISSING_NUMERIC
        if not_before:
            cert_date = parsedate_to_datetime(not_before).date()
            cert_age = max((observation - cert_date).days, 0)
        return TlsResult(https_enabled=1, ssl_valid=1, cert_age_days=cert_age, status="success")
    except ssl.SSLError as exc:
        return TlsResult(https_enabled=1, ssl_valid=0, tls_error=type(exc).__name__, status="invalid")
    except Exception as exc:
        return TlsResult(tls_error=type(exc).__name__, status="error")


def domain_age_days(domain_info: Any, observation: date) -> int:
    created = getattr(domain_info, "creation_date", None)
    if not created:
        return MISSING_NUMERIC
    try:
        if isinstance(created, datetime):
            created_date = created.date()
        else:
            created_date = created
        return max((observation - created_date).days, 0)
    except Exception:
        return MISSING_NUMERIC


def domain_expiry_days(domain_info: Any, observation: date) -> int:
    expires = getattr(domain_info, "expiration_date", None)
    if not expires:
        return MISSING_NUMERIC
    try:
        if isinstance(expires, datetime):
            expiry_date = expires.date()
        else:
            expiry_date = expires
        return (expiry_date - observation).days
    except Exception:
        return MISSING_NUMERIC


def find_favicon_url(page_url: str, html: str) -> str:
    if not html:
        return urljoin(page_url, "/favicon.ico")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all("link", href=True):
        rel = " ".join(tag.get("rel", [])).lower()
        if "icon" in rel:
            return urljoin(page_url, tag["href"])
    return urljoin(page_url, "/favicon.ico")


_HASH_BITS = 64
_PHASH_IMAGE_SIZE = 32
_PHASH_HASH_SIZE = 8
_PHASH_COSINES = [
    [
        math.cos(((2 * coordinate + 1) * frequency * math.pi) / (2 * _PHASH_IMAGE_SIZE))
        for coordinate in range(_PHASH_IMAGE_SIZE)
    ]
    for frequency in range(_PHASH_HASH_SIZE)
]


def _bits_to_hex(bits: list[int]) -> str:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def _token_hash64(token: str) -> int:
    # SimHash needs a stable per-token hash; this is not a content SHA hash.
    digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _simhash_text(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return ""

    weighted_features: dict[str, int] = {}
    for token in tokens:
        weighted_features[token] = weighted_features.get(token, 0) + 1
    for index in range(max(0, len(tokens) - 2)):
        shingle = " ".join(tokens[index:index + 3])
        weighted_features[shingle] = weighted_features.get(shingle, 0) + 2

    vector = [0] * _HASH_BITS
    for feature, weight in weighted_features.items():
        token_hash = _token_hash64(feature)
        for bit_index in range(_HASH_BITS):
            if token_hash & (1 << bit_index):
                vector[bit_index] += weight
            else:
                vector[bit_index] -= weight

    value = 0
    for bit_index, score in enumerate(vector):
        if score >= 0:
            value |= 1 << bit_index
    return f"{value:016x}"


def _image_phash(image_bytes: bytes | None) -> str:
    if not image_bytes or Image is None:
        return ""

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1)
            gray = image.convert("L").resize((_PHASH_IMAGE_SIZE, _PHASH_IMAGE_SIZE), resample)
            pixels = list(gray.getdata())
    except Exception:
        return ""

    coefficients: list[float] = []
    for u in range(_PHASH_HASH_SIZE):
        cos_u = _PHASH_COSINES[u]
        for v in range(_PHASH_HASH_SIZE):
            cos_v = _PHASH_COSINES[v]
            total = 0.0
            for y in range(_PHASH_IMAGE_SIZE):
                row = y * _PHASH_IMAGE_SIZE
                y_factor = cos_v[y]
                for x in range(_PHASH_IMAGE_SIZE):
                    total += pixels[row + x] * cos_u[x] * y_factor
            coefficients.append(total)

    comparable = coefficients[1:] or coefficients
    median = sorted(comparable)[len(comparable) // 2]
    return _bits_to_hex([int(coefficient > median) for coefficient in coefficients])


def _hex_hamming_distance(left: str, right: str) -> int | None:
    left = _clean_text(left).lower()
    right = _clean_text(right).lower()
    if not left or not right:
        return None
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return 0 if left == right else None


def _hash_matches(candidate: str, references: set[str], max_distance: int) -> bool:
    candidate = _clean_text(candidate).lower()
    if not candidate:
        return False
    for reference in references:
        reference = _clean_text(reference).lower()
        if not reference:
            continue
        if candidate == reference:
            return True
        distance = _hex_hamming_distance(candidate, reference)
        if distance is not None and distance <= max_distance:
            return True
    return False


def favicon_hash(favicon_bytes: bytes | None, favicon_url: str = "") -> str:
    return _image_phash(favicon_bytes)


def dom_hash(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if attr.startswith("data-") or attr in {"nonce", "integrity"}:
                del tag.attrs[attr]
    normalized = re.sub(r"\s+", " ", soup.get_text(" ", strip=True).lower())
    if not normalized:
        normalized = re.sub(r"\s+", " ", soup.decode().lower())
    return _simhash_text(normalized)


def analyze_page(
    context: UrlContext,
    fetch: FetchResult,
    matched_brand_domain: str,
    favicon_bytes: bytes | None = None,
    favicon_url: str = "",
    html_dom_hash: str = "",
    known_brand_favicon_hashes: dict[str, set[str]] | None = None,
    known_phish_favicon_hashes: set[str] | None = None,
    known_phish_dom_hashes: set[str] | None = None,
) -> PageEvidence:
    html = fetch.html or ""
    if not html:
        return PageEvidence(
            favicon_hash=favicon_hash(favicon_bytes, favicon_url),
            status="empty",
        )

    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text(" ", strip=True).lower()
    target_tokens = brand_tokens(matched_brand_domain or context.target_domain, context.cse_name)

    forms = soup.find_all("form")
    password = int(bool(soup.find("input", attrs={"type": re.compile("^password$", re.I)})))
    login_terms = ("login", "signin", "sign in", "password", "username", "user id", "otp")
    login_form = int(password or any(term in page_text for term in login_terms) or bool(forms))

    page_domain = registered_domain(context.detected_domain)
    external_action = 0
    for form in forms:
        action = _clean_text(form.get("action"))
        if not action or action.startswith(("#", "javascript:")):
            continue
        action_domain = registered_domain(urljoin(fetch.final_url or context.url, action))
        if action_domain and action_domain != page_domain:
            external_action = 1
            break

    logo_detected = 0
    logo_brand_match = 0
    for img in soup.find_all("img"):
        attrs = " ".join(
            _clean_text(img.get(attr))
            for attr in ("alt", "src", "class", "id", "title")
        ).lower()
        if "logo" in attrs:
            logo_detected = 1
        if target_tokens and any(token in attrs for token in target_tokens):
            logo_detected = 1
            logo_brand_match = 1

    metadata = " ".join(
        _clean_text(tag.get("content"))
        for tag in soup.find_all("meta")
        if tag.get("content")
    ).lower()
    title = _clean_text(soup.title.string if soup.title else "").lower()
    brand_text_match = int(
        bool(target_tokens)
        and any(token in text for token in target_tokens for text in (page_text, metadata, title))
    )
    # Generic "logo" filenames are common on parked pages; only a brand token or
    # target-brand logo metadata should count as phishing brand evidence.
    brand_present = int(brand_text_match or logo_brand_match)
    target_registered = registered_domain(matched_brand_domain or context.target_domain)
    visual_mismatch = int(
        brand_present
        and target_registered
        and page_domain
        and target_registered != page_domain
    )

    fav_hash = favicon_hash(favicon_bytes, favicon_url)
    brand_hashes = known_brand_favicon_hashes or {}
    target_hashes: set[str] = set()
    for key in {
        target_registered,
        normalize_hostname(matched_brand_domain or context.target_domain),
        registered_domain(context.target_domain),
        normalize_hostname(context.target_domain),
    }:
        if key:
            target_hashes.update(brand_hashes.get(key, set()))
    known_phish_favs = known_phish_favicon_hashes or set()
    html_hash = html_dom_hash or dom_hash(html)
    known_phish_doms = known_phish_dom_hashes or set()

    return PageEvidence(
        page_render_success=int(fetch.fetch_success and bool(html.strip())),
        has_login_form=login_form,
        has_password_input=password,
        form_action_external=external_action,
        brand_token_or_logo_present=brand_present,
        logo_detected=logo_detected,
        logo_brand_matches_target_brand=logo_brand_match,
        logo_brand_domain_mismatch=visual_mismatch,
        visual_brand_domain_mismatch=visual_mismatch,
        favicon_hash=fav_hash,
        favicon_hash_matches_target_brand=int(_hash_matches(fav_hash, target_hashes, IMAGE_PHASH_HAMMING_THRESHOLD)),
        favicon_hash_matches_known_phish=int(_hash_matches(fav_hash, known_phish_favs, IMAGE_PHASH_HAMMING_THRESHOLD)),
        html_dom_hash=html_hash,
        html_dom_hash_matches_known_phish=int(_hash_matches(html_hash, known_phish_doms, TEXT_SIMHASH_HAMMING_THRESHOLD)),
        status="success",
    )


def _binary_url_features(url: str, domain: str) -> dict[str, Any]:
    parsed = urlparse(ensure_url(url))
    return {
        "having_ip": int(bool(re.search(r"(([0-9]{1,3}\.){3}[0-9]{1,3})", url))),
        "have_at_symbol": int("@" in url),
        "url_length": len(url),
        "url_depth": parsed.path.count("/"),
        "redirection": int(url.rfind("//") > 7),
        "http_domain": int("http" in normalize_hostname(domain)),
        "tiny_url": int(bool(re.search(r"bit\.ly|tinyurl|goo\.gl|t\.co|ow\.ly", url, re.I))),
        "prefix_suffix": int("-" in normalize_hostname(domain)),
        "digit_count": sum(c.isdigit() for c in url),
        "special_char_count": sum(url.count(c) for c in ["?", "=", "&", "%", "-", "_", "@"]),
        "entropy": _entropy(url),
    }


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    probabilities = [float(value.count(c)) / len(value) for c in dict.fromkeys(value)]
    return -sum(p * math.log(p, 2) for p in probabilities)


def _source_label_normalized(value: str) -> str:
    text = _clean_text(value).lower()
    if "phish" in text:
        return "phishing"
    if "suspect" in text:
        return "suspected"
    if "legit" in text:
        return "legitimate"
    return text or "unknown"


def _safe_call(default: Any, func, *args, **kwargs) -> Any:
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        log.debug("Feature extraction step failed in %s: %s", getattr(func, "__name__", "unknown"), exc)
        return default


def extract_features_from_prefetch(
    context: UrlContext,
    fetch: FetchResult,
    cse_domains: list[str],
    domain_info: Any | None = None,
    dns_result: DnsResult | None = None,
    tls_result: TlsResult | None = None,
    hosting_result: HostingResult | None = None,
    favicon_bytes: bytes | None = None,
    favicon_url: str = "",
    html_dom_hash: str = "",
    ip_brand_counts: dict[str, int] | None = None,
    known_brand_favicon_hashes: dict[str, set[str]] | None = None,
    known_phish_favicon_hashes: set[str] | None = None,
    known_phish_dom_hashes: set[str] | None = None,
) -> dict[str, Any]:
    url = ensure_url(context.url)
    domain = normalize_hostname(context.detected_domain or url)
    observation = parse_observation_date(context.detection_date)
    lexical = _safe_call(
        {
            "lexical_similarity_score": 0.0,
            "min_edit_distance_to_brand_domain": MISSING_NUMERIC,
            "homoglyph_similarity_score": 0.0,
            "brand_token_in_subdomain": 0,
            "brand_token_in_path": 0,
            "brand_not_in_registered_domain": 0,
            "subdomain_depth": 0,
            "matched_brand_domain": "unknown",
            "registered_domain": registered_domain(domain) or "unknown",
        },
        compute_lexical_features,
        context,
        cse_domains,
    )
    dns = dns_result or _safe_call(
        DnsResult(status="error", dns_error="exception"),
        lookup_dns,
        domain,
        context.hosting_ip,
        context.dns_records,
    )
    tls = tls_result or _safe_call(
        TlsResult(status="error", tls_error="exception"),
        probe_tls,
        domain,
        observation,
    )
    hosting = hosting_result or HostingResult()
    page = _safe_call(
        PageEvidence(status="error"),
        analyze_page,
        context,
        fetch,
        lexical["matched_brand_domain"],
        favicon_bytes,
        favicon_url,
        html_dom_hash,
        known_brand_favicon_hashes,
        known_phish_favicon_hashes,
        known_phish_dom_hashes,
    )

    ips = [ip for ip in dns.resolved_ips.split(";") if ip]
    ip_counts = ip_brand_counts or {}
    ip_seen_many = MISSING_NUMERIC
    if ips:
        ip_seen_many = int(any(ip_counts.get(ip, 0) > 1 for ip in ips))
    nameserver_text = ";".join(getattr(domain_info, "nameservers", ()) or ()) or context.nameservers
    registrar_text = getattr(domain_info, "registrar_name", "") or context.registrar_name
    asn_text = hosting.asn_org or context.hosting_isp

    features: dict[str, Any] = {
        "url": url,
        "domain": domain,
        "target_brand_domain": normalize_hostname(context.target_domain),
        "critical_sector_entity_name": context.cse_name,
        "source_label": _source_label_normalized(context.source_label),
        "source_file": context.source_file,
        "source_row": context.source_row,
        "detection_date": context.detection_date,
        "evidence_file": context.evidence_file,
        "sandbox_verdict": context.sandbox_verdict,
        "sandbox_reason": context.sandbox_reason,
        "application_id": context.application_id,
        "source_of_detection": context.source_of_detection,
        **_binary_url_features(url, domain),
        **lexical,
        "dns_check_pass": dns.dns_check_pass,
        "dns_resolves_to_ip": dns.dns_resolves_to_ip,
        "resolved_ips": dns.resolved_ips,
        "cname_exists": dns.cname_exists,
        "cname_chain_length": dns.cname_chain_length,
        "nameserver_reputation_score": reputation_score(nameserver_text, "nameserver"),
        "passive_dns_first_seen_days": MISSING_NUMERIC,
        "domain_age_days": domain_age_days(domain_info, observation),
        "domain_expiry_days": domain_expiry_days(domain_info, observation),
        "registrar_reputation_score": reputation_score(registrar_text, "registrar"),
        "asn_reputation_score": reputation_score(asn_text, "asn"),
        "ip_seen_with_many_brands": ip_seen_many,
        "https_enabled": tls.https_enabled,
        "ssl_valid": tls.ssl_valid,
        "cert_age_days": tls.cert_age_days,
        "ct_log_first_seen_days": tls.ct_log_first_seen_days,
        "url_fetch_success": fetch.fetch_success,
        "page_fetch_success": fetch.fetch_success,
        "redirect_count": fetch.redirect_count,
        "redirect_cross_domain_count": fetch.redirect_cross_domain_count,
        "page_render_success": page.page_render_success,
        "has_login_form": page.has_login_form,
        "has_password_input": page.has_password_input,
        "form_action_external": page.form_action_external,
        "brand_token_or_logo_present": page.brand_token_or_logo_present,
        "logo_detected": page.logo_detected,
        "logo_brand_matches_target_brand": page.logo_brand_matches_target_brand,
        "logo_brand_domain_mismatch": page.logo_brand_domain_mismatch,
        "visual_brand_domain_mismatch": page.visual_brand_domain_mismatch,
        "favicon_hash": page.favicon_hash,
        "favicon_hash_matches_target_brand": page.favicon_hash_matches_target_brand,
        "favicon_hash_matches_known_phish": page.favicon_hash_matches_known_phish,
        "html_dom_hash": page.html_dom_hash,
        "html_dom_hash_matches_known_phish": page.html_dom_hash_matches_known_phish,
        "same_html_hash_domain_count_7d": page.same_html_hash_domain_count_7d,
        "fetch_status_code": fetch.status_code,
        "fetch_error": fetch.error,
        "dns_error": dns.dns_error,
        "tls_error": tls.tls_error,
        "dns_status": dns.status,
        "http_status": fetch.status,
        "tls_status": tls.status,
        "page_status": page.status,
    }
    return features


def classify_features(row: dict[str, Any]) -> dict[str, Any]:
    lexical_high = float(row.get("lexical_similarity_score") or 0.0) >= HIGH_LEXICAL_THRESHOLD
    dns_ok = int(row.get("dns_check_pass") or 0) == 1
    fetch_ok = int(row.get("page_fetch_success") or 0) == 1
    login = int(row.get("has_login_form") or 0) == 1
    password = int(row.get("has_password_input") or 0) == 1
    brand_claim = int(row.get("brand_token_or_logo_present") or 0) == 1
    visual_mismatch = int(row.get("visual_brand_domain_mismatch") or 0) == 1
    phish_favicon = int(row.get("favicon_hash_matches_known_phish") or 0) == 1
    campaign_count = int(row.get("same_html_hash_domain_count_7d") or 0)
    source_label = _source_label_normalized(str(row.get("source_label", "")))

    no_login_form = int(not login)
    no_brand_visual_claim = int(not brand_claim)
    suspected_inactive = int(lexical_high and not dns_ok and not fetch_ok)
    suspected_active_infra = int(lexical_high and dns_ok and not fetch_ok)
    suspected_active_no_phish_evidence = int(
        lexical_high and dns_ok and fetch_ok and not login and not brand_claim
    )
    phishing_campaign = int(lexical_high and dns_ok and campaign_count > CAMPAIGN_DOMAIN_THRESHOLD and password)
    phishing = int(
        lexical_high
        and dns_ok
        and (
            (password and brand_claim)
            or visual_mismatch
            or phish_favicon
        )
    )

    status = "suspected_unconfirmed"
    reason = "high lexical similarity but insufficient phishing evidence"
    label = 0
    include_in_ml = int(lexical_high)

    if not lexical_high:
        status = "below_high_lexical_threshold"
        reason = f"lexical_similarity_score < {HIGH_LEXICAL_THRESHOLD}"
        label = 1 if source_label == "phishing" else 0
    elif phishing_campaign:
        status = "phishing_campaign"
        reason = "same DOM hash appears across multiple domains in 7 days and password input exists"
        label = 1
    elif phishing:
        status = "phishing"
        if password and brand_claim:
            reason = "password input plus brand token/logo evidence"
        elif visual_mismatch:
            reason = "visual brand-domain mismatch"
        else:
            reason = "favicon hash matches known phishing hash"
        label = 1
    elif source_label == "phishing":
        status = "source_verified_phishing"
        reason = "source label is trusted historical phishing"
        label = 1
    elif suspected_inactive:
        status = "suspected_inactive"
        reason = "high lexical similarity but no DNS and no page fetch"
    elif suspected_active_infra:
        status = "suspected_active_infra"
        reason = "DNS active but page fetch failed"
    elif suspected_active_no_phish_evidence:
        status = "suspected_active_no_phish_evidence"
        reason = "active page without login form or brand visual claim"

    source_positive = source_label == "phishing"
    label_conflict = int((source_positive and label == 0) or (not source_positive and label == 1 and source_label in {"suspected", "legitimate"}))

    row["high_lexical_similarity"] = int(lexical_high)
    row["no_login_form"] = no_login_form
    row["no_brand_visual_claim"] = no_brand_visual_claim
    row["suspected_inactive"] = suspected_inactive
    row["suspected_active_infra"] = suspected_active_infra
    row["suspected_active_no_phish_evidence"] = suspected_active_no_phish_evidence
    row["phishing"] = phishing
    row["phishing_campaign"] = phishing_campaign
    row["validation_status"] = status
    row["validation_reason"] = reason
    row["label_conflict"] = label_conflict
    row["include_in_ml"] = include_in_ml
    row["label"] = label
    return row


def finalize_feature_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Populate campaign counts and labels, then split final ML rows from audit rows."""
    hash_to_observations: dict[str, list[tuple[str, date]]] = {}
    for row in rows:
        html_hash = str(row.get("html_dom_hash") or "")
        domain = str(row.get("domain") or "")
        if html_hash and domain:
            hash_to_observations.setdefault(html_hash, []).append(
                (domain, parse_observation_date(row.get("detection_date")))
            )

    finalized: list[dict[str, Any]] = []
    for row in rows:
        html_hash = str(row.get("html_dom_hash") or "")
        if html_hash:
            observation = parse_observation_date(row.get("detection_date"))
            domains_in_window = {
                domain
                for domain, seen_date in hash_to_observations.get(html_hash, [])
                if abs((observation - seen_date).days) <= 7
            }
            row["same_html_hash_domain_count_7d"] = len(domains_in_window)
        else:
            row["same_html_hash_domain_count_7d"] = 0
        if row.get("extraction_status") == "failed":
            finalized.append(row)
        else:
            finalized.append(classify_features(row))

    final_rows = [row for row in finalized if int(row.get("include_in_ml") or 0) == 1]
    return final_rows, finalized
