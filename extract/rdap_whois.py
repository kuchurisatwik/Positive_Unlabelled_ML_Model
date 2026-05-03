# ============================================================
# RDAP + WHOIS Domain Registration Lookup
# ============================================================
#
# Strategy:
#   1. RDAP (primary)  — HTTP/JSON based, highly concurrent via aiohttp.
#      IANA bootstrap maps TLDs → RDAP server URLs.
#   2. WHOIS (fallback) — Traditional socket protocol, throttled to
#      prevent IP blocking (max 2 concurrent queries).
#
# Both async and sync interfaces are provided:
#   - RDAPClient       — async, for the async pipeline
#   - lookup_domain()  — sync wrapper, for the sync feature extractor
# ============================================================

from __future__ import annotations

import asyncio
import logging
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp
import tldextract

log = logging.getLogger(__name__)
_TLD_EXTRACTOR = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=())

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DomainInfo:
    """Standardised result from RDAP or WHOIS lookup."""

    creation_date: datetime | None = None
    expiration_date: datetime | None = None
    registrant_name: str = ""
    registrar_name: str = ""
    nameservers: tuple[str, ...] = ()
    source: str = ""            # "rdap" or "whois"
    error: str | None = None
    rdap_status: str = "unknown"
    whois_status: str = "unknown"

    @property
    def age_days(self) -> int:
        """Domain age in days (creation → expiration).  -1 if unknown."""
        if self.creation_date:
            try:
                created = self.creation_date
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                return max((datetime.now(timezone.utc) - created).days, 0)
            except Exception:
                return -1
        return -1

    @property
    def expiry_days(self) -> int:
        """Days from now to expiration.  -1 if unknown."""
        if self.expiration_date:
            try:
                expires = self.expiration_date
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=timezone.utc)
                return (expires - datetime.now(timezone.utc)).days
            except Exception:
                return -1
        return -1

    @property
    def is_privacy_protected(self) -> int:
        """1 if the registrant name suggests WHOIS privacy, else 0."""
        if not self.registrant_name:
            return 0
        name_lower = self.registrant_name.lower()
        return 1 if any(k in name_lower for k in (
            "privacy", "redacted", "withheld", "not disclosed",
            "data protected", "contact privacy", "domain protection",
            "whoisguard", "domains by proxy",
        )) else 0


# ---------------------------------------------------------------------------
# RDAP Bootstrap — maps TLDs to RDAP server base URLs
# ---------------------------------------------------------------------------

_IANA_BOOTSTRAP_URL = "https://data.iana.org/rdap/dns.json"


async def _load_rdap_bootstrap(
    session: aiohttp.ClientSession,
    timeout: float = 15.0,
) -> dict[str, str]:
    """Download the IANA RDAP bootstrap and return {tld: rdap_base_url}.

    The bootstrap JSON structure is:
    {
      "services": [
        [["com", "net"], ["https://rdap.verisign.com/com/v1/"]],
        ...
      ]
    }
    """
    bootstrap: dict[str, str] = {}
    try:
        async with session.get(
            _IANA_BOOTSTRAP_URL,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                log.warning("RDAP bootstrap HTTP %d", resp.status)
                return bootstrap

            data = await resp.json(content_type=None)
            for entry in data.get("services", []):
                tlds, urls = entry[0], entry[1]
                rdap_url = urls[0].rstrip("/")
                for tld in tlds:
                    bootstrap[tld.lower()] = rdap_url

        log.info("RDAP bootstrap loaded: %d TLDs mapped", len(bootstrap))
    except Exception as exc:
        log.warning("Failed to load RDAP bootstrap: %s", exc)

    return bootstrap


# ---------------------------------------------------------------------------
# RDAP date parsing helpers
# ---------------------------------------------------------------------------

def _parse_rdap_date(value: str | None) -> datetime | None:
    """Parse an RFC 3339 / ISO 8601 date from RDAP JSON."""
    if not value:
        return None
    # Strip trailing 'Z' and any sub-second precision beyond 6 digits
    clean = value.replace("Z", "+00:00")
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(clean[:len(clean)], fmt)
        except (ValueError, IndexError):
            continue
    return None


def _extract_rdap_dates(data: dict) -> tuple[datetime | None, datetime | None]:
    """Extract creation and expiration dates from RDAP JSON response."""
    creation = None
    expiration = None

    for event in data.get("events", []):
        action = event.get("eventAction", "")
        date_str = event.get("eventDate", "")

        if action == "registration" and not creation:
            creation = _parse_rdap_date(date_str)
        elif action == "expiration" and not expiration:
            expiration = _parse_rdap_date(date_str)

    return creation, expiration


def _extract_rdap_registrant(data: dict) -> str:
    """Extract registrant name from RDAP JSON response."""
    for entity in data.get("entities", []):
        roles = entity.get("roles", [])
        if "registrant" in roles:
            # Try vCard
            vcard = entity.get("vcardArray", [])
            if len(vcard) >= 2:
                for entry in vcard[1]:
                    if entry[0] == "fn":
                        return str(entry[3])
            # Try handle as fallback
            handle = entity.get("handle", "")
            if handle:
                return handle

    return ""


def _extract_rdap_registrar(data: dict) -> str:
    """Extract registrar/provider name from RDAP JSON when available."""
    direct = data.get("registrarName") or data.get("registrar") or data.get("sponsor")
    if direct:
        return str(direct)

    for entity in data.get("entities", []):
        roles = [str(role).lower() for role in entity.get("roles", [])]
        if "registrar" not in roles:
            continue
        vcard = entity.get("vcardArray", [])
        if len(vcard) >= 2:
            for entry in vcard[1]:
                if entry[0] in {"fn", "org"} and entry[3]:
                    return str(entry[3])
        handle = entity.get("handle", "")
        if handle:
            return str(handle)
    return ""


def _extract_rdap_nameservers(data: dict) -> tuple[str, ...]:
    names: set[str] = set()
    for item in data.get("nameservers", []):
        name = item.get("ldhName") or item.get("unicodeName") or item.get("name")
        if name:
            names.add(str(name).strip(".").lower())
    return tuple(sorted(name for name in names if name))


# ---------------------------------------------------------------------------
# ASYNC RDAP CLIENT
# ---------------------------------------------------------------------------

class RDAPClient:
    """Async RDAP client with throttled WHOIS fallback.

    Usage::

        async with aiohttp.ClientSession() as session:
            client = RDAPClient(session)
            await client.init()

            info = await client.lookup("example.com")
            print(info.age_days, info.is_privacy_protected)

            client.shutdown()
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        rdap_concurrency: int = 50,
        whois_concurrency: int = 2,
        rdap_timeout: float = 10.0,
        whois_timeout: float = 15.0,
        whois_delay: float = 0.5,
        rdap_retries: int = 2,
    ):
        self._session = session
        self._bootstrap: dict[str, str] = {}
        self._rdap_sem = asyncio.Semaphore(rdap_concurrency)
        self._whois_sem = asyncio.Semaphore(whois_concurrency)
        self._rdap_timeout = rdap_timeout
        self._whois_timeout = whois_timeout
        self._whois_delay = whois_delay
        self._rdap_retries = rdap_retries
        self._whois_executor = ThreadPoolExecutor(
            max_workers=whois_concurrency,
            thread_name_prefix="whois",
        )
        self._cache: dict[str, DomainInfo] = {}
        self._inflight: dict[str, asyncio.Task[DomainInfo]] = {}
        self._initialized = False

    async def init(self) -> None:
        """Load the IANA RDAP bootstrap (call once at startup)."""
        self._bootstrap = await _load_rdap_bootstrap(
            self._session, timeout=self._rdap_timeout
        )
        self._initialized = True

    def shutdown(self) -> None:
        """Clean up the WHOIS thread pool."""
        try:
            self._whois_executor.shutdown(wait=False)
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Public lookup
    # -----------------------------------------------------------------

    async def lookup(self, domain: str) -> DomainInfo:
        """RDAP lookup with WHOIS fallback.

        1. Try RDAP (high concurrency, fast, HTTP-based)
        2. If RDAP fails → fall back to WHOIS (throttled, 2 concurrent max)
        """
        domain = domain.lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]

        if domain in self._cache:
            return self._cache[domain]
        if domain in self._inflight:
            return await self._inflight[domain]

        task = asyncio.create_task(self._lookup_uncached(domain))
        self._inflight[domain] = task
        try:
            info = await task
        finally:
            self._inflight.pop(domain, None)
        self._cache[domain] = info
        return info

    async def _lookup_uncached(self, domain: str) -> DomainInfo:
        """RDAP lookup with WHOIS fallback for a normalized, uncached domain."""

        info = await self._rdap_lookup(domain)

        if info is None or (info.creation_date is None and info.expiration_date is None):
            whois_info = await self._whois_fallback(domain)
            if whois_info is not None:
                if info is not None:
                    whois_info.rdap_status = info.rdap_status
                info = whois_info

        if info is None:
            info = DomainInfo(
                error="Both RDAP and WHOIS failed",
                rdap_status="error",
                whois_status="error",
            )
        return info

    # -----------------------------------------------------------------
    # RDAP — highly concurrent
    # -----------------------------------------------------------------

    async def _rdap_lookup(self, domain: str) -> DomainInfo | None:
        """Query the RDAP server for domain registration info."""
        if not self._initialized or not self._bootstrap:
            return DomainInfo(error="RDAP bootstrap unavailable", rdap_status="unavailable", whois_status="pending")

        # Find the RDAP base URL for this TLD
        ext = _TLD_EXTRACTOR(domain)
        tld = ext.suffix.lower()

        # Handle multi-level TLDs (e.g., co.uk → try "co.uk" then "uk")
        rdap_base = self._bootstrap.get(tld)
        if rdap_base is None and "." in tld:
            rdap_base = self._bootstrap.get(tld.split(".")[-1])
        if rdap_base is None:
            log.debug("No RDAP server for TLD '%s' (domain: %s)", tld, domain)
            return DomainInfo(error="No RDAP server", rdap_status="unavailable", whois_status="pending")

        rdap_url = f"{rdap_base}/domain/{domain}"

        last_error = ""
        for attempt in range(self._rdap_retries + 1):
            try:
                async with self._rdap_sem:
                    async with self._session.get(
                        rdap_url,
                        timeout=aiohttp.ClientTimeout(total=self._rdap_timeout),
                        headers={"Accept": "application/rdap+json"},
                    ) as resp:
                        if resp.status == 404:
                            log.debug("RDAP 404 for %s", domain)
                            return DomainInfo(error="RDAP 404", rdap_status="not_found", whois_status="pending")
                        if resp.status in (429, 500, 502, 503, 504):
                            last_error = f"RDAP HTTP {resp.status}"
                            if attempt < self._rdap_retries:
                                await asyncio.sleep(0.5 * (2 ** attempt))
                                continue
                            return DomainInfo(error=last_error, rdap_status="rate_limited" if resp.status == 429 else "error", whois_status="pending")
                        if resp.status != 200:
                            log.debug("RDAP HTTP %d for %s", resp.status, domain)
                            return DomainInfo(error=f"RDAP HTTP {resp.status}", rdap_status="error", whois_status="pending")

                        data = await resp.json(content_type=None)
                        creation, expiration = _extract_rdap_dates(data)
                        registrant = _extract_rdap_registrant(data)
                        registrar = _extract_rdap_registrar(data)
                        nameservers = _extract_rdap_nameservers(data)

                        return DomainInfo(
                            creation_date=creation,
                            expiration_date=expiration,
                            registrant_name=registrant,
                            registrar_name=registrar,
                            nameservers=nameservers,
                            source="rdap",
                            rdap_status="success",
                            whois_status="skipped",
                        )

            except asyncio.TimeoutError:
                last_error = "RDAP timeout"
            except aiohttp.ClientError as exc:
                last_error = f"RDAP client error: {type(exc).__name__}"
            except Exception as exc:
                last_error = f"RDAP unexpected error: {type(exc).__name__}"

            if attempt < self._rdap_retries:
                await asyncio.sleep(0.5 * (2 ** attempt))

        log.debug("%s for %s", last_error, domain)
        return DomainInfo(error=last_error or "RDAP error", rdap_status="error", whois_status="pending")

    # -----------------------------------------------------------------
    # WHOIS fallback — throttled to prevent IP blocking
    # -----------------------------------------------------------------

    async def _whois_fallback(self, domain: str) -> DomainInfo | None:
        """Fallback to python-whois, throttled via semaphore."""
        loop = asyncio.get_running_loop()

        try:
            async with self._whois_sem:
                await asyncio.sleep(self._whois_delay)

                info = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._whois_executor,
                        _sync_whois_lookup,
                        domain,
                        self._whois_timeout,
                    ),
                    timeout=self._whois_timeout + 5,
                )
                return info

        except asyncio.TimeoutError:
            log.debug("WHOIS timeout for %s", domain)
            return DomainInfo(error="WHOIS timeout", source="whois", rdap_status="fallback", whois_status="timeout")
        except Exception as exc:
            log.debug("WHOIS fallback error for %s: %s", domain, exc)
            return DomainInfo(error=str(exc), source="whois", rdap_status="fallback", whois_status="error")


# ---------------------------------------------------------------------------
# Sync WHOIS lookup (runs inside thread pool)
# ---------------------------------------------------------------------------

def _sync_whois_lookup(domain: str, timeout: float = 15.0) -> DomainInfo | None:
    """Blocking WHOIS lookup with timeout.

    Runs inside a thread executor; the caller manages concurrency.
    """
    try:
        import whois

        # Set socket timeout for the WHOIS query
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)

        try:
            w = whois.whois(domain)
        finally:
            socket.setdefaulttimeout(old_timeout)

        if w is None:
            return None

        creation = w.creation_date
        expiration = w.expiration_date
        registrant = str(w.registrant_name) if w.registrant_name else ""
        registrar = str(w.registrar) if getattr(w, "registrar", None) else ""
        nameservers = getattr(w, "name_servers", None) or getattr(w, "nameservers", None) or ()

        if isinstance(creation, list):
            creation = creation[0]
        if isinstance(expiration, list):
            expiration = expiration[0]
        if isinstance(nameservers, str):
            nameservers = [nameservers]

        return DomainInfo(
            creation_date=creation,
            expiration_date=expiration,
            registrant_name=registrant,
            registrar_name=registrar,
            nameservers=tuple(sorted(str(name).strip(".").lower() for name in nameservers if name)),
            source="whois",
            rdap_status="fallback",
            whois_status="success",
        )

    except socket.timeout:
        log.debug("WHOIS socket timeout for %s", domain)
        return DomainInfo(error="socket timeout", source="whois", rdap_status="fallback", whois_status="timeout")
    except ConnectionResetError:
        log.debug("WHOIS connection reset for %s", domain)
        return DomainInfo(error="connection reset", source="whois", rdap_status="fallback", whois_status="error")
    except Exception as exc:
        log.debug("WHOIS error for %s: %s", domain, type(exc).__name__)
        return DomainInfo(error=str(exc), source="whois", rdap_status="fallback", whois_status="error")
