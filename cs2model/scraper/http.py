"""Polite HTTP client: plain `requests`, per-host throttling, retries with backoff, disk cache.

No browser impersonation or anti-bot workarounds are used. If a source answers 401/403
(access control), we raise `AccessDenied` and the caller skips that source. 429/5xx/timeouts are
retried a bounded number of times with jittered exponential backoff (honouring Retry-After), then
raise `UpstreamUnavailable`; an optional per-source `CircuitBreaker` stops calls during long outages.
"""
import email.utils
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path

import requests

ROOT = Path(os.environ.get("CS2_HOME") or Path(__file__).resolve().parents[2])
CACHE_DIR = Path(os.environ.get("CS2_CACHE_DIR", ROOT / "data" / "cache"))
USER_AGENT = os.environ.get("CS2_USER_AGENT", "cs2-props-model/0.1 (personal research)")

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
_last_hit: dict[str, float] = {}


log = logging.getLogger(__name__)
SAFE_HEADERS = {"date", "server", "content-type", "retry-after", "cf-ray", "cf-cache-status",
                "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"}


class AccessDenied(Exception):
    pass


class UpstreamUnavailable(Exception):
    """Temporary upstream failure (429, 5xx, timeout, connection error) after bounded retries."""

    def __init__(self, msg, status=None, retry_after=None, detail=None):
        super().__init__(msg)
        self.status, self.retry_after, self.detail = status, retry_after, detail


class CircuitOpen(UpstreamUnavailable):
    """Raised without a request while a source's circuit breaker is open."""


class CircuitBreaker:
    """Opens after `threshold` consecutive failed fetches; while open no requests are sent. After the cooldown one trial
    request is let through: success closes it, failure reopens it with a doubled cooldown (capped)."""

    def __init__(self, threshold=2, cooldown=300.0, max_cooldown=1800.0):
        self.threshold, self.cooldown, self.max_cooldown = threshold, cooldown, max_cooldown
        self.failures, self.n_open, self.open_until, self.last_error = 0, 0, 0.0, None

    def allow(self, now=None):
        return (now or time.time()) >= self.open_until

    def remaining(self, now=None):
        return max(0.0, self.open_until - (now or time.time()))

    def success(self):
        self.failures, self.n_open, self.open_until, self.last_error = 0, 0, 0.0, None

    def failure(self, err, retry_after=None, hard=False, now=None):
        self.failures += 1
        self.last_error = err
        if hard or self.failures >= self.threshold:
            cd = min(self.max_cooldown, self.cooldown * 2 ** self.n_open)
            if retry_after:
                cd = max(cd, min(retry_after, self.max_cooldown))
            self.open_until = (now or time.time()) + cd
            self.n_open += 1


def retry_after_seconds(value, now=None):
    """Retry-After as seconds (delta-seconds or HTTP-date); None if absent or unparseable."""
    if not value:
        return None
    value = str(value).strip()
    if value.isdigit():
        return float(value)
    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return None if dt is None else max(0.0, dt.timestamp() - (now or time.time()))


def backoff_delay(attempt, base=1.0, cap=8.0):
    """Full-jitter exponential backoff: uniform(0, min(cap, base * 2**attempt))."""
    return random.uniform(0, min(cap, base * 2 ** attempt))


def safe_headers(headers):
    return {k: v for k, v in (headers or {}).items() if k.lower() in SAFE_HEADERS}


def _detail(r):
    try:
        d = r.json()
        return str(d.get("title") or d.get("detail") or "")[:60] if isinstance(d, dict) else ""
    except Exception:  # noqa: BLE001
        return ""


def _key_path(key: str) -> Path:
    h = hashlib.sha1(key.encode()).hexdigest()
    return CACHE_DIR / h[:2] / f"{h}.json"


def cache_get(key: str, ttl: float | None):
    """ttl=None means never expire (immutable historical data)."""
    p = _key_path(key)
    if not p.exists():
        return None
    if ttl is not None and time.time() - p.stat().st_mtime > ttl:
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def cache_put(key: str, value) -> None:
    p = _key_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value))
    tmp.replace(p)


def _throttle(host: str, min_interval: float) -> None:
    wait = _last_hit.get(host, 0.0) + min_interval - time.time()
    if wait > 0:
        time.sleep(wait + random.uniform(0, 0.25 * min_interval))
    _last_hit[host] = time.time()


def _cache_key(method, url, params, body):
    return f"{method} {url} {json.dumps(params, sort_keys=True)} {json.dumps(body, sort_keys=True)}"


def cached_at(url, *, params=None, method="GET", body=None):
    """Unix time the cached response for this request was stored, or None."""
    p = _key_path(_cache_key(method, url, params, body))
    return p.stat().st_mtime if p.exists() else None


def fetch_json(url, *, params=None, headers=None, method="GET", body=None,
               ttl: float | None = 0, min_interval=0.6, retries=4, timeout=30, max_backoff=8.0,
               circuit: CircuitBreaker | None = None):
    """GET/POST JSON with cache (ttl seconds; 0 = no cache; None = forever).
    Request headers are never logged (they may carry API keys)."""
    key = _cache_key(method, url, params, body)
    if ttl != 0:
        hit = cache_get(key, ttl)
        if hit is not None:
            return hit
    host = url.split("/")[2]
    path = url.split("?")[0]
    if circuit is not None and not circuit.allow():
        raise CircuitOpen(f"{host} paused after repeated failures, next try in {circuit.remaining():.0f}s "
                          f"(last: {circuit.last_error})", retry_after=circuit.remaining())
    for attempt in range(retries + 1):
        _throttle(host, min_interval)
        t0 = time.time()
        try:
            r = _session.request(method, url, params=params, headers=headers, json=body, timeout=timeout)
        except requests.RequestException as e:
            log.warning("%s %s %s -> %s after %.0f ms (attempt %d/%d)", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        method, path, type(e).__name__, (time.time() - t0) * 1000, attempt + 1, retries + 1)
            if attempt == retries:
                err = UpstreamUnavailable(f"{type(e).__name__} from {host}")
                if circuit is not None:
                    circuit.failure(str(err))
                raise err from e
            time.sleep(backoff_delay(attempt, cap=max_backoff))
            continue
        ms = (time.time() - t0) * 1000
        if r.status_code >= 400:
            log.warning("%s %s %s -> %d in %.0f ms (attempt %d/%d) %s", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        method, path, r.status_code, ms, attempt + 1, retries + 1, safe_headers(r.headers))
        if r.status_code in (401, 403):
            err = AccessDenied(f"{r.status_code} from {host}")
            if circuit is not None:
                circuit.failure(f"access denied ({r.status_code})", hard=True)
            raise err
        if r.status_code == 429 or r.status_code >= 500:
            ra = retry_after_seconds(r.headers.get("Retry-After"))
            if attempt == retries or (ra is not None and ra > max_backoff):
                det = _detail(r)
                err = UpstreamUnavailable(f"{r.status_code}{' ' + det if det else ''} from {host}",
                                          status=r.status_code, retry_after=ra, detail=det)
                if circuit is not None:
                    circuit.failure(str(err), retry_after=ra)
                raise err
            time.sleep(ra if ra is not None else backoff_delay(attempt, cap=max_backoff))
            continue
        r.raise_for_status()
        data = r.json()
        log.debug("%s %s -> %d in %.0f ms", method, path, r.status_code, ms)
        if circuit is not None:
            circuit.success()
        if ttl != 0:
            cache_put(key, data)
        return data
