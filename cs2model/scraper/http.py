"""Polite HTTP client: plain `requests`, per-host throttling, retries with backoff, disk cache.

No browser impersonation or anti-bot workarounds are used. If a source answers 401/403
(access control), we raise `AccessDenied` and the caller skips that source.
"""
import hashlib
import json
import os
import random
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.environ.get("CS2_CACHE_DIR", ROOT / "data" / "cache"))
USER_AGENT = os.environ.get("CS2_USER_AGENT", "cs2-props-model/0.1 (personal research)")

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
_last_hit: dict[str, float] = {}


class AccessDenied(Exception):
    pass


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


def fetch_json(url, *, params=None, headers=None, method="GET", body=None,
               ttl: float | None = 0, min_interval=0.6, retries=4, timeout=30):
    """GET/POST JSON with cache (ttl seconds; 0 = no cache; None = forever)."""
    key = f"{method} {url} {json.dumps(params, sort_keys=True)} {json.dumps(body, sort_keys=True)}"
    if ttl != 0:
        hit = cache_get(key, ttl)
        if hit is not None:
            return hit
    host = url.split("/")[2]
    delay = 2.0
    for attempt in range(retries + 1):
        _throttle(host, min_interval)
        try:
            r = _session.request(method, url, params=params, headers=headers, json=body, timeout=timeout)
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code in (401, 403):
            raise AccessDenied(f"{r.status_code} from {host}")
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == retries:
                r.raise_for_status()
            ra = r.headers.get("Retry-After")
            time.sleep(float(ra) if ra and ra.isdigit() else delay)
            delay *= 2
            continue
        r.raise_for_status()
        data = r.json()
        if ttl != 0:
            cache_put(key, data)
        return data
