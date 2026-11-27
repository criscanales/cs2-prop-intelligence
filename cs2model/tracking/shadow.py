"""Shadow recalibration: a frozen second Platt layer on the production P(over), logged next to production before match
start so it can be checked on new matches. It never feeds picks, edges, tiers or the board."""
import hashlib
import json

from ..scraper.http import ROOT
from ..training.calibration import Platt

PATH = ROOT / "config" / "shadow_calibration.json"


def load(path=None):
    """(version tag, {market key: Platt}) or None when no shadow calibration is configured."""
    path = path or PATH
    if not path.exists():
        return None
    raw = path.read_bytes()
    d = json.loads(raw)
    return f"{d['version']}:{hashlib.sha256(raw).hexdigest()[:12]}", {k: Platt.from_params(v) for k, v in d["params"].items()}


def key(stat, maps):
    return f"{stat}_{'series' if '+' in str(maps) else 'map'}"


def p_over(cal, stat, maps, p):
    c = cal.get(key(stat, maps))
    return float(c.predict([p])[0]) if c is not None else None
