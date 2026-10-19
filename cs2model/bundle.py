"""Local data bundle: export this install's bo3.gg history and trained models to one zip, import it on another computer.

User-initiated local transfer only. Excluded: the tracking log (your picks), config/settings, the HTTP cache, saved
odds snapshots, and anything that can hold keys or tokens. Calibrators travel as plain numbers, never as pickles.
"""
import hashlib
import io
import json
import shutil
import time
import zipfile
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .features.build import AWP_PATH, FEATURES
from .scraper.bo3_history import DATA, HIST_PATH, merge_history, save_history
from .scraper.http import ROOT
from .training import bootstrap as B
from .training.calibration import Platt
from .training.train import MODEL_DIR

FORMAT, VERSION = "cs2props-bundle", 1
EXPORT_DIR = ROOT / "exports"
MAX_BYTES = 1_000_000_000  # uncompressed size cap per member
DATA_FILES = {"data/player_maps.csv": HIST_PATH, "data/awp_maps.csv": AWP_PATH,
              "data/team_logos.json": DATA / "team_logos.json", "data/player_photos.json": DATA / "player_photos.json",
              "data/photo_frames.json": DATA / "photo_frames.json"}
MODEL_FILES = {f"models/{f}": MODEL_DIR / f for f in ("kills_lgbm.txt", "hs_lgbm.txt", "artifacts.json", "backtest.json",
                                                       "backtest_rows.csv")}
CALIB = "models/calibrators.json"
ALLOWED = set(DATA_FILES) | set(MODEL_FILES) | {CALIB, "manifest.json"}
REQUIRED_COLS = ("date", "match_id", "game_id", "map_number", "map_name", "tier", "team_id", "opp_id", "player_id",
                 "player", "won", "rounds", "kills", "headshots")
EXCLUDED = ["data/tracking.db (your logged picks)", "config/ (settings, payouts)", "data/cache/ (HTTP cache)",
            "data/pinnacle_last_good.json", "API keys, cookies and tokens"]


class BundleError(Exception):
    """Bundle can't be used; the message says why."""


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _app_version() -> str:
    try:
        import subprocess
        return subprocess.run(["git", "describe", "--tags", "--always"], cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001  frozen app: no git
        return "desktop"


def export_bundle(out_dir: Path | None = None, now: float | None = None) -> Path:
    if not HIST_PATH.exists():
        raise BundleError("No player history to export yet.")
    out_dir = out_dir or EXPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {name: p.read_bytes() for name, p in {**DATA_FILES, **MODEL_FILES}.items() if p.exists()}
    have_model = all(n in payload for n in MODEL_FILES) and (MODEL_DIR / "calibrators.joblib").exists()
    if have_model:
        cal = joblib.load(MODEL_DIR / "calibrators.joblib")  # our own local file
        payload[CALIB] = json.dumps({k: v.to_params() for k, v in cal.items()}, indent=1).encode()
    else:
        payload = {k: v for k, v in payload.items() if not k.startswith("models/")}
    h = pd.read_csv(io.BytesIO(payload["data/player_maps.csv"]), usecols=["date", "game_id"])
    d = pd.to_datetime(h.date, utc=True, format="ISO8601")
    cov = B.read_coverage() or {}
    manifest = {
        "format": FORMAT, "bundle_version": VERSION, "app_version": _app_version(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "history": {"rows": int(len(h)), "maps": int(h.game_id.nunique()), "from": str(d.min()), "to": str(d.max()),
                    "complete": B.is_complete(), "target_days": cov.get("target_days")},
        "model": None, "excluded": EXCLUDED,
        "files": {n: {"sha256": _sha(b), "bytes": len(b)} for n, b in payload.items()},
    }
    if have_model:
        art = json.loads(payload["models/artifacts.json"])
        manifest["model"] = {"features": art.get("features"), "trained_through": art.get("trained_through"),
                             "n_rows": art.get("n_rows"), "history": art.get("history"),
                             "lightgbm": lgb.__version__, "trained_utc": time.strftime(
                                 "%Y-%m-%dT%H:%M:%SZ", time.gmtime((MODEL_DIR / "artifacts.json").stat().st_mtime))}
    path = out_dir / f"cs2props-bundle-{time.strftime('%Y%m%d-%H%M', time.gmtime(now))}.zip"
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for n, b in payload.items():
            z.writestr(n, b)
        z.writestr("manifest.json", json.dumps(manifest, indent=1))
    tmp.replace(path)
    return path


def read_bundle(src) -> tuple[dict, dict]:
    """Validate a bundle (path, bytes or file object) and return (manifest, {name: bytes}). Raises BundleError."""
    try:
        z = zipfile.ZipFile(io.BytesIO(src) if isinstance(src, bytes) else src)
    except (zipfile.BadZipFile, OSError) as e:
        raise BundleError(f"Not a valid bundle zip ({e}).") from e
    with z:
        names = z.namelist()
        bad = [n for n in names if n not in ALLOWED]
        if bad:
            raise BundleError(f"Unexpected files in bundle: {', '.join(bad[:5])}. Only bundles exported by this app can be imported.")
        if "manifest.json" not in names:
            raise BundleError("Missing manifest.json: not a CS2 Props bundle.")
        if any(i.file_size > MAX_BYTES for i in z.infolist()):
            raise BundleError("A file in the bundle is too large.")
        try:
            man = json.loads(z.read("manifest.json"))
        except ValueError as e:
            raise BundleError("manifest.json is not valid JSON.") from e
        if man.get("format") != FORMAT:
            raise BundleError("Not a CS2 Props bundle (wrong format tag).")
        if not isinstance(man.get("bundle_version"), int) or man["bundle_version"] > VERSION:
            raise BundleError(f"Bundle version {man.get('bundle_version')} is newer than this app supports ({VERSION}). Update the app.")
        files = man.get("files") or {}
        if set(files) != set(names) - {"manifest.json"}:
            raise BundleError("Bundle contents don't match its manifest.")
        out = {}
        for n, meta in files.items():
            try:
                b = z.read(n)
            except (zipfile.BadZipFile, KeyError, OSError) as e:
                raise BundleError(f"{n} can't be read: {e}") from e
            if len(b) != meta.get("bytes") or _sha(b) != meta.get("sha256"):
                raise BundleError(f"{n} failed the integrity check (checksum mismatch). The file is damaged or was modified.")
            out[n] = b
    if "data/player_maps.csv" not in out:
        raise BundleError("Bundle has no player history.")
    return man, out


def _check_models(man, files, tmp: Path) -> str | None:
    """None if the bundled models work with this app, otherwise the reason they don't."""
    if not all(n in files for n in (*MODEL_FILES, CALIB)):
        return "the bundle has no complete model"
    art = json.loads(files["models/artifacts.json"])
    if list(art.get("features") or []) != list(FEATURES):
        return "the bundled model uses a different feature set than this app version"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        for n in MODEL_FILES:
            (tmp / Path(n).name).write_bytes(files[n])
        for f in ("kills_lgbm.txt", "hs_lgbm.txt"):
            bst = lgb.Booster(model_file=str(tmp / f))
            if bst.feature_name() != list(FEATURES):
                return f"{f} expects different inputs than this app version"
            bst.predict(np.zeros((1, len(FEATURES))), raw_score=True)
        cal = {k: Platt.from_params(v) for k, v in json.loads(files[CALIB]).items()}
        if not {"kills_map", "headshots_map"} <= set(cal):
            return "the bundled calibrators are incomplete"
        for c in cal.values():
            c.predict([0.3, 0.5, 0.7])
        for k in ("shrink", "alpha", "rho", "win_ratio", "rounds_table"):
            if k not in art:
                return f"artifacts.json is missing '{k}'"
        joblib.dump(cal, tmp / "calibrators.joblib")
    except Exception as e:  # noqa: BLE001
        return f"the bundled model failed to load ({e})"
    return None


def import_bundle(src, replace_models: bool = False) -> dict:
    """Merge the bundle's history into this install (no duplicate player-maps) and install its models when there are
    none locally, or when replace_models is set. Returns a report; raises BundleError and changes nothing on failure."""
    man, files = read_bundle(src)
    try:
        new = pd.read_csv(io.BytesIO(files["data/player_maps.csv"]))
    except Exception as e:  # noqa: BLE001
        raise BundleError(f"player_maps.csv can't be parsed ({e}).") from e
    miss = [c for c in REQUIRED_COLS if c not in new.columns]
    if miss:
        raise BundleError(f"History is missing columns: {', '.join(miss)}.")
    if len(new) != (man.get("history") or {}).get("rows"):
        raise BundleError("History row count doesn't match the manifest.")
    tmp = ROOT / "models_import_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        why_not = _check_models(man, files, tmp)
        had_model = (MODEL_DIR / "artifacts.json").exists()
        install = why_not is None and (not had_model or replace_models)
        local_complete = B.is_complete() if HIST_PATH.exists() else False
        old = pd.read_csv(HIST_PATH) if HIST_PATH.exists() else None
        merged = merge_history(old, new)
        n_old = 0 if old is None else len(old)
        save_history(merged)
        if "data/awp_maps.csv" in files:
            a = pd.read_csv(io.BytesIO(files["data/awp_maps.csv"]))
            if AWP_PATH.exists():
                a = pd.concat([pd.read_csv(AWP_PATH), a], ignore_index=True)
            a.drop_duplicates(["game_id", "player_id"], keep="first").to_csv(AWP_PATH, index=False)
        for n in ("data/team_logos.json", "data/player_photos.json", "data/photo_frames.json"):
            if n in files:
                p = DATA_FILES[n]
                cur = json.loads(p.read_text()) if p.exists() else {}
                p.write_text(json.dumps({**json.loads(files[n]), **cur}))
        backup = None
        if install:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            if had_model:
                backup = MODEL_DIR.parent / f"models_backup_{time.strftime('%Y%m%d-%H%M%S')}"
                shutil.copytree(MODEL_DIR, backup)
            for f in ("kills_lgbm.txt", "hs_lgbm.txt", "calibrators.joblib", "backtest.json", "backtest_rows.csv", "artifacts.json"):
                shutil.copy2(tmp / f, MODEL_DIR / f)  # artifacts.json last
        span = B.history_span()
        from .parlay.engine import load_yaml
        target = int(load_yaml("settings.yaml")["history_days"])
        complete = bool((man.get("history") or {}).get("complete")) or local_complete or span["days"] >= target
        B.write_coverage(complete=complete, target_days=target, phase="done" if complete else "backfill",
                         imported={"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "created": man.get("created_utc")})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"rows_in_bundle": len(new), "rows_added": len(merged) - n_old, "duplicates_skipped": len(new) - (len(merged) - n_old),
            "models_installed": install, "models_skipped_reason": None if install else (
                why_not or "kept your existing models (tick 'Replace my models' to use the bundle's)"),
            "model_backup": str(backup) if backup else None, "history_complete": complete,
            "history_from": str(span["from"]), "history_to": str(span["to"]), "bundle": man}
