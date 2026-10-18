"""Per-photo crop so every bo3.gg portrait shows the face at the same size and position.

bo3.gg player photos are transparent half-body cut-outs whose framing varies a little between
players. For each photo we find the face (OpenCV Haar cascade, falling back to the alpha mask)
and store a CSS background crop: [size %, x %, y %].
"""
import hashlib
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

from .bo3_history import DATA

log = logging.getLogger(__name__)
FRAMES_PATH = DATA / "photo_frames.json"
CACHE = DATA / "cache" / "photos"
FACE_W = 0.50    # face width as a share of the portrait width
FACE_Y = 0.46    # vertical position of the face centre
HAIR_MIN = 0.09  # minimum gap above the top of the hair
SIDE_MIN = 0.04  # minimum gap beside the hair
DEFAULT = [175.0, 50.0, 4.0]  # original crop, used when no face is found
VERSION = 4


def _image(url):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (hashlib.md5(url.encode()).hexdigest() + ".webp")
    if not f.exists():
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        f.write_bytes(r.content)
    return np.asarray(Image.open(io.BytesIO(f.read_bytes())).convert("RGBA"))


def _face(a):
    try:
        import cv2
    except ImportError:
        return None
    H, W = a.shape[:2]
    al = a[..., 3:4] / 255.0
    g = cv2.cvtColor((a[..., :3] * al + 128 * (1 - al)).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    cc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    fs = [f for f in cc.detectMultiScale(g, 1.05, 5, minSize=(int(W * .12), int(W * .12))) if f[1] < H * 0.6 and abs(f[0] + f[2] / 2 - W / 2) < 0.12 * W]
    if not fs:
        return None
    x, y, w, h = max(fs, key=lambda f: f[2] * f[3])
    if not 0.2 * W <= w <= 0.42 * W:
        return None
    return x + w / 2, y + h / 2, float(np.clip(w, 0.25 * W, 0.36 * W))  # clamp rare partial / oversized detections


def frame(a):
    H, W = a.shape[:2]
    m = a[..., 3] > 128
    rows = np.where(m.sum(1) > 0.04 * W)[0]
    if not len(rows):
        return DEFAULT
    top = rows[0]
    f = _face(a)
    if f is None:
        return DEFAULT
    cx, cy, fw = f
    s = FACE_W / fw
    band = m[top:int(cy + 0.25 * fw)]  # hair/head above the cheeks
    col = band.sum(0) >= max(3, 0.03 * len(band))
    lo = hi = int(cx)
    while lo > 0 and col[lo - 1]:
        lo -= 1
    while hi < W - 1 and col[hi + 1]:
        hi += 1
    room = 0.5 - SIDE_MIN
    s = max(min(s, room / max(cx - lo, 1), room / max(hi - cx, 1)), 0.8 * s)  # zoom out (a little) rather than crop hair
    ox, oy = 0.5 - cx * s, FACE_Y - cy * s
    oy = max(oy, HAIR_MIN - top * s)
    zw, zh = s * W, s * H

    def pct(o, z):
        return 50.0 if abs(1 - z) < 1e-3 else o / (1 - z) * 100
    return [round(zw * 100, 1), round(pct(ox, zw), 1), round(pct(oy, zh), 1)]


def frames(photos, workers=8):
    """{player_id: [size %, x %, y %]} for `photos` ({player_id: url}); cached on disk by url."""
    try:
        cache = json.loads(FRAMES_PATH.read_text())
    except (OSError, ValueError):
        cache = {}
    if cache.get("_v") != VERSION:
        cache = {"_v": VERSION}
    todo = {u for u in photos.values() if u not in cache}

    def one(u):
        try:
            return u, frame(_image(u))
        except Exception as e:  # noqa: BLE001  a bad photo just keeps the default crop
            log.warning("portrait %s: %s", u, e)
            return u, None
    if todo:
        with ThreadPoolExecutor(workers) as ex:
            for u, fr in ex.map(one, sorted(todo)):
                if fr is not None:
                    cache[u] = fr
        FRAMES_PATH.parent.mkdir(exist_ok=True)
        FRAMES_PATH.write_text(json.dumps(cache))
    return {pid: cache[u] for pid, u in photos.items() if u in cache}
