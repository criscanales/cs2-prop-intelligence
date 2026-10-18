import re
import unicodedata
from difflib import SequenceMatcher

_TEAM_NOISE = re.compile(r"\b(esports?|e-sports|team|gaming|club|cs2|cs|csgo|gg)\b")


def norm_player(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def norm_team(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    s = _TEAM_NOISE.sub(" ", s)
    return re.sub(r"[^a-z0-9]", "", s)


def team_similarity(a: str, b: str) -> float:
    a, b = norm_team(a), norm_team(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.9
    return SequenceMatcher(None, a, b).ratio()
