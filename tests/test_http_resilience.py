import json
from datetime import datetime, timezone

import pytest
import requests

from cs2model.scraper import books, http


class Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


@pytest.fixture
def net(monkeypatch):
    """Queue of fake responses/exceptions; records request count and sleeps."""
    st = {"queue": [], "calls": 0, "sleeps": []}

    def request(*a, **k):
        st["calls"] += 1
        r = st["queue"].pop(0) if len(st["queue"]) > 1 else st["queue"][0]
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(http._session, "request", request)
    monkeypatch.setattr(http.time, "sleep", lambda s: st["sleeps"].append(s))
    return st


def get(**kw):
    return http.fetch_json("https://api.example.invalid/x", min_interval=0, **kw)


MAINT = {"title": "MAINTENANCE", "detail": "API is currently undergoing maintenance", "status": 503}


def test_200_closes_circuit(net):
    cb = http.CircuitBreaker()
    cb.failures = 1
    net["queue"] = [Resp(200, [1, 2])]
    assert get(circuit=cb) == [1, 2] and cb.failures == 0 and cb.allow() and net["calls"] == 1


def test_403_is_auth_not_retried_and_opens_circuit(net):
    cb = http.CircuitBreaker()
    net["queue"] = [Resp(403)]
    with pytest.raises(http.AccessDenied):
        get(circuit=cb)
    assert net["calls"] == 1 and net["sleeps"] == [] and not cb.allow()
    with pytest.raises(http.CircuitOpen):
        get(circuit=cb)
    assert net["calls"] == 1


def test_429_honours_retry_after(net):
    net["queue"] = [Resp(429, headers={"Retry-After": "3"}), Resp(200, {"ok": 1})]
    assert get() == {"ok": 1} and net["sleeps"] == [3.0] and net["calls"] == 2


def test_long_retry_after_fails_fast_and_sets_cooldown(net):
    cb = http.CircuitBreaker(threshold=1, cooldown=60)
    net["queue"] = [Resp(429, headers={"Retry-After": "900"})]
    with pytest.raises(http.UpstreamUnavailable) as e:
        get(circuit=cb, max_backoff=4)
    assert e.value.status == 429 and e.value.retry_after == 900 and net["calls"] == 1 and net["sleeps"] == []
    assert cb.remaining() > 800


def test_retry_after_http_date():
    now = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc).timestamp()
    assert http.retry_after_seconds("Mon, 19 Oct 2026 12:02:00 GMT", now=now) == 120
    assert http.retry_after_seconds("soon") is None and http.retry_after_seconds(None) is None


def test_503_bounded_jittered_backoff_then_circuit(net):
    cb = http.CircuitBreaker(threshold=2, cooldown=300)
    net["queue"] = [Resp(503, MAINT)]
    with pytest.raises(http.UpstreamUnavailable) as e:
        get(circuit=cb, retries=2, max_backoff=4)
    assert "503 MAINTENANCE" in str(e.value) and not isinstance(e.value, http.CircuitOpen)
    assert net["calls"] == 3 and len(net["sleeps"]) == 2 and all(0 <= s <= 4 for s in net["sleeps"])
    assert cb.allow()
    with pytest.raises(http.UpstreamUnavailable):
        get(circuit=cb, retries=2, max_backoff=4)
    assert not cb.allow() and 299 < cb.remaining() <= 300
    with pytest.raises(http.CircuitOpen):
        get(circuit=cb, retries=2)
    assert net["calls"] == 6
    cb.open_until = 0  # cooldown elapsed: one trial request, failure reopens with a doubled cooldown
    with pytest.raises(http.UpstreamUnavailable):
        get(circuit=cb, retries=0)
    assert net["calls"] == 7 and cb.remaining() > 599
    cb.open_until = 0
    net["queue"] = [Resp(200, {"ok": 1})]
    assert get(circuit=cb) == {"ok": 1} and cb.allow() and cb.n_open == 0


def test_backoff_bounds():
    for a in range(10):
        assert 0 <= http.backoff_delay(a, base=1, cap=8) <= min(8, 2 ** a)


def test_timeout_is_transient(net):
    net["queue"] = [requests.Timeout("slow")]
    with pytest.raises(http.UpstreamUnavailable, match="Timeout"):
        get(retries=1)
    assert net["calls"] == 2
    net["queue"] = [requests.ConnectionError("reset"), Resp(200, {"ok": 1})]
    assert get(retries=1) == {"ok": 1}


def test_diagnostics_never_log_request_headers(net, caplog):
    net["queue"] = [Resp(503, MAINT, {"CF-RAY": "abc", "Set-Cookie": "s=1", "Server": "cloudflare"})]
    with caplog.at_level("WARNING"):
        with pytest.raises(http.UpstreamUnavailable):
            http.fetch_json("https://api.example.invalid/x", headers={"X-API-Key": "SECRET123"}, min_interval=0, retries=0)
    text = caplog.text
    assert "503" in text and "abc" in text and "SECRET123" not in text and "s=1" not in text and " ms " in text


PIN_MATCHUPS = [{"id": 1, "type": "matchup", "league": {"name": "CS2 - Test Cup"}, "startTime": "2026-10-20T12:00:00Z", "bestOfX": 3,
                 "participants": [{"alignment": "home", "name": "Alpha"}, {"alignment": "away", "name": "Beta"}]}]
PIN_MARKETS = [{"matchupId": 1, "period": 0, "type": "moneyline", "status": "open",
                "prices": [{"designation": "home", "price": -150}, {"designation": "away", "price": 130}]}]


def test_pinnacle_stale_data_is_labelled_and_not_priced(monkeypatch, tmp_path):
    monkeypatch.setattr(books, "PINNACLE_LAST_GOOD", tmp_path / "pin.json")
    monkeypatch.setattr(books, "cached_at", lambda u: datetime(2026, 10, 19, 11, 7, tzinfo=timezone.utc).timestamp())
    monkeypatch.setattr(books, "fetch_json", lambda u, **k: PIN_MATCHUPS if u.endswith("matchups") else PIN_MARKETS)
    odds, errs = books.fetch_all_books(("pinnacle",))
    assert len(odds) == 1 and errs == [] and odds[0].p_series_team1 > 0.5
    assert json.loads((tmp_path / "pin.json").read_text())["fetched_at"].startswith("2026-10-19T11:07")

    def down(u, **k):
        raise http.UpstreamUnavailable("503 MAINTENANCE from guest.api.arcadia.pinnacle.com", status=503)
    monkeypatch.setattr(books, "fetch_json", down)
    odds, errs = books.fetch_all_books(("pinnacle",))
    assert odds == [] and len(errs) == 1
    assert "temporarily unavailable" in errs[0] and "2026-10-19 11:07 UTC" in errs[0]
    assert "stale and not used for pricing" in errs[0] and "1 matches" in errs[0]

    def denied(u, **k):
        raise http.AccessDenied("403 from guest.api.arcadia.pinnacle.com")
    monkeypatch.setattr(books, "fetch_json", denied)
    odds, errs = books.fetch_all_books(("pinnacle",))
    assert odds == [] and "access denied" in errs[0] and "not an outage" in errs[0]


def test_other_books_unaffected_by_pinnacle_outage(monkeypatch):
    def down(*a, **k):
        raise http.CircuitOpen("paused")
    monkeypatch.setattr(books, "fetch_pinnacle", down)
    monkeypatch.setattr(books, "fetch_polymarket", lambda: ["poly"])
    monkeypatch.setattr(books, "pinnacle_last_good", lambda: None)
    got = books.fetch_all_books(("pinnacle", "polymarket"))
    assert got[0] == ["poly"] and len(got[1]) == 1 and got[1][0].startswith("pinnacle: temporarily unavailable")
