"""Speed statistics, and the HTTP API's own behaviour (auth, validation, response shape)."""
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from backend import main, pipeline, speed


def _signal(clock, feed, publish_to_seen, seen_to_signal, backfill=False, fast_lane=False, analysis="Quick"):
    processed = clock.utc()
    seen = processed - timedelta(seconds=seen_to_signal)
    published = seen - timedelta(seconds=publish_to_seen)
    iso = lambda dt: dt.isoformat() + "Z"
    pipeline.state.trade_history.append({
        "Headline": f"{feed} {publish_to_seen} {seen_to_signal}", "Timestamp": iso(published), "Seen At": iso(seen),
        "Processed At": iso(processed), "Feed": feed, "Backfill": backfill, "Fast Lane": fast_lane, "Analysis": analysis,
    })


def test_speed_splits_source_delay_from_bot_delay(clock):
    _signal(clock, "Wire", publish_to_seen=30, seen_to_signal=10, fast_lane=True)
    _signal(clock, "Wire", publish_to_seen=90, seen_to_signal=20, fast_lane=True)
    _signal(clock, "Blog", publish_to_seen=600, seen_to_signal=100, analysis="Deep")

    stats = speed.stats()
    assert stats["signals"] == 3 and stats["backfill_excluded"] == 0
    assert stats["overall"]["publish_to_seen"] == {"count": 3, "median": 90, "p90": 600}
    assert stats["overall"]["seen_to_signal"]["median"] == 20
    assert stats["overall"]["publish_to_signal"]["median"] == 110

    wire, blog = stats["by_feed"]  # most signals first
    assert wire["feed"] == "Wire" and wire["publish_to_seen"]["median"] == 60
    assert blog["feed"] == "Blog" and blog["publish_to_seen"]["median"] == 600
    assert {row["lane"]: row["seen_to_signal"]["median"] for row in stats["by_lane"]} == {"Fast lane": 15, "Batched": 100}
    assert {row["analysis"]: row["seen_to_signal"]["count"] for row in stats["by_analysis"]} == {"Deep": 1, "Quick": 2}


def test_backfill_is_left_out_of_source_delay_but_not_bot_delay(clock):
    _signal(clock, "Wire", publish_to_seen=20, seen_to_signal=5)
    _signal(clock, "Wire", publish_to_seen=3000, seen_to_signal=15, backfill=True)  # sat in the feed during a restart

    stats = speed.stats()
    assert stats["backfill_excluded"] == 1
    assert stats["overall"]["publish_to_seen"] == {"count": 1, "median": 20, "p90": 20}
    assert stats["overall"]["seen_to_signal"]["count"] == 2


def test_signals_without_a_seen_time_are_skipped(clock):
    pipeline.state.trade_history.append({"Headline": "old format", "Timestamp": clock.iso(), "Processed At": clock.iso()})
    assert speed.stats()["signals"] == 0
    assert speed.stats()["overall"]["publish_to_seen"] == {"count": 0, "median": None, "p90": None}


def test_a_publish_time_in_the_future_counts_as_zero_delay(clock):
    _signal(clock, "Wire", publish_to_seen=-120, seen_to_signal=5)
    assert speed.stats()["overall"]["publish_to_seen"]["median"] == 0


@pytest.fixture
def client():
    # No `with`: the lifespan (scheduler loops, Finnhub socket) must not start in tests.
    return TestClient(main.app)


def test_status_reports_health_fields(client, clock):
    status = client.get("/api/status").json()
    assert status["data_day"] == pipeline.local_today()
    assert status["signals_today"] == 0 and status["write_endpoints_protected"] is False


def test_read_endpoints_answer(client, clock):
    for path in ("/api/signals", "/api/usage", "/api/settings", "/api/errors", "/api/speed", "/api/scorecard", "/api/market-regime", "/api/ticker-bar", "/api/watchlist"):
        assert client.get(path).status_code == 200, path


def test_errors_endpoint_lists_newest_first(client):
    pipeline.report_error("Source", "first")
    pipeline.report_error("Source", "second")
    assert [e["message"] for e in client.get("/api/errors").json()["errors"]] == ["second", "first"]


def test_scorecard_days_is_validated(client, clock):
    assert client.get("/api/scorecard?days=30").status_code == 200
    assert client.get("/api/scorecard?days=0").status_code == 400
    assert client.get("/api/scorecard?days=99").status_code == 400


def test_bad_query_values_are_rejected(client):
    assert client.get("/api/market-data?category=nonsense").status_code == 400
    assert client.get("/api/sectors?timeframe=5Y").status_code == 400


def test_write_endpoints_are_open_without_an_admin_token(client):
    assert client.patch("/api/settings", json={"refresh_interval_seconds": 25}).status_code == 200
    assert pipeline.state.refresh_interval_seconds == 25


def test_write_endpoints_require_the_admin_token_when_set(client, monkeypatch):
    monkeypatch.setattr(main, "ADMIN_TOKEN", "secret")
    assert client.patch("/api/settings", json={"active": False}).status_code == 401
    assert client.patch("/api/settings", json={"active": False}, headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert client.delete("/api/watchlist/NVDA").status_code == 401
    assert client.post("/api/scan").status_code == 401
    assert pipeline.state.active is True and any(w["symbol"] == "NVDA" for w in pipeline.state.watchlist)

    assert client.patch("/api/settings", json={"active": False}, headers={"X-Admin-Token": "secret"}).status_code == 200
    assert pipeline.state.active is False
    assert client.get("/api/signals").status_code == 200  # reading never needs the token
    assert client.get("/api/status").json()["write_endpoints_protected"] is True


def test_settings_interval_is_bounded(client):
    assert client.patch("/api/settings", json={"refresh_interval_seconds": 2}).status_code == 400
    assert client.patch("/api/settings", json={"refresh_interval_seconds": 99999}).status_code == 400
