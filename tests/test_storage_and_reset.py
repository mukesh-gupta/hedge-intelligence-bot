"""Where signals are stored in Upstash, and the once-a-day deletion at midnight IST."""
import json
import time
from datetime import datetime

from backend import pipeline


def _stored(upstash):
    return [json.loads(item)["Headline"] for item in upstash.db.get("signals", [])]


def _in_memory():
    return [alert["Headline"] for alert in pipeline.state.trade_history]


def test_each_signal_is_one_small_push_not_a_rewrite(upstash, publish):
    for i in range(3):
        publish(f"signal {i}", Summary="x" * 900)
    assert _stored(upstash) == ["signal 2", "signal 1", "signal 0"]  # newest first
    pushes = [c for c in upstash.commands if c[0] == "LPUSH"]
    assert len(pushes) == 3 and all(len(c) == 3 for c in pushes)  # one value per push


def test_stored_signals_are_capped(upstash, publish, monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_STORED_ALERTS", 4)
    for i in range(6):
        publish(f"signal {i}")
    assert _stored(upstash) == ["signal 5", "signal 4", "signal 3", "signal 2"]
    assert len(pipeline.state.trade_history) == 4


def test_signals_reload_from_upstash(upstash, publish):
    publish("first")
    publish("second")
    assert [a["Headline"] for a in pipeline.load_trade_history()] == ["second", "first"]


def test_load_is_empty_when_upstash_is_not_configured():
    assert pipeline.load_trade_history() == []


def test_only_recent_seen_markers_are_persisted(upstash):
    now = time.time()
    pipeline.save_processed_headlines({"five hours ago": now - 5 * 3600, "a minute ago": now - 60, "now": now})
    assert list(json.loads(upstash.db["processed_headlines"])) == ["a minute ago", "now"]


def test_day_boundary_is_midnight_india_time(clock):
    clock.set(2026, 10, 4, 18, 29, 59)  # 23:59:59 IST
    assert pipeline.local_today() == "2026-10-04"
    assert pipeline.start_of_today_utc() == datetime(2026, 10, 3, 18, 30)
    clock.set(2026, 10, 4, 18, 30, 1)  # 00:00:01 IST
    assert pipeline.local_today() == "2026-10-05"
    assert pipeline.start_of_today_utc() == datetime(2026, 10, 4, 18, 30)


def _start_at(clock, *when):
    """Sets the clock and tells the pipeline this is the day it has been running in, so a
    later move past midnight is the only day change it sees."""
    clock.set(*when)
    pipeline.state.token_usage_date = pipeline.local_today()


def _cycle(monkeypatch, clock):
    monkeypatch.setattr(pipeline, "fetch_live_financial_news", lambda: [])
    pipeline.state.last_scan_time = clock.now
    pipeline.run_pipeline_cycle()


def test_nothing_is_deleted_before_midnight(upstash, clock, publish, monkeypatch):
    _start_at(clock, 2026, 10, 4, 18, 20)
    publish("evening news")
    clock.set(2026, 10, 4, 18, 29)
    _cycle(monkeypatch, clock)
    assert _in_memory() == ["evening news"] and _stored(upstash) == ["evening news"]


def test_midnight_deletes_the_previous_day(upstash, clock, publish, monkeypatch):
    _start_at(clock, 2026, 10, 4, 4, 0)
    publish("morning news")
    clock.set(2026, 10, 4, 18, 20)
    publish("evening news")
    upstash.db["trade_history"] = "[]"  # the pre-migration copy
    upstash.db["watchlist"] = json.dumps([{"symbol": "NVDA", "label": "NVDA"}])
    pipeline.state.groq_tokens_today = 5000
    pipeline.state.model_tokens_today = {pipeline.GROQ_FAST_MODEL: 5000}
    pipeline.state.processed_headlines = {"seen 5 hours ago": clock.now - 5 * 3600, "seen 10 minutes ago": clock.now - 600}

    clock.set(2026, 10, 4, 18, 30, 30)  # 30 seconds into the new day
    _cycle(monkeypatch, clock)

    assert _in_memory() == [] and "signals" not in upstash.db
    assert "trade_history" not in upstash.db
    assert "watchlist" in upstash.db  # a setting, not daily data
    assert pipeline.state.groq_tokens_today == 0 and pipeline.state.model_tokens_today == {}
    # Recent markers survive on purpose: wiping them would re-publish last night's stories.
    assert list(pipeline.state.processed_headlines) == ["seen 10 minutes ago"]


def test_a_story_analyzed_just_after_midnight_belongs_to_the_new_day(upstash, clock, monkeypatch):
    clock.set(2026, 10, 4, 18, 31)  # 00:01 IST
    alert = {"Timestamp": "2026-10-04T18:28:00Z", "Headline": "published 23:58, analyzed 00:01", "Sentiment": "NEUTRAL"}
    pipeline._publish_alert(alert["Headline"], alert)
    pipeline.delete_previous_days_data()
    assert _in_memory() == ["published 23:58, analyzed 00:01"]


def test_restart_removes_leftovers_and_keeps_today(upstash, clock, publish):
    clock.set(2026, 10, 5, 3, 0)
    publish("today")
    old = {"Headline": "two days old, no Processed At", "Timestamp": "2026-10-03T09:00:00Z"}
    upstash.db["signals"].append(json.dumps(old))

    pipeline.state.trade_history = pipeline.load_trade_history()  # what startup does
    pipeline.delete_previous_days_data()

    assert _in_memory() == ["today"] and _stored(upstash) == ["today"]


def test_a_second_cycle_on_the_same_day_deletes_nothing(upstash, clock, publish, monkeypatch):
    _start_at(clock, 2026, 10, 5, 3, 0)
    publish("today")
    _cycle(monkeypatch, clock)
    clock.advance(hours=5)
    _cycle(monkeypatch, clock)
    assert _in_memory() == ["today"]
