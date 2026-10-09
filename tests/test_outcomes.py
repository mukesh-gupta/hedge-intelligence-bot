"""Result tracking: which signals are tracked, how a check is scored, and that the
scorecard survives restarts and the midnight wipe."""
import pytest

from backend import outcomes, pipeline


@pytest.fixture
def quotes(monkeypatch, clock):
    """Fake price feed. set(symbol, price) quotes it as trading right now."""
    book = {}
    monkeypatch.setattr(outcomes, "fetch_quote", lambda symbol: book.get(symbol))

    class Quotes:
        @staticmethod
        def set(symbol, price, traded_minutes_ago=0):
            book[symbol] = {"price": price, "trade_time": clock.now - traded_minutes_ago * 60}

        @staticmethod
        def freeze(symbol, price, trade_time):
            book[symbol] = {"price": price, "trade_time": trade_time}

        @staticmethod
        def remove(symbol):
            book.pop(symbol, None)

        book_ = book

    return Quotes


def test_only_signals_with_a_direction_and_a_priced_ticker_are_tracked(quotes, publish, upstash):
    quotes.set("AAA", 100.0)
    quotes.set("BBB", 50.0)
    bullish = publish("bullish", "STRONG_BULLISH", buy="AAA, ZZZ")
    bearish = publish("bearish", "BEARISH", sell="BBB")
    neutral = publish("neutral", "NEUTRAL", buy="AAA")
    no_ticker = publish("no ticker", "BULLISH")
    no_price = publish("no price", "BULLISH", buy="NOPRICE")

    assert outcomes.run_cycle() == {"started": 2, "scored": 0}
    assert bullish["Outcome"] == {"ticker": "AAA", "direction": "UP", "entry_price": 100.0, "1h": None, "1d": None}
    assert bearish["Outcome"]["ticker"] == "BBB" and bearish["Outcome"]["direction"] == "DOWN"
    assert neutral["Outcome"] is None and no_ticker["Outcome"] is None and no_price["Outcome"] is None
    assert len(upstash.db["outcome_pending"]) == 2


def test_a_call_whose_only_ticker_is_on_the_other_side_is_scored_that_way(quotes, publish):
    # Bullish on the rupee is published as "sell INR=X": the call is that INR=X falls.
    quotes.set("INR=X", 88.0)
    alert = publish("rupee rises", "BULLISH", sell="INR=X")
    outcomes.run_cycle()
    assert alert["Outcome"]["ticker"] == "INR=X" and alert["Outcome"]["direction"] == "DOWN"


def test_every_signal_has_an_outcome_key_from_the_start(publish):
    # Later updates must only replace the value: the API may be serializing the dict.
    assert "Outcome" in publish("anything", "NEUTRAL")


@pytest.mark.parametrize("sentiment, side, new_price, expected", [
    ("BULLISH", "buy", 101.0, ("right", 1.0)),
    ("BULLISH", "buy", 99.0, ("wrong", -1.0)),
    ("BULLISH", "buy", 100.05, ("flat", 0.05)),
    ("BEARISH", "sell", 99.8, ("right", -0.2)),
    ("BEARISH", "sell", 100.5, ("wrong", 0.5)),
    ("BEARISH", "sell", 99.95, ("flat", -0.05)),
])
def test_scoring_after_one_hour(quotes, publish, clock, sentiment, side, new_price, expected):
    quotes.set("AAA", 100.0)
    alert = publish("call", sentiment, **{side: "AAA"})
    outcomes.run_cycle()

    clock.advance(minutes=59)
    quotes.set("AAA", new_price)
    assert outcomes.run_cycle()["scored"] == 0  # not due yet

    clock.advance(minutes=2)
    quotes.set("AAA", new_price)
    assert outcomes.run_cycle()["scored"] == 1
    result = alert["Outcome"]["1h"]
    assert (result["outcome"], result["change_percent"]) == expected
    assert alert["Outcome"]["1d"] is None


def test_scorecard_counts_and_breaks_down(quotes, publish, clock):
    quotes.set("AAA", 100.0)
    quotes.set("BBB", 50.0)
    publish("deep india", "BULLISH", buy="AAA", Impact=9, Analysis="Deep", Region="India")
    publish("quick global", "BEARISH", sell="BBB", Impact=6)
    outcomes.run_cycle()
    clock.advance(minutes=61)
    quotes.set("AAA", 102.0)  # right
    quotes.set("BBB", 51.0)  # wrong
    outcomes.run_cycle()

    card = outcomes.scorecard(7)
    hour = card["horizons"]["1h"]
    assert hour["overall"] == {"right": 1, "wrong": 1, "flat": 0, "closed": 0, "accuracy": 50.0}
    assert [(r["value"], r["accuracy"]) for r in hour["by_impact"]] == [("9", 100.0), ("6", 0.0)]
    assert {r["value"] for r in hour["by_analysis"]} == {"Deep", "Quick"}
    assert {r["value"] for r in hour["by_region"]} == {"India", "Global"}
    assert {r["value"] for r in hour["by_direction"]} == {"UP", "DOWN"}
    assert hour["by_source"][0]["value"] == "Reuters"
    assert card["horizons"]["1d"]["overall"]["accuracy"] is None
    assert card["pending"] == 2 and len(card["recent"]) == 2 and card["since"] == pipeline.local_today()


def test_restart_restores_tracking_and_shows_results_on_signals(quotes, publish, clock, upstash):
    quotes.set("AAA", 100.0)
    publish("call", "BULLISH", buy="AAA")
    outcomes.run_cycle()
    clock.advance(minutes=61)
    quotes.set("AAA", 101.0)
    outcomes.run_cycle()

    outcomes.pending.clear()
    outcomes.stats.clear()
    outcomes.recent.clear()
    pipeline.state.trade_history = pipeline.load_trade_history()
    outcomes.load()

    assert len(outcomes.pending) == 1
    assert outcomes.scorecard(7)["horizons"]["1h"]["overall"]["right"] == 1
    assert pipeline.state.trade_history[0]["Outcome"]["1h"]["outcome"] == "right"


def test_midnight_wipe_leaves_tracking_alone_and_the_one_day_check_still_lands(quotes, publish, clock, upstash, monkeypatch):
    clock.set(2026, 10, 5, 4, 0)  # Mon 09:30 IST
    pipeline.state.token_usage_date = pipeline.local_today()
    quotes.set("AAA", 100.0)
    publish("call", "BULLISH", buy="AAA")
    outcomes.run_cycle()

    clock.set(2026, 10, 5, 18, 31)  # past midnight IST
    monkeypatch.setattr(pipeline, "fetch_live_financial_news", lambda: [])
    pipeline.state.last_scan_time = clock.now
    pipeline.run_pipeline_cycle()
    assert pipeline.state.trade_history == [] and "signals" not in upstash.db
    assert len(upstash.db["outcome_pending"]) == 1

    clock.set(2026, 10, 6, 4, 1)  # a day after the signal
    quotes.set("AAA", 97.0)
    assert outcomes.run_cycle()["scored"] == 2  # the 1h and 1d checks were both due
    assert outcomes.scorecard(7)["horizons"]["1d"]["overall"]["wrong"] == 1
    # Counted under the day the signal was published, not the day it was checked.
    assert list(outcomes.stats) == ["2026-10-05"]
    assert outcomes.pending == {} and upstash.db["outcome_pending"] == {}


def test_a_closed_market_is_not_scored_right_or_wrong(quotes, publish, clock):
    clock.set(2026, 10, 10, 6, 0)  # Saturday
    friday_close = clock.now - 16 * 3600
    quotes.freeze("CCC", 200.0, friday_close)
    alert = publish("weekend news", "BULLISH", buy="CCC")
    outcomes.run_cycle()

    clock.advance(hours=1, minutes=1)
    outcomes.run_cycle()
    assert alert["Outcome"]["1h"]["outcome"] == "closed"

    clock.advance(hours=23)
    outcomes.run_cycle()
    assert alert["Outcome"]["1d"] is None  # still waiting for a session

    clock.advance(hours=27)  # Monday
    quotes.set("CCC", 204.0, traded_minutes_ago=1)
    outcomes.run_cycle()
    assert alert["Outcome"]["1d"]["outcome"] == "right" and alert["Outcome"]["1d"]["change_percent"] == 2.0

    overall = outcomes.scorecard(7)["horizons"]
    assert overall["1h"]["overall"] == {"right": 0, "wrong": 0, "flat": 0, "closed": 1, "accuracy": None}
    assert overall["1d"]["overall"]["right"] == 1


def test_one_day_check_gives_up_on_a_market_that_never_reopens(quotes, publish, clock):
    quotes.freeze("DDD", 10.0, clock.now - 3600)
    alert = publish("halted stock", "BULLISH", buy="DDD")
    outcomes.run_cycle()
    for _ in range(40):
        clock.advance(hours=3)
        outcomes.run_cycle()
    assert alert["Outcome"]["1d"]["outcome"] == "closed" and outcomes.pending == {}


def test_price_lookup_failures_are_retried_then_given_up(quotes, publish, clock):
    quotes.set("EEE", 10.0)
    alert = publish("price feed dies", "BULLISH", buy="EEE")
    outcomes.run_cycle()
    quotes.remove("EEE")

    clock.advance(hours=1, minutes=1)
    outcomes.run_cycle()
    assert alert["Outcome"]["1h"] is None  # retried later, not scored on missing data
    for _ in range(outcomes.MAX_QUOTE_FAILURES + 1):
        clock.advance(minutes=11)
        outcomes.run_cycle()
    assert alert["Outcome"]["1h"]["outcome"] == "closed"


def test_scorecard_window_and_old_days_are_pruned(clock, upstash):
    outcomes.stats["2026-08-01"] = {"1h": {"overall:all": [5, 5, 0, 0]}}
    outcomes.stats["2026-10-03"] = {"1h": {"overall:all": [3, 1, 0, 0]}}
    outcomes.stats[pipeline.local_today()] = {"1h": {"overall:all": [1, 0, 0, 0]}}
    upstash.db["outcome_stats"] = {"2026-08-01": "{}"}

    assert outcomes.scorecard(1)["horizons"]["1h"]["overall"]["right"] == 1
    assert outcomes.scorecard(7)["horizons"]["1h"]["overall"]["right"] == 4
    assert outcomes.scorecard(7)["since"] == "2026-10-03"

    outcomes.run_cycle()
    assert "2026-08-01" not in outcomes.stats and "2026-08-01" not in upstash.db["outcome_stats"]


def test_upstash_writes_are_batched(quotes, publish, clock, upstash):
    for i in range(8):
        quotes.set(f"T{i}", 100.0)
        publish(f"call {i}", "BULLISH", buy=f"T{i}")
    upstash.commands.clear()
    outcomes.run_cycle()
    assert [c[0] for c in upstash.commands] == ["HSET"]  # eight predictions, one command

    clock.advance(minutes=61)
    for i in range(8):
        quotes.set(f"T{i}", 101.0)
    upstash.commands.clear()
    outcomes.run_cycle()
    assert len(upstash.commands) <= 5


def test_checks_per_run_are_capped(quotes, publish, clock):
    for i in range(outcomes.MAX_CHECKS_PER_RUN + 10):
        quotes.set(f"T{i}", 100.0)
        publish(f"call {i}", "BULLISH", buy=f"T{i}")
    outcomes.run_cycle()
    clock.advance(minutes=61)
    assert outcomes.run_cycle()["scored"] == outcomes.MAX_CHECKS_PER_RUN
    assert outcomes.run_cycle()["scored"] == 10


def _decided(right, wrong):
    return [right, wrong, 0, 0]


def test_hit_rate_comes_from_the_buckets_a_signal_falls_in(publish):
    outcomes.stats["2026-10-04"] = {"1d": {
        "impact:9": _decided(20, 10), "analysis:Deep": _decided(30, 20), "region:India": _decided(10, 10),
        "direction:UP": _decided(5, 5), "direction:DOWN": _decided(0, 40),
    }}
    outcomes.stats["2026-10-05"] = {"1d": {"region:India": _decided(20, 20)}, "1h": {"impact:9": _decided(100, 0)}}
    alert = publish("call", "BULLISH", buy="AAA", Impact=9, Analysis="Deep", Region="India")
    # impact 9: 20/30; Deep: 30/50; India over both days: 30/60; UP has only 10 decided, so it
    # is left out; the 1-hour tallies are not used.
    assert alert["Hit Rate"] == {"percent": round((20 / 30 + 30 / 50 + 30 / 60) / 3 * 100, 1), "sample": 30, "horizon": "1d"}
    # DOWN has 40 decided (all wrong), so it joins the mean for a bearish call.
    assert publish("bearish call", "BEARISH", sell="AAA", Impact=9, Analysis="Deep", Region="India")["Hit Rate"]["percent"] == round((20 / 30 + 30 / 50 + 30 / 60 + 0) / 4 * 100, 1)


def test_hit_rate_is_none_without_a_call_or_enough_history(publish):
    assert publish("no history yet", "BULLISH", buy="AAA")["Hit Rate"] is None
    outcomes.stats["2026-10-04"] = {"1d": {"impact:7": _decided(29, 0)}}
    assert publish("too thin", "BULLISH", buy="AAA", Impact=7)["Hit Rate"] is None
    outcomes.stats["2026-10-05"] = {"1d": {"impact:7": _decided(1, 0)}}
    assert publish("enough now", "BULLISH", buy="AAA", Impact=7)["Hit Rate"] == {"percent": 100.0, "sample": 30, "horizon": "1d"}
    assert publish("neutral", "NEUTRAL", Impact=7)["Hit Rate"] is None
