"""The AI fallback chain, triage parsing, and how the main loop routes headlines.
No test here calls a real model: the provider functions are replaced."""
import json
from datetime import timedelta

import pytest

from backend import pipeline


@pytest.fixture
def providers(monkeypatch):
    """Replaces the three provider calls. `answers` maps a provider name to what it returns;
    anything absent returns None (a failure). `calls` records the order tried."""
    calls, answers = [], {}
    monkeypatch.setattr(pipeline, "GEMINI_API_KEY", "x")
    monkeypatch.setattr(pipeline, "OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(pipeline, "_call_groq", lambda model, extra, prompt, temp: (calls.append(model), answers.get(model))[1])
    monkeypatch.setattr(pipeline, "_call_gemini", lambda prompt, temp: (calls.append("gemini"), answers.get("gemini"))[1])
    monkeypatch.setattr(pipeline, "_call_openrouter", lambda prompt, temp: (calls.append("openrouter"), answers.get("openrouter"))[1])
    return calls, answers


def test_retry_after_is_read_from_groq_error_text():
    assert pipeline._retry_after_seconds("Limit 200000, Used 199870. Please try again in 7m12.5s. Need more") == 432.5
    assert pipeline._retry_after_seconds("try again in 2h3m") == 7380
    assert pipeline._retry_after_seconds("no hint here") == 60


def test_each_tier_has_its_own_chain(providers, clock):
    calls, _ = providers
    pipeline.call_llm("x", tier="fast")
    assert calls == [pipeline.GROQ_FAST_MODEL, pipeline.GROQ_BACKUP_MODEL, "gemini", "openrouter"]
    calls.clear()
    pipeline.state.ai_unavailable_until = 0
    pipeline.call_llm("x", tier="deep")
    assert calls == [pipeline.GROQ_DEEP_MODEL, "gemini", pipeline.GROQ_BACKUP_MODEL, "openrouter"]


def test_tiers_never_use_each_others_primary_model(providers, clock):
    # Seen live: deep analysis fell back onto the triage model and spent its daily tokens.
    calls, _ = providers
    pipeline.call_llm("x", tier="deep")
    assert pipeline.GROQ_FAST_MODEL not in calls
    calls.clear()
    pipeline.state.ai_unavailable_until = 0
    pipeline.call_llm("x", tier="fast")
    assert pipeline.GROQ_DEEP_MODEL not in calls


def test_first_provider_that_answers_wins(providers, clock):
    calls, answers = providers
    answers["gemini"] = "from gemini"
    assert pipeline.call_llm("x", tier="deep") == "from gemini"
    assert calls == [pipeline.GROQ_DEEP_MODEL, "gemini"]


def test_rate_limited_model_is_skipped_until_its_limit_lifts(providers, clock):
    calls, answers = providers
    answers[pipeline.GROQ_BACKUP_MODEL] = "ok"
    pipeline._mark_unavailable(pipeline.GROQ_FAST_MODEL, "Please try again in 5m0s", 429)
    pipeline.call_llm("x", tier="fast")
    assert calls == [pipeline.GROQ_BACKUP_MODEL]
    clock.advance(minutes=6)
    calls.clear()
    pipeline.call_llm("x", tier="fast")
    assert calls[0] == pipeline.GROQ_FAST_MODEL


def test_all_providers_down_starts_a_cooldown(providers, clock):
    calls, _ = providers
    assert pipeline.call_llm("x", tier="fast") is None
    calls.clear()
    assert pipeline.call_llm("x", tier="fast") is None
    assert calls == []  # not retried during the cooldown
    clock.advance(seconds=pipeline.AI_COOLDOWN_SECONDS + 1)
    pipeline.call_llm("x", tier="fast")
    assert calls


def test_deep_threshold_rises_when_the_deep_model_is_nearly_spent(clock):
    assert pipeline.deep_impact_threshold() == pipeline.DEEP_ANALYSIS_MIN_IMPACT
    pipeline.state.model_tokens_today[pipeline.GROQ_DEEP_MODEL] = int(0.85 * pipeline.GROQ_DAILY_TOKEN_LIMIT)
    assert pipeline.deep_impact_threshold() == pipeline.DEEP_IMPACT_WHEN_LOW_BUDGET


def test_deep_threshold_rises_while_the_deep_model_is_rate_limited(clock):
    pipeline._mark_unavailable(pipeline.GROQ_DEEP_MODEL, "Please try again in 10m0s", 429)
    assert pipeline.deep_impact_threshold() == pipeline.DEEP_IMPACT_WHEN_LOW_BUDGET


HEADLINES = ["RBI hikes repo rate in surprise move", "Local diesel prices above average", "Nike shares plummet 10%"]


def _triage(monkeypatch, answer):
    monkeypatch.setattr(pipeline, "call_llm", lambda prompt, tier="deep", temperature=0.0: answer)
    return pipeline.triage_headlines(HEADLINES)


def test_triage_reads_scores_and_drops_low_impact(monkeypatch):
    result = _triage(monkeypatch, json.dumps([
        {"n": 1, "impact": 9, "direction": "bearish", "region": "india", "tickers": ["^NSEI"], "sector": "Finance"},
        {"n": 2, "impact": 2, "direction": "MIXED", "region": "Global", "tickers": [], "sector": "Energy"},
        {"n": 3, "impact": 8, "direction": "BEARISH", "region": "Global", "tickers": ["NKE", "LULU", "DECK", "ADDYY"], "sector": "Apparel"},
    ]))
    assert set(result) == {HEADLINES[0], HEADLINES[2]}
    assert result[HEADLINES[0]] == {"impact": 9, "fresh": True, "direction": "BEARISH", "region": "India", "tickers": ["^NSEI"], "sector": "Finance"}
    assert result[HEADLINES[2]]["tickers"] == ["NKE", "LULU", "DECK"]  # capped at three


def test_triage_tolerates_code_fences_and_odd_values(monkeypatch):
    result = _triage(monkeypatch, '```json\n[{"n": 1, "impact": 12, "direction": "sideways", "region": "Mars"}, {"n": 99, "impact": 8}, {"impact": 8}]\n```')
    assert result == {HEADLINES[0]: {"impact": 10, "fresh": True, "direction": "MIXED", "region": None, "tickers": [], "sector": "General Markets"}}


def test_triage_reads_the_fresh_flag(monkeypatch):
    # "Nike shares plummet 10%" reports a move that already happened: fresh is false. A
    # missing flag counts as fresh, so a model that drops the field cannot silence every call.
    result = _triage(monkeypatch, json.dumps([
        {"n": 1, "impact": 9, "fresh": True, "direction": "BEARISH"},
        {"n": 2, "impact": 7, "direction": "BULLISH"},
        {"n": 3, "impact": 8, "fresh": False, "direction": "BEARISH", "tickers": ["NKE"]},
    ]))
    assert [result[h]["fresh"] for h in HEADLINES] == [True, True, False]


def test_triage_with_no_qualifying_headlines(monkeypatch):
    assert _triage(monkeypatch, "[]") == {}
    assert _triage(monkeypatch, "I could not find anything relevant.") == {}


def test_triage_returns_none_when_every_provider_is_down(monkeypatch):
    # None, not {}: the caller requeues the batch instead of discarding it.
    assert _triage(monkeypatch, None) is None


@pytest.fixture
def loop(monkeypatch, clock):
    """Runs run_pipeline_cycle with scanning, triage and deep analysis replaced."""
    record = {"triaged": [], "deep": [], "triage_result": {}}
    monkeypatch.setattr(pipeline, "fetch_live_financial_news", lambda: [])
    monkeypatch.setattr(pipeline, "append_signal", lambda alert: None)
    monkeypatch.setattr(pipeline, "resolve_ticker", lambda t: t)

    def fake_triage(batch):
        record["triaged"].append(list(batch))
        result = record["triage_result"]
        return None if result is None else {h: result[h] for h in batch if h in result}

    def fake_deep(headline, sector, ticker):
        record["deep"].append(headline)
        return {"sentiment": "BEARISH", "sector": sector, "summary": "s", "buy_targets": [], "sell_targets": ["NKE"],
                "strategy": "st", "category": "Signal", "key_takeaways": [], "ripple_effects": [], "market_data": None,
                "grounded_ticker": ticker}

    monkeypatch.setattr(pipeline, "triage_headlines", fake_triage)
    monkeypatch.setattr(pipeline, "run_deep_analysis", fake_deep)

    def queue(headline, priority=False, waited=0, published_minutes_ago=1):
        pipeline.state.pending_headlines.append(headline)
        published = clock.utc() - timedelta(minutes=published_minutes_ago)
        pipeline.state.headline_metadata[headline] = {
            "published_at": published.isoformat() + "Z",
            "queued_at": clock.now - waited, "priority": priority, "source": "Test", "feed": "Test", "region": "Global",
        }

    def run():
        pipeline.state.last_scan_time = clock.now  # scanning is not what these tests exercise
        return pipeline.run_pipeline_cycle()

    record["queue"], record["run"] = queue, run
    return record


def test_a_few_ordinary_headlines_wait_for_a_batch(loop):
    loop["queue"]("a", waited=10)
    loop["queue"]("b", waited=5)
    loop["run"]()
    assert loop["triaged"] == []


def test_waiting_headlines_are_triaged_after_the_maximum_wait(loop):
    loop["queue"]("a", waited=pipeline.TRIAGE_MAX_WAIT_SECONDS + 1)
    loop["queue"]("b", waited=5)
    loop["run"]()
    assert sorted(loop["triaged"][0]) == ["a", "b"]


def test_a_full_batch_is_triaged_at_once(loop):
    for i in range(pipeline.TRIAGE_MIN_BATCH):
        loop["queue"](f"h{i}", waited=1)
    loop["run"]()
    assert len(loop["triaged"][0]) == pipeline.TRIAGE_MIN_BATCH


def test_priority_headline_takes_the_fast_lane_and_brings_the_rest(loop):
    loop["queue"]("ordinary", waited=5)
    loop["queue"]("from the wire", priority=True, waited=1)
    loop["run"]()
    assert sorted(loop["triaged"][0]) == ["from the wire", "ordinary"]


def test_fast_lane_is_rate_limited(loop, clock):
    loop["queue"]("wire 1", priority=True)
    loop["run"]()
    clock.advance(seconds=20)
    loop["queue"]("wire 2", priority=True)
    loop["run"]()
    assert len(loop["triaged"]) == 1
    clock.advance(seconds=pipeline.FAST_LANE_MIN_INTERVAL_SECONDS)
    loop["run"]()
    assert len(loop["triaged"]) == 2


def test_fast_lane_switches_off_when_the_triage_budget_is_low(loop):
    pipeline.state.model_tokens_today[pipeline.GROQ_FAST_MODEL] = int(0.85 * pipeline.GROQ_DAILY_TOKEN_LIMIT)
    loop["queue"]("from the wire", priority=True, waited=1)
    loop["run"]()
    assert loop["triaged"] == []


def test_stale_headlines_are_dropped_from_the_queue(loop):
    loop["queue"]("old news", waited=pipeline.TRIAGE_MAX_WAIT_SECONDS + 1, published_minutes_ago=pipeline.MAX_HEADLINE_AGE_MINUTES + 5)
    loop["run"]()
    assert loop["triaged"] == [] and pipeline.state.pending_headlines == []


def test_batch_is_requeued_when_ai_is_down(loop):
    loop["triage_result"] = None
    loop["queue"]("a", priority=True)
    loop["run"]()
    assert pipeline.state.pending_headlines == ["a"]


def test_impact_decides_between_dropped_quick_and_deep(loop):
    triage = lambda impact: {"impact": impact, "direction": "BEARISH", "region": "India", "tickers": ["NKE"], "sector": "Apparel"}
    loop["triage_result"] = {"minor": triage(5), "big": triage(8)}
    for headline in ("minor", "big", "ignored"):
        loop["queue"](headline, priority=True)
    loop["run"]()

    by_headline = {a["Headline"]: a for a in pipeline.state.trade_history}
    assert set(by_headline) == {"minor", "big"}  # "ignored" scored below the minimum
    assert by_headline["minor"]["Analysis"] == "Quick" and by_headline["minor"]["Sell Tickers"] == "NKE"
    assert by_headline["big"]["Analysis"] == "Deep" and loop["deep"] == ["big"]
    assert by_headline["big"]["Region"] == "India"  # the triage region, not the feed's
    assert by_headline["big"]["Feed"] == "Test" and by_headline["big"]["Fast Lane"] is True
    assert all(a["Seen At"] is None or isinstance(a["Seen At"], str) for a in by_headline.values())


def test_a_move_that_already_happened_gets_no_call_and_no_deep_analysis(loop):
    # Seen live: "US Telecom Stocks Sink 9%" published as BEARISH on VZ after the fall, and
    # "Gold hits two-month low" as BEARISH on gold, which then bounced. Chasing a finished
    # move is a coin flip, so a stale headline is shown for reference with no call on it.
    loop["triage_result"] = {
        "Nike shares plummet 10% after earnings": {"impact": 9, "fresh": False, "direction": "BEARISH", "region": "Global", "tickers": ["NKE"], "sector": "Apparel"},
    }
    loop["queue"]("Nike shares plummet 10% after earnings", priority=True)
    loop["run"]()
    loop["run"]()
    (alert,) = pipeline.state.trade_history
    assert loop["deep"] == []
    assert alert["Analysis"] == "Quick" and alert["Catalyst"] == "Priced in"
    assert alert["Sentiment"] == "NEUTRAL" and alert["Buy Tickers"] == "" and alert["Sell Tickers"] == ""
    assert alert["Tickers"] == "NKE"  # still named, for the reader
    assert "already happened" in alert["Execution Blueprint"]


def test_quick_signal_puts_a_currency_on_the_side_its_pair_moves(loop):
    # "Rupee rises" is bullish on the rupee; the Yahoo pair INR=X (rupees per dollar) falls.
    loop["triage_result"] = {"Rupee rises 23 paise": {"impact": 6, "fresh": True, "direction": "BULLISH", "region": "Forex", "tickers": ["INR"], "sector": "Forex"}}
    loop["queue"]("Rupee rises 23 paise", priority=True)
    loop["run"]()
    (alert,) = pipeline.state.trade_history
    assert alert["Sentiment"] == "BULLISH" and alert["Buy Tickers"] == "" and alert["Sell Tickers"] == "INR=X"
    assert alert["Catalyst"] == "Fresh" and alert["Ticker Names"] == {"INR=X": "USD/INR"}


def test_failed_deep_analysis_still_publishes_a_quick_signal(loop, monkeypatch):
    monkeypatch.setattr(pipeline, "run_deep_analysis", lambda *a: {"sentiment": "ERROR", "market_data": None, "grounded_ticker": None})
    loop["triage_result"] = {"big": {"impact": 9, "direction": "BULLISH", "region": "Global", "tickers": ["AAPL"], "sector": "Tech"}}
    loop["queue"]("big", priority=True)
    loop["run"]()
    assert [(a["Headline"], a["Analysis"], a["Buy Tickers"]) for a in pipeline.state.trade_history] == [("big", "Quick", "AAPL")]


def test_paused_pipeline_does_nothing(loop):
    pipeline.state.active = False
    loop["queue"]("from the wire", priority=True)
    loop["run"]()
    assert loop["triaged"] == []
