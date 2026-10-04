"""Result tracking: did a signal's call come true?

When a signal with a direction and a ticker is published, the ticker's price is recorded.
It is checked again after each horizon (1 hour, 1 day) and the call is scored right, wrong
or flat. Scores are added to a per-day scorecard that /api/scorecard aggregates.

All of this lives in its own Upstash keys, which the midnight signal wipe never touches:
a signal is deleted at the end of its day, but its 1-day result arrives the day after."""
import collections
import json
import time
import urllib.parse
from datetime import datetime, timedelta

import yfinance as yf

from backend import pipeline, storage

HORIZONS = {"1h": 3600, "1d": 86400}
# A move smaller than this in either direction is "flat": neither right nor wrong.
MIN_MOVE_PERCENT = 0.1
# A market that hasn't traded since the signal can't confirm or refute it. The 1-day check
# then waits for the next session (a Saturday signal is scored on Monday) and gives up
# after GIVE_UP_SECONDS; the 1-hour check is simply recorded as "closed".
RETRY_SECONDS = 3 * 3600
GIVE_UP_SECONDS = 4 * 86400
QUOTE_RETRY_SECONDS = 600
MAX_QUOTE_FAILURES = 6
# Price lookups per run, so a backlog after a restart can't hog the CPU.
MAX_CHECKS_PER_RUN = 25
QUOTE_CACHE_SECONDS = 60
STATS_KEEP_DAYS = 30
RECENT_KEEP = 200

PENDING_KEY = "outcome_pending"  # hash: signal id -> prediction awaiting its checks
STATS_KEY = "outcome_stats"  # hash: day (YYYY-MM-DD, of the signal) -> that day's tallies
RECENT_KEY = "outcome_recent"  # list: latest scored results, newest first

RIGHT, WRONG, FLAT, CLOSED = range(4)  # positions in a tally: [right, wrong, flat, closed]
_OUTCOME_INDEX = {"right": RIGHT, "wrong": WRONG, "flat": FLAT, "closed": CLOSED}

pending = {}
stats = {}  # day -> horizon -> "dimension:value" -> [right, wrong, flat, closed]
recent = collections.deque(maxlen=RECENT_KEEP)
_quote_cache = {}
_stats_day = None


def signal_id(alert):
    """Same identity the frontend derives for a signal: publish time plus headline."""
    return f"{alert.get('Timestamp')}__{alert.get('Headline')}"


def fetch_quote(symbol):
    """Latest price and the time of the last trade, as {"price", "trade_time"}; None if
    unavailable. Yahoo's chart endpoint directly: one small JSON response and no pandas,
    a fraction of the CPU of a yfinance history download. yfinance is the fallback in case
    Yahoo refuses the direct request."""
    cached = _quote_cache.get(symbol)
    if cached and time.time() - cached["fetched_at"] < QUOTE_CACHE_SECONDS:
        return cached["quote"]
    quote = None
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?range=1d&interval=1m"
        body, _ = pipeline._http_get(url)
        meta = json.loads(body)["chart"]["result"][0]["meta"]
        quote = {"price": float(meta["regularMarketPrice"]), "trade_time": int(meta["regularMarketTime"])}
    except Exception:
        try:
            meta = yf.Ticker(symbol).get_history_metadata()
            quote = {"price": float(meta["regularMarketPrice"]), "trade_time": int(meta["regularMarketTime"].timestamp())}
        except Exception:
            quote = None
    if quote and quote["price"] > 0:
        _quote_cache[symbol] = {"quote": quote, "fetched_at": time.time()}
        return quote
    return None


def _prediction_from(alert):
    """The one directional call a signal makes: its sentiment, on the first ticker it names
    on that side (first buy ticker if bullish, first sell ticker if bearish). None when the
    signal is neutral or names no ticker — there is nothing to score."""
    sentiment = (alert.get("Sentiment") or "").upper()
    if "BULLISH" in sentiment:
        direction, tickers = "UP", alert.get("Buy Tickers")
    elif "BEARISH" in sentiment:
        direction, tickers = "DOWN", alert.get("Sell Tickers")
    else:
        return None
    ticker = next((t.strip() for t in (tickers or "").split(",") if t.strip()), None)
    if not ticker:
        return None
    return {
        "id": signal_id(alert),
        "headline": alert.get("Headline"),
        "ticker": ticker,
        "direction": direction,
        "day": pipeline.local_today(),
        "impact": alert.get("Impact"),
        "analysis": alert.get("Analysis") or "Deep",
        "region": alert.get("Region"),
        "source": alert.get("Source"),
        "results": {h: None for h in HORIZONS},
        "quote_failures": 0,
    }


def _tally(prediction, horizon, outcome):
    day = stats.setdefault(prediction["day"], {}).setdefault(horizon, {})
    dimensions = {
        "overall": "all",
        "impact": prediction.get("impact"),
        "analysis": prediction.get("analysis"),
        "region": prediction.get("region"),
        "direction": prediction.get("direction"),
        "source": prediction.get("source"),
    }
    for dimension, value in dimensions.items():
        if value is not None:
            day.setdefault(f"{dimension}:{value}", [0, 0, 0, 0])[_OUTCOME_INDEX[outcome]] += 1


def _public_outcome(prediction):
    """What /api/signals shows on the signal itself."""
    return {
        "ticker": prediction["ticker"],
        "direction": prediction["direction"],
        "entry_price": prediction["entry_price"],
        **{h: prediction["results"].get(h) for h in HORIZONS},
    }


def _attach_to_signals(predictions):
    """Shows each prediction's progress on its signal in /api/signals. Only ever replaces
    the value of the "Outcome" key, which every signal already has (see
    pipeline._publish_alert): this runs on a worker thread while the API may be serializing
    the same dicts, and adding a key mid-serialization would raise."""
    by_id = {p["id"]: p for p in predictions}
    for alert in list(pipeline.state.trade_history):
        prediction = by_id.get(signal_id(alert))
        if prediction and "Outcome" in alert:
            alert["Outcome"] = _public_outcome(prediction)


def load():
    """Restores pending predictions, the scorecard and recent results. Runs at import,
    before the server accepts requests, so giving signals their "Outcome" key here is safe.
    Every tracked signal from today still has its 1-day check pending, so the pending
    predictions cover everything in the feed."""
    replies = storage.redis_pipeline([["HGETALL", PENDING_KEY], ["HGETALL", STATS_KEY], ["LRANGE", RECENT_KEY, 0, -1]])
    if replies is not None:
        raw_pending, raw_stats, raw_recent = replies
        pending.update({raw_pending[i]: json.loads(raw_pending[i + 1]) for i in range(0, len(raw_pending or []), 2)})
        stats.update({raw_stats[i]: json.loads(raw_stats[i + 1]) for i in range(0, len(raw_stats or []), 2)})
        recent.extend(json.loads(item) for item in reversed(raw_recent or []))
    for alert in pipeline.state.trade_history:
        alert.setdefault("Outcome", None)
    _attach_to_signals(pending.values())


def _drop_old_stats(commands):
    """Once a day: scorecard days older than STATS_KEEP_DAYS are removed."""
    global _stats_day
    today = pipeline.local_today()
    if _stats_day == today:
        return
    _stats_day = today
    cutoff = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=STATS_KEEP_DAYS)).strftime("%Y-%m-%d")
    old = [day for day in stats if day < cutoff]
    for day in old:
        del stats[day]
    if old:
        commands.append(["HDEL", STATS_KEY, *old])


def run_cycle():
    """One pass: start tracking newly published signals, then score every check that has
    come due. Called once a minute from the scheduler, off the event loop."""
    now = time.time()
    changed, touched_days, scored = {}, set(), []
    started = lookups = 0

    # 1. Newly published signals: record the entry price.
    queue = pipeline.state.outcome_queue
    while queue:
        prediction = _prediction_from(queue.popleft())
        if prediction is None or prediction["id"] in pending:
            continue
        quote = fetch_quote(prediction["ticker"])
        if quote is None:
            continue  # no price to compare against later: this signal isn't tracked
        prediction.update(
            entry_price=quote["price"],
            entry_trade_time=quote["trade_time"],
            entry_at=now,
            due={h: now + seconds for h, seconds in HORIZONS.items()},
        )
        pending[prediction["id"]] = changed[prediction["id"]] = prediction
        started += 1

    # 2. Checks that have come due, oldest first.
    due_checks = sorted(
        ((p["due"][h], p, h) for p in pending.values() for h in HORIZONS if p["results"][h] is None and p["due"][h] <= now),
        key=lambda check: check[0],
    )
    for _, prediction, horizon in due_checks:
        if lookups >= MAX_CHECKS_PER_RUN:
            break
        lookups += 1
        changed[prediction["id"]] = prediction
        quote = fetch_quote(prediction["ticker"])
        if quote is None:
            prediction["quote_failures"] += 1
            if prediction["quote_failures"] <= MAX_QUOTE_FAILURES:
                prediction["due"][horizon] = now + QUOTE_RETRY_SECONDS
                continue
            result = {"outcome": "closed", "checked_at": now}
        elif quote["trade_time"] <= prediction["entry_trade_time"]:
            # No trade since the signal: the market has been closed the whole time.
            if horizon != "1h" and now - prediction["entry_at"] < GIVE_UP_SECONDS:
                prediction["due"][horizon] = now + RETRY_SECONDS
                continue
            result = {"outcome": "closed", "checked_at": now}
        else:
            change = (quote["price"] - prediction["entry_price"]) / prediction["entry_price"] * 100
            in_favor = change if prediction["direction"] == "UP" else -change
            outcome = "right" if in_favor >= MIN_MOVE_PERCENT else "wrong" if in_favor <= -MIN_MOVE_PERCENT else "flat"
            result = {"outcome": outcome, "change_percent": round(change, 2), "price": quote["price"], "checked_at": now}
        prediction["results"][horizon] = result
        _tally(prediction, horizon, result["outcome"])
        touched_days.add(prediction["day"])
        scored.append({**{k: prediction.get(k) for k in ("id", "headline", "ticker", "direction", "entry_price", "impact", "analysis", "region", "source", "day")}, "horizon": horizon, **result})

    # 3. Show progress on the signals, then retire predictions with every horizon scored.
    recent.extend(scored)
    _attach_to_signals(changed.values())
    finished = [pid for pid, p in changed.items() if all(p["results"][h] is not None for h in HORIZONS)]
    for prediction_id in finished:
        del pending[prediction_id]
        del changed[prediction_id]

    # 4. Persist, with as few commands as possible (Upstash's free tier counts each one).
    commands = []
    if changed:
        commands.append(["HSET", PENDING_KEY, *[x for pid, p in changed.items() for x in (pid, json.dumps(p))]])
    if finished:
        commands.append(["HDEL", PENDING_KEY, *finished])
    if touched_days:
        commands.append(["HSET", STATS_KEY, *[x for day in touched_days for x in (day, json.dumps(stats[day]))]])
    if scored:
        commands.append(["LPUSH", RECENT_KEY, *[json.dumps(s) for s in scored]])
        commands.append(["LTRIM", RECENT_KEY, 0, RECENT_KEEP - 1])
    _drop_old_stats(commands)
    if commands and storage.is_configured() and storage.redis_pipeline(commands) is None:
        pipeline.report_error("Persisting signal results", "Upstash write failed")
    return {"started": started, "scored": len(scored)}


def _summary(tally):
    right, wrong, flat, closed = tally
    decided = right + wrong
    return {
        "right": right,
        "wrong": wrong,
        "flat": flat,
        "closed": closed,
        # Share of calls that moved the predicted way, among those that moved at all.
        "accuracy": round(right / decided * 100, 1) if decided else None,
    }


def scorecard(days=7):
    """Tallies for signals from the last `days` days, per horizon and broken down by impact,
    analysis type, region, direction and source."""
    today = datetime.strptime(pipeline.local_today(), "%Y-%m-%d")
    first_day = (today - timedelta(days=days - 1)).strftime("%Y-%m-%d")
    horizons = {}
    for horizon in HORIZONS:
        totals = {}
        for day, per_horizon in stats.items():
            if day < first_day:
                continue
            for key, tally in per_horizon.get(horizon, {}).items():
                merged = totals.setdefault(key, [0, 0, 0, 0])
                for i, count in enumerate(tally):
                    merged[i] += count
        breakdown = collections.defaultdict(list)
        for key, tally in totals.items():
            dimension, _, value = key.partition(":")
            breakdown[dimension].append({"value": value, **_summary(tally)})
        for rows in breakdown.values():
            rows.sort(key=lambda row: row["right"] + row["wrong"] + row["flat"], reverse=True)
        horizons[horizon] = {
            "overall": _summary(totals.get("overall:all", [0, 0, 0, 0])),
            "by_impact": sorted(breakdown["impact"], key=lambda row: -int(row["value"])),
            "by_analysis": breakdown["analysis"],
            "by_region": breakdown["region"],
            "by_direction": breakdown["direction"],
            "by_source": breakdown["source"][:10],
        }
    tracked_days = sorted(day for day in stats if day >= first_day)
    return {
        "days": days,
        "since": tracked_days[0] if tracked_days else None,
        "min_move_percent": MIN_MOVE_PERCENT,
        "pending": len(pending),
        "horizons": horizons,
        "recent": list(reversed(recent))[:50],
    }


load()
