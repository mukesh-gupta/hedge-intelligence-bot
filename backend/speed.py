"""How fast news becomes a signal, measured from today's signals.

The delay from an article's publish time to its signal has two parts with different causes:

  publish -> seen    how long after publishing the headline showed up in a feed the bot
                     polls. Mostly the source: feeds list articles late, and some are
                     polled less often. Nothing in the pipeline can shorten it except
                     choosing faster sources.
  seen -> signal     the bot's own time: waiting for triage, the AI calls, the queue.

Backfill signals (already in the feed when the bot first polled it after a restart) are
left out of publish -> seen: their lateness is the restart's, not the source's."""
import statistics
from datetime import datetime

from backend import pipeline


def _parse(iso):
    try:
        return datetime.fromisoformat(iso.removesuffix("Z"))
    except (AttributeError, ValueError):
        return None


def _summary(seconds):
    """Median and 90th percentile of a list of delays, in whole seconds."""
    if not seconds:
        return {"count": 0, "median": None, "p90": None}
    ordered = sorted(seconds)
    return {
        "count": len(ordered),
        "median": round(statistics.median(ordered)),
        "p90": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]),
    }


def stats():
    """Delays across today's signals: overall, per feed, by analysis type and by lane."""
    rows = []
    for alert in list(pipeline.state.trade_history):
        published, seen, processed = (_parse(alert.get(k)) for k in ("Timestamp", "Seen At", "Processed At"))
        if not (published and seen and processed):
            continue  # signals from before "Seen At" was recorded
        rows.append({
            "feed": alert.get("Feed") or "Unknown",
            "analysis": alert.get("Analysis") or "Deep",
            "lane": "Fast lane" if alert.get("Fast Lane") else "Batched",
            "backfill": bool(alert.get("Backfill")),
            # A publish time slightly in the future (clock skew, timezone slips) counts as 0.
            "publish_to_seen": max(0.0, (seen - published).total_seconds()),
            "seen_to_signal": max(0.0, (processed - seen).total_seconds()),
        })
    live = [r for r in rows if not r["backfill"]]

    def group(key, source_rows):
        groups = {}
        for row in source_rows:
            groups.setdefault(row[key], []).append(row)
        return groups

    return {
        "signals": len(rows),
        "backfill_excluded": len(rows) - len(live),
        "overall": {
            "publish_to_seen": _summary([r["publish_to_seen"] for r in live]),
            "seen_to_signal": _summary([r["seen_to_signal"] for r in rows]),
            "publish_to_signal": _summary([r["publish_to_seen"] + r["seen_to_signal"] for r in live]),
        },
        "by_feed": sorted(
            (
                {
                    "feed": feed,
                    "publish_to_seen": _summary([r["publish_to_seen"] for r in feed_rows]),
                    "seen_to_signal": _summary([r["seen_to_signal"] for r in feed_rows]),
                }
                for feed, feed_rows in group("feed", live).items()
            ),
            key=lambda row: -row["publish_to_seen"]["count"],
        ),
        "by_analysis": [
            {"analysis": analysis, "seen_to_signal": _summary([r["seen_to_signal"] for r in analysis_rows])}
            for analysis, analysis_rows in sorted(group("analysis", rows).items())
        ],
        "by_lane": [
            {"lane": lane, "seen_to_signal": _summary([r["seen_to_signal"] for r in lane_rows])}
            for lane, lane_rows in sorted(group("lane", rows).items(), reverse=True)
        ],
    }
