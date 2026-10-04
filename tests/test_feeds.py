"""Feed parsing and ingestion: what becomes a queued headline, and with what metadata."""
from datetime import datetime, timedelta

import pytest

from backend import pipeline

RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>Feed</title>
<item>
  <title><![CDATA[Bajaj Auto shares crash 9%, M&amp;M hits 52-week low]]></title>
  <link>https://example.com/a</link>
  <pubDate>Fri, 02 Oct 2026 21:57:21 +0530</pubDate>
</item>
<item>
  <title>Fed holds rates   steady - Reuters</title>
  <link>https://example.com/b</link>
  <pubDate>Fri, 02 Oct 2026 16:00:00 GMT</pubDate>
  <source url="https://reuters.com">Reuters</source>
</item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry>
  <title>ECB publishes decision</title>
  <link href="https://example.com/ecb"/>
  <updated>2026-10-02T12:00:00Z</updated>
</entry>
</feed>"""

NO_TIMEZONE = b"""<rss><channel><item>
  <title>RBI appoints new Executive Director</title>
  <pubDate>Fri, 02 Oct 2026 17:05:00</pubDate>
</item></channel></rss>"""


def test_rss_dates_are_converted_to_utc():
    items = pipeline._parse_feed_fast(RSS)
    assert items[0][2] == datetime(2026, 10, 2, 16, 27, 21)  # 21:57 IST
    assert items[1][2] == datetime(2026, 10, 2, 16, 0, 0)
    assert items[1][3] == "Reuters"


def test_atom_entries_are_parsed():
    title, link, published, publisher = pipeline._parse_feed_fast(ATOM)[0]
    assert (title, link, published, publisher) == ("ECB publishes decision", "https://example.com/ecb", datetime(2026, 10, 2, 12, 0), None)


def test_date_without_timezone_falls_back_to_feedparser():
    # An unlabeled time can't be placed (RBI's are IST), so the fast parser declines the feed.
    assert pipeline._parse_feed_fast(NO_TIMEZONE) is None


def test_malformed_xml_falls_back_to_feedparser():
    assert pipeline._parse_feed_fast(b"<rss><channel><item><title>Unclosed") is None


def _fake_get(body, headers=None):
    return lambda url, request_headers=None: (body, headers or {})


def test_fetch_rss_cleans_titles(monkeypatch):
    monkeypatch.setattr(pipeline, "_http_get", _fake_get(RSS))
    items = pipeline._fetch_rss({"name": "Test", "url": "https://example.com/rss"})
    assert items[0][0] == "Bajaj Auto shares crash 9%, M&M hits 52-week low"  # entity unescaped
    assert items[1][0] == "Fed holds rates steady"  # publisher suffix and extra spaces removed


def test_fetch_rss_skips_an_unchanged_feed(monkeypatch):
    source = {"name": "Test", "url": "https://example.com/rss"}
    monkeypatch.setattr(pipeline, "_http_get", _fake_get(RSS))
    assert pipeline._fetch_rss(source) is not None
    assert pipeline._fetch_rss(source) is None  # same bytes as last time: not parsed again
    monkeypatch.setattr(pipeline, "_http_get", lambda url, headers=None: (None, None))
    assert pipeline._fetch_rss(source) is None  # 304 Not Modified


def test_fetch_rss_sends_validators_from_the_previous_response(monkeypatch):
    source = {"name": "Test", "url": "https://example.com/rss"}
    seen_headers = []

    def fake_get(url, headers=None):
        seen_headers.append(headers)
        return RSS, {"ETag": '"abc"', "Last-Modified": "Fri, 02 Oct 2026 16:00:00 GMT"}

    monkeypatch.setattr(pipeline, "_http_get", fake_get)
    pipeline._fetch_rss(source)
    pipeline._fetch_rss(source)
    assert seen_headers[0] == {}
    assert seen_headers[1] == {"If-None-Match": '"abc"', "If-Modified-Since": "Fri, 02 Oct 2026 16:00:00 GMT"}


SOURCES = [
    {"name": "Wire", "url": "https://example.com/wire", "region": "Global", "priority": True},
    {"name": "Blog", "url": "https://example.com/blog", "region": "India"},
]


@pytest.fixture
def feeds(monkeypatch, clock):
    """Lets a test say what each source returns on the next poll."""
    content = {}
    monkeypatch.setattr(pipeline, "NEWS_SOURCES", SOURCES)
    monkeypatch.setattr(pipeline, "_fetch_source", lambda source: (source, content.get(source["name"])))

    def minutes_ago(minutes):
        return clock.utc() - timedelta(minutes=minutes)

    content["minutes_ago"] = minutes_ago
    return content


def test_ingestion_keeps_fresh_relevant_headlines(feeds):
    ago = feeds["minutes_ago"]
    feeds["Wire"] = [
        ("Fed cuts rates by 50 bps in emergency move", "https://example.com/1", ago(5), None),
        ("Fed cut rates last month, a look back", "https://example.com/2", ago(90), None),  # too old
        ("Local bakery wins award", "https://example.com/3", ago(5), None),  # not market news
    ]
    assert pipeline.fetch_live_financial_news() == ["Fed cuts rates by 50 bps in emergency move"]
    meta = pipeline.state.headline_metadata["Fed cuts rates by 50 bps in emergency move"]
    assert meta["feed"] == "Wire" and meta["region"] == "Global" and meta["priority"] is True
    assert meta["link"] == "https://example.com/1"
    # Rejected headlines are still marked seen, so they aren't re-examined every scan.
    assert "Local bakery wins award" in pipeline.state.processed_headlines


def test_same_headline_is_not_queued_twice(feeds):
    item = ("Oil prices surge 4% on supply shock", None, feeds["minutes_ago"](2), None)
    feeds["Wire"] = [item]
    assert len(pipeline.fetch_live_financial_news()) == 1
    feeds["Blog"] = [item]
    assert pipeline.fetch_live_financial_news() == []


def test_reworded_duplicate_from_another_outlet_is_dropped(feeds):
    ago = feeds["minutes_ago"]
    feeds["Wire"] = [("Hong Kong Stocks Suffer Biggest Slump Since March", None, ago(3), None)]
    feeds["Blog"] = [("Hong Kong Stocks Slump Most Since March, Led by Financials", None, ago(2), None)]
    assert len(pipeline.fetch_live_financial_news()) == 1


def test_first_poll_after_startup_is_marked_backfill(feeds, clock):
    feeds["Wire"] = [("Gold jumps to record high", None, feeds["minutes_ago"](30), None)]
    pipeline.fetch_live_financial_news()
    assert pipeline.state.headline_metadata["Gold jumps to record high"]["backfill"] is True

    clock.advance(minutes=1)
    feeds["Wire"] = [("Copper slides on China demand worries", None, feeds["minutes_ago"](1), None)]
    pipeline.fetch_live_financial_news()
    assert pipeline.state.headline_metadata["Copper slides on China demand worries"]["backfill"] is False


def test_failed_fetch_does_not_count_as_the_first_poll(feeds, clock):
    feeds["Wire"] = None  # fetch failed
    pipeline.fetch_live_financial_news()
    feeds["Wire"] = [("Gold jumps to record high", None, feeds["minutes_ago"](30), None)]
    pipeline.fetch_live_financial_news()
    assert pipeline.state.headline_metadata["Gold jumps to record high"]["backfill"] is True


def test_undated_items_are_new_only_after_the_first_poll(feeds, clock):
    feeds["Wire"] = [("RBI announces repo rate decision", None, None, None)]
    assert pipeline.fetch_live_financial_news() == []  # no way to tell its age on first sight
    clock.advance(minutes=1)
    feeds["Wire"] = [("RBI announces repo rate decision", None, None, None), ("RBI hikes repo rate by 25 bps", None, None, None)]
    assert pipeline.fetch_live_financial_news() == ["RBI hikes repo rate by 25 bps"]


def test_sources_are_polled_no_more_often_than_their_interval(monkeypatch, clock):
    polled = []
    monkeypatch.setattr(pipeline, "NEWS_SOURCES", [
        {"name": "Every scan", "url": "u1", "region": "Global"},
        {"name": "Every 2 min", "url": "u2", "region": "Global", "every": 120},
    ])
    monkeypatch.setattr(pipeline, "_fetch_source", lambda source: (polled.append(source["name"]), (source, None))[1])
    pipeline.fetch_live_financial_news()
    clock.advance(seconds=30)
    pipeline.fetch_live_financial_news()
    clock.advance(seconds=100)
    pipeline.fetch_live_financial_news()
    assert polled == ["Every scan", "Every 2 min", "Every scan", "Every scan", "Every 2 min"]
