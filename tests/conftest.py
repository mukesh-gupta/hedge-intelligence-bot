"""Shared test setup. Tests never touch the network, Upstash or an AI provider."""
import calendar
import json
import os
import socket
import time as real_time
from datetime import datetime as real_datetime
from datetime import timezone

import pytest

# Must happen before `backend` is imported. Set-but-empty rather than unset: the pipeline
# calls load_dotenv(), which would otherwise pull a developer's real keys in from .env.
for _name in (
    "UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN", "FINNHUB_API_KEY", "GEMINI_API_KEY",
    "OPENROUTER_API_KEY", "ALPHAVANTAGE_API_KEY", "ADMIN_TOKEN", "RENDER_EXTERNAL_URL",
):
    os.environ[_name] = ""
os.environ["GROQ_API_KEY"] = "test-key"  # the Groq client refuses to construct without one

from backend import outcomes, pipeline, storage  # noqa: E402


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any real connection attempt fails the test instead of silently reaching the internet.
    Loopback is allowed: asyncio builds its own socket pair on Windows."""
    real_connect = socket.socket.connect

    def guarded_connect(sock, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise RuntimeError(f"test tried to open a network connection to {address!r}")
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """Every test starts from an empty pipeline: the modules keep their state at module
    level, so without this one test's signals would leak into the next."""
    monkeypatch.setattr(pipeline, "state", pipeline.PipelineState())
    for registry in (
        pipeline._source_last_polled, pipeline._feed_validators, pipeline._feed_body_hashes,
        pipeline._undated_titles, pipeline._polled_sources, pipeline._recent_story_tokens,
        outcomes.pending, outcomes.stats, outcomes.recent, outcomes._quote_cache,
    ):
        registry.clear()
    monkeypatch.setattr(outcomes, "_stats_day", None)


class Clock:
    """A clock the test controls. `now` is epoch seconds (UTC)."""

    def __init__(self):
        self.now = calendar.timegm((2026, 10, 5, 4, 0, 0))  # Mon 5 Oct 2026, 09:30 IST

    def set(self, year, month, day, hour=0, minute=0, second=0):
        self.now = calendar.timegm((year, month, day, hour, minute, second))

    def advance(self, hours=0, minutes=0, seconds=0):
        self.now += hours * 3600 + minutes * 60 + seconds

    def utc(self):
        return real_datetime.fromtimestamp(self.now, timezone.utc).replace(tzinfo=None)

    def iso(self):
        return self.utc().isoformat() + "Z"


@pytest.fixture
def clock(monkeypatch):
    """Replaces the clock the pipeline and result tracking read, so a test can step across
    midnight or a day in an instant."""
    fake = Clock()

    class FakeDateTime(real_datetime):
        @classmethod
        def utcnow(cls):
            return fake.utc()

    class FakeTime:
        @staticmethod
        def time():
            return fake.now

        @staticmethod
        def sleep(_seconds):
            pass

        def __getattr__(self, name):
            return getattr(real_time, name)

    monkeypatch.setattr(pipeline, "datetime", FakeDateTime)
    monkeypatch.setattr(pipeline, "time", FakeTime())
    monkeypatch.setattr(outcomes, "time", FakeTime())
    pipeline.state.token_usage_date = pipeline.local_today()
    return fake


class FakeUpstash:
    """An in-memory stand-in for Upstash's REST API, covering the commands the bot uses."""

    def __init__(self):
        self.db = {}
        self.commands = []

    def run(self, command):
        self.commands.append(command)
        op, key, *args = command
        if op == "LRANGE":
            return list(self.db.get(key, []))
        if op == "LLEN":
            return len(self.db.get(key, []))
        if op == "LPUSH":
            for value in args:
                self.db.setdefault(key, []).insert(0, value)
            return len(self.db[key])
        if op == "RPUSH":
            self.db.setdefault(key, []).extend(args)
            return len(self.db[key])
        if op == "LTRIM":
            self.db[key] = self.db.get(key, [])[int(args[0]):int(args[1]) + 1]
            return "OK"
        if op == "DEL":
            return 1 if self.db.pop(key, None) is not None else 0
        if op == "HSET":
            self.db.setdefault(key, {}).update(dict(zip(args[0::2], args[1::2])))
            return len(args) // 2
        if op == "HDEL":
            return sum(1 for field in args if self.db.get(key, {}).pop(field, None) is not None)
        if op == "HGETALL":
            return [x for pair in self.db.get(key, {}).items() for x in pair]
        raise ValueError(f"FakeUpstash does not implement {op}")

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    def post(self, url, headers=None, data=None, timeout=None):
        if url.endswith("/pipeline"):
            return self._Response([{"result": self.run(c)} for c in json.loads(data)])
        self.db[url.split("/set/")[1]] = data
        return self._Response({"result": "OK"})

    def get(self, url, headers=None, timeout=None):
        return self._Response({"result": self.db.get(url.split("/get/")[1])})


@pytest.fixture
def upstash(monkeypatch):
    fake = FakeUpstash()
    monkeypatch.setattr(storage, "UPSTASH_URL", "https://fake.upstash.io")
    monkeypatch.setattr(storage, "UPSTASH_TOKEN", "token")
    monkeypatch.setattr(storage._session, "post", fake.post)
    monkeypatch.setattr(storage._session, "get", fake.get)
    return fake


@pytest.fixture
def publish(clock):
    """Publishes a signal through the real _publish_alert, as the pipeline would."""

    def _publish(headline, sentiment="BULLISH", buy="", sell="", **fields):
        alert = {
            "Timestamp": clock.iso(), "Headline": headline, "Sentiment": sentiment,
            "Buy Tickers": buy, "Sell Tickers": sell,
            "Impact": 7, "Analysis": "Quick", "Region": "Global", "Source": "Reuters",
            **fields,
        }
        pipeline._publish_alert(headline, alert)
        return alert

    return _publish
