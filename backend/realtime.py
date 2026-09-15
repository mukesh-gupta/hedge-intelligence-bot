import asyncio
import json
import os

import websockets

from backend import pipeline

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_WS_URL = f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}"

# Maps a watchlist symbol (as stored/added by the user, yfinance-style) to the
# symbol Finnhub's websocket expects. Symbols not listed here pass through
# unchanged, which is correct for plain US equity tickers (AAPL, TSLA, MS,
# GS, ...) since Finnhub uses the same ticker format for those. Anything we
# can't confidently map (indices, other futures/forex) returns None and is
# simply left off the live feed — it keeps using the existing 45s-warmed
# yfinance path instead.
SYMBOL_MAP = {
    "GC=F": "OANDA:XAU_USD",
    "SI=F": "OANDA:XAG_USD",
    "BTC-USD": "BINANCE:BTCUSDT",
}

_UNMAPPABLE_SUFFIXES = ("=F", ".NYB")


def to_finnhub_symbol(watchlist_symbol: str) -> str | None:
    if watchlist_symbol in SYMBOL_MAP:
        return SYMBOL_MAP[watchlist_symbol]
    if watchlist_symbol.startswith("^") or watchlist_symbol.endswith(_UNMAPPABLE_SUFFIXES):
        return None
    return watchlist_symbol


class RealtimeStore:
    """Holds one shared Finnhub websocket connection for the whole process and
    fans out every trade tick to any number of SSE subscribers via per-client
    asyncio.Queues. There is exactly one upstream connection regardless of how
    many browser tabs are connected — Finnhub's free tier is rate/connection
    limited, so this matters."""

    def __init__(self):
        self.latest: dict[str, dict] = {}  # finnhub_symbol -> {"price", "t"}
        self._subscribers: set[asyncio.Queue] = set()
        self._ws = None
        self._subscribed: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def add_subscriber(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        return q

    def remove_subscriber(self, q: asyncio.Queue):
        self._subscribers.discard(q)

    def _broadcast(self, update: dict):
        for q in list(self._subscribers):
            try:
                q.put_nowait(update)
            except asyncio.QueueFull:
                pass  # a slow client falling behind shouldn't block everyone else

    async def _ensure_subscribed(self, ws, desired: set[str]):
        for sym in desired - self._subscribed:
            await ws.send(json.dumps({"type": "subscribe", "symbol": sym}))
        for sym in self._subscribed - desired:
            await ws.send(json.dumps({"type": "unsubscribe", "symbol": sym}))
        self._subscribed = set(desired)

    def desired_symbols(self) -> set[str]:
        symbols = {to_finnhub_symbol(w["symbol"]) for w in pipeline.state.watchlist}
        symbols.discard(None)
        return symbols

    def resubscribe_from_any_thread(self):
        """Call after a watchlist mutation. Safe to call from FastAPI's sync
        request-handling threads, not just the event loop thread."""
        if self._loop is None or self._ws is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._ensure_subscribed(self._ws, self.desired_symbols()), self._loop
        )

    async def run(self):
        if not FINNHUB_API_KEY:
            return  # feature simply stays off; REST/warmed-cache path is unaffected
        self._loop = asyncio.get_running_loop()
        backoff = 1
        while True:
            try:
                async with websockets.connect(FINNHUB_WS_URL, open_timeout=10) as ws:
                    self._ws = ws
                    self._subscribed = set()
                    backoff = 1
                    await self._ensure_subscribed(ws, self.desired_symbols())
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("type") != "trade":
                            continue
                        for tick in msg.get("data", []):
                            update = {"symbol": tick["s"], "price": tick["p"], "t": tick["t"]}
                            self.latest[tick["s"]] = update
                            self._broadcast(update)
            except Exception as e:
                pipeline.report_error("Finnhub realtime", e)
            finally:
                self._ws = None
            await asyncio.sleep(min(backoff, 30))
            backoff *= 2


store = RealtimeStore()
