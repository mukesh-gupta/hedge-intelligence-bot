import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend import pipeline, scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    yield
    scheduler.stop()


app = FastAPI(title="Hedge Intelligence Terminal API", lifespan=lifespan)

# Wide open for localhost dev — the future React dev server (Vite default: 5173) calls this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_response_time_header(request, call_next):
    """Exposes exactly how fast each response was server-side, in milliseconds — proof
    the GET endpoints are serving from the pre-warmed cache rather than blocking on a
    live yfinance/AlphaVantage call (which would show up here as seconds, not ms)."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


@app.get("/api/signals")
def get_signals():
    return {"signals": pipeline.state.trade_history}


@app.get("/api/ticker-bar")
def get_ticker_bar():
    """Reads strictly from the pre-warmed cache — never triggers a live fetch itself, so
    this always answers instantly even if yfinance is slow or Render's shared CPU is
    throttled at that moment. See prefetch_all_market_data() for why."""
    return {"ticker_bar": pipeline.state.cached_ticker_bar}


@app.get("/api/usage")
def get_usage():
    s = pipeline.state
    return {
        "groq_tokens_today": s.groq_tokens_today,
        "openrouter_tokens_today": s.openrouter_tokens_today,
        "gemini_tokens_today": s.gemini_tokens_today,
        "av_calls_today": s.av_calls_today,
        "headlines_queued": len(s.pending_headlines) + len(s.qualified_headlines),
        "ai_cooldown_remaining": max(0, int(s.ai_unavailable_until - time.time())),
        "last_error": s.last_error,
    }


@app.get("/api/status")
def get_status():
    s = pipeline.state
    cooldown_active = time.time() < s.ai_unavailable_until
    return {
        "news_feed": bool(s.last_scan_time > 0 or s.pending_headlines),
        "market_data": bool(s.market_data_cache) or pipeline.ALPHAVANTAGE_API_KEY is not None,
        "ai_engine": not cooldown_active,
        "data_pipeline": s.last_error is None or not cooldown_active,
        "cache_age_seconds": round(time.time() - s.cache_last_updated, 1) if s.cache_last_updated else None,
    }


@app.post("/api/scan")
def trigger_scan():
    new_headlines = pipeline.fetch_live_financial_news()
    pipeline.state.pending_headlines.extend(new_headlines)
    return {"new_headlines_found": len(new_headlines)}


@app.get("/api/market-regime")
def get_market_regime():
    return pipeline.compute_market_regime()


@app.get("/api/market-data")
def get_market_data(category: str = "indices", history: bool = True):
    if category not in pipeline.MARKET_DATA_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Unknown category '{category}'. Valid: {list(pipeline.MARKET_DATA_CATEGORIES.keys())}")
    rows = pipeline.state.cached_market_data.get(category, [])
    if not history:
        rows = [{k: v for k, v in row.items() if k != "history"} for row in rows]
    return {"category": category, "assets": rows}


@app.get("/api/sectors")
def get_sectors(timeframe: str = "1D"):
    if timeframe not in pipeline.TIMEFRAME_LOOKBACK_TRADING_DAYS:
        raise HTTPException(status_code=400, detail=f"Unknown timeframe '{timeframe}'. Valid: {list(pipeline.TIMEFRAME_LOOKBACK_TRADING_DAYS.keys())}")
    return {"timeframe": timeframe, "sectors": pipeline.state.cached_sectors.get(timeframe, [])}


@app.get("/api/watchlist")
def get_watchlist():
    return {"watchlist": pipeline.state.cached_watchlist}


class WatchlistAddRequest(BaseModel):
    symbol: str
    label: Optional[str] = None


@app.post("/api/watchlist")
def post_watchlist(body: WatchlistAddRequest):
    pipeline.add_to_watchlist(body.symbol, body.label)
    # Mutations are rare, user-initiated, and expected to reflect immediately — unlike the
    # GET above, it's fine for this one to do a real (small, ~10-symbol) live fetch inline
    # rather than waiting for the next scheduled warm cycle.
    return {"watchlist": pipeline.refresh_watchlist_cache()}


@app.delete("/api/watchlist/{symbol}")
def delete_watchlist(symbol: str):
    pipeline.remove_from_watchlist(symbol)
    return {"watchlist": pipeline.refresh_watchlist_cache()}


@app.get("/api/settings")
def get_settings():
    s = pipeline.state
    return {"active": s.active, "refresh_interval_seconds": s.refresh_interval_seconds}


class SettingsUpdateRequest(BaseModel):
    active: Optional[bool] = None
    refresh_interval_seconds: Optional[int] = None


@app.patch("/api/settings")
def patch_settings(body: SettingsUpdateRequest):
    s = pipeline.state
    if body.active is not None:
        s.active = body.active
    if body.refresh_interval_seconds is not None:
        if not (5 <= body.refresh_interval_seconds <= 3600):
            raise HTTPException(status_code=400, detail="refresh_interval_seconds must be between 5 and 3600")
        s.refresh_interval_seconds = body.refresh_interval_seconds
    return {"active": s.active, "refresh_interval_seconds": s.refresh_interval_seconds}
