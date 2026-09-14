import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/api/signals")
def get_signals():
    return {"signals": pipeline.state.trade_history}


@app.get("/api/ticker-bar")
def get_ticker_bar():
    rows = []
    for symbol, label in pipeline.TICKER_BAR_SYMBOLS:
        data = pipeline.fetch_ticker_bar_data(symbol)
        rows.append({"symbol": symbol, "label": label, "data": data})
    return {"ticker_bar": rows}


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
    }


@app.post("/api/scan")
def trigger_scan():
    new_headlines = pipeline.fetch_live_financial_news()
    pipeline.state.pending_headlines.extend(new_headlines)
    return {"new_headlines_found": len(new_headlines)}
