import concurrent.futures
import json
import os
import re
import time
from datetime import datetime

import feedparser
import requests
import yfinance as yf
from dotenv import load_dotenv
from groq import Groq

from backend import storage

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=15.0, max_retries=1)
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash-lite"

AI_COOLDOWN_SECONDS = 90
AV_MIN_SECONDS_BETWEEN_CALLS = 13
YF_CACHE_SECONDS = 60
BATCH_FILTER_SIZE = 20

TRADE_HISTORY_FILE = "trade_history.json"
MAX_STORED_ALERTS = 200
PROCESSED_HEADLINES_FILE = "processed_headlines.json"
MAX_STORED_HEADLINES = 500
WATCHLIST_FILE = "watchlist.json"
DEFAULT_WATCHLIST = [
    {"symbol": "NDAQ", "label": "NDAQ"},
    {"symbol": "MS", "label": "MS"},
    {"symbol": "GS", "label": "GS"},
    {"symbol": "ICE", "label": "ICE"},
    {"symbol": "NVDA", "label": "NVDA"},
    {"symbol": "ORCL", "label": "ORCL"},
    {"symbol": "AAPL", "label": "AAPL"},
    {"symbol": "TSLA", "label": "TSLA"},
    {"symbol": "GC=F", "label": "GOLD"},
    {"symbol": "BTC-USD", "label": "BTC"},
]


class PipelineState:
    """Plain, thread-safe-ish module-level state replacing st.session_state — there is
    no per-browser-session concept in a backend process, just one shared pipeline state."""

    def __init__(self):
        self.trade_history = load_trade_history()
        self.processed_headlines = load_processed_headlines()
        self.pending_headlines = []
        self.qualified_headlines = []
        self.headline_metadata = {}

        self.token_usage_date = datetime.now().strftime("%Y-%m-%d")
        self.groq_tokens_today = 0
        self.openrouter_tokens_today = 0
        self.gemini_tokens_today = 0
        self.av_calls_today = 0
        self.market_data_cache = {}
        self.symbol_cache = {}
        self.last_av_call_time = 0

        self.last_error = None
        self.ai_unavailable_until = 0
        self.last_scan_time = 0

        # Runtime-adjustable settings — mutable via PATCH /api/settings instead of being
        # fixed constants, so the React Settings screen can actually control the scheduler.
        self.refresh_interval_seconds = 30
        self.active = True

        self.watchlist = load_watchlist()

        # Pre-computed, request-ready results the warmer writes to and every GET endpoint
        # reads from directly — never live-fetched inline during a request, no matter how
        # stale, so a slow/throttled yfinance call can never show up as request latency.
        self.cached_ticker_bar = []
        self.cached_market_data = {}
        self.cached_sectors = {}
        self.cached_watchlist = []
        self.cache_last_updated = None

    def roll_daily_usage_if_needed(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if self.token_usage_date != today:
            self.token_usage_date = today
            self.groq_tokens_today = 0
            self.openrouter_tokens_today = 0
            self.gemini_tokens_today = 0
            self.av_calls_today = 0
            self.market_data_cache = {}
            self.symbol_cache = {}


def load_trade_history():
    if storage.is_configured():
        return storage.redis_get_json("trade_history", [])
    try:
        with open(TRADE_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_trade_history(history):
    trimmed = history[:MAX_STORED_ALERTS]
    if storage.is_configured():
        if not storage.redis_set_json("trade_history", trimmed):
            report_error("Persisting trade history", "Upstash write failed")
        return
    try:
        with open(TRADE_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(trimmed, f)
    except Exception as e:
        report_error("Persisting trade history", e)


def load_processed_headlines():
    if storage.is_configured():
        return set(storage.redis_get_json("processed_headlines", []))
    try:
        with open(PROCESSED_HEADLINES_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except Exception:
        return set()


def save_processed_headlines(headlines_set):
    trimmed = list(headlines_set)[-MAX_STORED_HEADLINES:]
    if storage.is_configured():
        if not storage.redis_set_json("processed_headlines", trimmed):
            report_error("Persisting seen-headlines", "Upstash write failed")
        return
    try:
        with open(PROCESSED_HEADLINES_FILE, "w", encoding="utf-8") as f:
            json.dump(trimmed, f)
    except Exception as e:
        report_error("Persisting seen-headlines", e)


def load_watchlist():
    if storage.is_configured():
        return storage.redis_get_json("watchlist", list(DEFAULT_WATCHLIST))
    try:
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return list(DEFAULT_WATCHLIST)


def save_watchlist(watchlist):
    if storage.is_configured():
        if not storage.redis_set_json("watchlist", watchlist):
            report_error("Persisting watchlist", "Upstash write failed")
        return
    try:
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(watchlist, f)
    except Exception as e:
        report_error("Persisting watchlist", e)


def report_error(source, message):
    """Record the failure on shared state instead of firing a Streamlit toast —
    the API layer surfaces this via GET /api/status."""
    state.last_error = {"source": source, "message": str(message)[:300], "time": datetime.now().strftime("%I:%M:%S %p")}
    print(f"[pipeline] {source} error: {str(message)[:200]}")


state = PipelineState()


def call_llm(prompt, temperature=0.0):
    """Try Groq first; if it genuinely fails (e.g. daily quota exhausted), fall back to
    OpenRouter, then Gemini. Returns None only if all three fail. No max_tokens cap —
    Groq's gpt-oss-120b is a reasoning model that spends a large, variable amount of its
    budget on hidden internal reasoning before writing the visible answer; capping output
    length caused it to hit the limit mid-thought and return empty responses."""
    if time.time() < state.ai_unavailable_until:
        return None

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        if completion.usage:
            state.groq_tokens_today += completion.usage.total_tokens
        return completion.choices[0].message.content.strip()
    except Exception as e:
        report_error("Groq", e)

    if OPENROUTER_API_KEY:
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json={"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": temperature},
                timeout=20
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("usage"):
                state.openrouter_tokens_today += payload["usage"].get("total_tokens", 0)
            return payload["choices"][0]["message"]["content"].strip()
        except Exception as e2:
            report_error("OpenRouter fallback", e2)

    if GEMINI_API_KEY:
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                params={"key": GEMINI_API_KEY},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature}
                },
                timeout=20
            )
            response.raise_for_status()
            payload = response.json()
            usage = payload.get("usageMetadata", {})
            if usage:
                state.gemini_tokens_today += usage.get("totalTokenCount", 0)
            return payload["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e3:
            report_error("Gemini fallback", e3)

    state.ai_unavailable_until = time.time() + AI_COOLDOWN_SECONDS
    print(f"[pipeline] All AI providers unavailable — pausing analysis for {AI_COOLDOWN_SECONDS}s")
    return None


MARKET_RELEVANCE_KEYWORDS = [
    "stock", "share", "market", "index", "nasdaq", "s&p", "dow jones", "sensex", "nifty",
    "bank nifty", "banknifty", "f&o", "futures and options", "nse", "bse",
    "fed", "rate", "inflation", "gdp", "earnings", "ipo", "merger", "acquisition",
    "oil", "gold", "silver", "copper", "commodity", "crude", "opec", "gas",
    "crypto", "bitcoin", "currency", "dollar", "yen", "yuan", "euro",
    "tariff", "trade war", "sanctions", "central bank", "treasury", "bond", "yield",
    "recession", "selloff", "sell-off", "rally", "surge", "plunge", "slump", "crash",
    "ceo", "bankruptcy", "layoff", "tech stock",
    "chip", "semiconductor", "ai ", "gpu", "cpu", "foundry", "chipmaker", "fab",
    "memory chip", "dram", "nand", "hbm", "storage", "data center", "datacenter",
    "cloud", "server", "nvidia", "tsmc", "asml",
    "war", "conflict", "military", "missile", "strike", "invasion", "attack",
    "hormuz", "strait", "geopolitical", "diplomacy", "ceasefire", "troops", "sanction",
    "renewable", "solar", "wind turbine", "nuclear", "uranium", "coal", "lng",
    "pipeline", "refinery", "power grid", "electricity", "battery", "lithium",
    "cobalt", "rare earth", "ev ", "electric vehicle", "automaker",
    "mining", "iron ore", "steel", "zinc", "nickel", "aluminum",
    "soybean", "cocoa", "livestock", "crop", "harvest", "farm subsidy",
    "drug", "fda", "vaccine", "clinical trial", "biotech", "pharma",
    "housing", "mortgage", "real estate", "construction", "homebuilder",
    "5g", "spectrum", "telecom", "broadband",
    "retail sales", "consumer spending", "e-commerce"
]


def passes_keyword_prefilter(headline):
    lower = headline.lower()
    return any(keyword in lower for keyword in MARKET_RELEVANCE_KEYWORDS)


def run_groq_filter_batch(headlines):
    headlines = [h for h in headlines if passes_keyword_prefilter(h)]
    if not headlines:
        return []
    numbered = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines))
    prompt = f"""Below are {len(headlines)} news headlines. For each one, decide if it would
likely move a stock price, sector, index, or commodity today (a market selloff/rally, a major
company shock, a central bank decision, a geopolitical event affecting markets, a big earnings
surprise). Ignore purely personal-finance advice, opinion/listicle content, or routine analyst notes.

{numbered}

Respond with ONLY the numbers of the headlines that qualify, comma-separated, nothing else.
If none qualify, respond with exactly: NONE"""
    raw_text = call_llm(prompt)
    if raw_text is None or raw_text.strip().upper() == "NONE":
        return []
    try:
        numbers = [int(n) for n in re.findall(r"\d+", raw_text)]
        return [headlines[n - 1] for n in numbers if 1 <= n <= len(headlines)]
    except Exception as e:
        report_error("Batch filter (parse)", e)
        return []


def analyze_ripple_effect(headline):
    prompt = f"""
    Headline: "{headline}"
    Identify the single market sector and company/ticker most directly and immediately impacted (the "primary" pick).
    Prefer a specific, real company or stock (e.g. Oracle, ORCL, Reliance Industries) over a broad index or
    commodity ETF (e.g. avoid SPY, QQQ, GLD, USO) unless the headline is genuinely only about a broad
    index/commodity with no specific company angle at all.
    Then list up to 4 additional related plays that could ripple from this news across the supply chain,
    commodities, or global markets (e.g. a chip-demand headline might ripple to memory makers, storage,
    data-center operators, power/utilities, or relevant commodities) — again preferring specific companies
    over broad ETFs. For each, give a direction. Keep it compact — names/tickers only, no explanations.
    Respond strictly as JSON, nothing else:
    {{
        "sector": "Sector name",
        "primary_ticker": "Best guess company name or symbol",
        "ripple_effects": [
            {{"name": "Company or ticker", "direction": "BULLISH or BEARISH"}}
        ]
    }}
    """
    raw_text = call_llm(prompt)
    if raw_text is None:
        return "General Markets", None, []
    try:
        cleaned = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        return parsed.get("sector", "General Markets"), parsed.get("primary_ticker"), parsed.get("ripple_effects", [])
    except Exception as e:
        report_error("Ripple analysis (parse)", e)
        return "General Markets", None, []


def av_get(params):
    if not ALPHAVANTAGE_API_KEY:
        return None
    elapsed = time.time() - state.last_av_call_time
    if elapsed < AV_MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(AV_MIN_SECONDS_BETWEEN_CALLS - elapsed)
    resp = requests.get("https://www.alphavantage.co/query", params={**params, "apikey": ALPHAVANTAGE_API_KEY}, timeout=15)
    state.last_av_call_time = time.time()
    state.av_calls_today += 1
    payload = resp.json()
    if "Note" in payload or "Information" in payload:
        return None
    return payload


def resolve_ticker_yf(company_or_ticker):
    try:
        results = yf.Search(company_or_ticker, timeout=10).quotes
        for r in results:
            if r.get("quoteType") == "EQUITY":
                return r.get("symbol")
        return results[0]["symbol"] if results else None
    except Exception:
        return None


def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    deltas = closes.diff().dropna()
    gains = deltas.clip(lower=0)
    losses = -deltas.clip(upper=0)
    avg_gain = gains.rolling(period).mean().iloc[-1]
    avg_loss = losses.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def get_cached_market_data(key):
    entry = state.market_data_cache.get(key)
    if not entry:
        return None
    if entry["source"] == "av":
        return entry["data"]
    if time.time() - entry["cached_at"] < YF_CACHE_SECONDS:
        return entry["data"]
    return None


def set_cached_market_data(key, data, source):
    state.market_data_cache[key] = {"data": data, "source": source, "cached_at": time.time()}


def fetch_market_data_yf(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="2mo", timeout=10)
        if hist.empty or len(hist) < 2:
            return None
        closes = hist["Close"]
        latest_price = float(closes.iloc[-1])
        prev_price = float(closes.iloc[-2])
        change_percent = f"{((latest_price - prev_price) / prev_price) * 100:.2f}%" if prev_price else None
        return {
            "price": round(latest_price, 2),
            "change_percent": change_percent,
            "rsi": compute_rsi(closes),
            "as_of": str(hist.index[-1].date())
        }
    except Exception:
        return None


TICKER_BAR_SYMBOLS = [
    ("^GSPC", "S&P 500"),
    ("^IXIC", "NASDAQ"),
    ("^DJI", "DOW"),
    ("DX-Y.NYB", "DXY"),
    ("GC=F", "GOLD"),
    ("CL=F", "OIL (WTI)"),
    ("BTC-USD", "BTC"),
]


def fetch_ticker_bar_data(symbol):
    cache_key = f"TICKERBAR:{symbol}"
    cached = get_cached_market_data(cache_key)
    if cached:
        return cached
    data = fetch_market_data_yf(symbol)
    if data:
        set_cached_market_data(cache_key, data, "yf")
    return data


def resolve_ticker(company_or_ticker):
    if not company_or_ticker:
        return None
    cache = state.symbol_cache
    if company_or_ticker in cache:
        return cache[company_or_ticker]
    resolved = None
    try:
        payload = av_get({"function": "SYMBOL_SEARCH", "keywords": company_or_ticker})
        if payload:
            matches = payload.get("bestMatches", [])
            resolved = matches[0]["1. symbol"] if matches else None
    except Exception:
        resolved = None
    if not resolved:
        resolved = resolve_ticker_yf(company_or_ticker)
    cache[company_or_ticker] = resolved
    return resolved


def fetch_market_data(ticker):
    cached = get_cached_market_data(ticker)
    if cached:
        return cached
    try:
        quote_payload = av_get({"function": "GLOBAL_QUOTE", "symbol": ticker})
        quote = quote_payload.get("Global Quote", {}) if quote_payload else {}

        rsi_payload = av_get({"function": "RSI", "symbol": ticker, "interval": "daily", "time_period": 14, "series_type": "close"})
        rsi_series = rsi_payload.get("Technical Analysis: RSI", {}) if rsi_payload else {}
        latest_rsi = next(iter(rsi_series.values()), {}).get("RSI") if rsi_series else None

        data = {
            "price": quote.get("05. price"),
            "change_percent": quote.get("10. change percent"),
            "rsi": latest_rsi
        }
        if data["price"] or data["rsi"]:
            set_cached_market_data(ticker, data, "av")
            return data
    except Exception:
        pass

    data = fetch_market_data_yf(ticker)
    if data:
        set_cached_market_data(ticker, data, "yf")
    return data


COMMODITY_MAP = [
    (["wti", "crude oil", "oil price", "oil prices", "barrel"], "WTI", "daily", "WTI Crude Oil", None),
    (["brent"], "BRENT", "daily", "Brent Crude Oil", None),
    (["natural gas", " lng "], "NATURAL_GAS", "daily", "Natural Gas", None),
    (["copper"], "COPPER", "monthly", "Copper", None),
    (["aluminum", "aluminium"], "ALUMINUM", "monthly", "Aluminum", None),
    (["wheat"], "WHEAT", "monthly", "Wheat", None),
    (["corn"], "CORN", "monthly", "Corn", None),
    (["cotton"], "COTTON", "monthly", "Cotton", None),
    (["sugar"], "SUGAR", "monthly", "Sugar", None),
    (["coffee"], "COFFEE", "monthly", "Coffee", None),
    (["gold"], "GOLD_SILVER_SPOT", None, "Gold Spot", "gold"),
    (["silver"], "GOLD_SILVER_SPOT", None, "Silver Spot", "silver"),
]


def detect_commodity(headline):
    lower_headline = headline.lower()
    for keywords, function, interval, display_name, metal_key in COMMODITY_MAP:
        if any(keyword in lower_headline for keyword in keywords):
            return function, interval, display_name, metal_key
    return None


INDEX_MAP = [
    (["bank nifty", "banknifty", "nifty bank"], "^NSEBANK", "Bank Nifty"),
    (["sensex", "bse"], "^BSESN", "Sensex"),
    (["nifty", "nse", "f&o", "futures and options", "futures & options"], "^NSEI", "Nifty 50"),
]


def detect_index(headline):
    lower_headline = headline.lower()
    for keywords, yf_symbol, display_name in INDEX_MAP:
        if any(keyword in lower_headline for keyword in keywords):
            return yf_symbol, display_name
    return None


YF_COMMODITY_SYMBOLS = {
    "WTI": "CL=F",
    "BRENT": "BZ=F",
    "NATURAL_GAS": "NG=F",
    "COPPER": "HG=F",
    "WHEAT": "ZW=F",
    "CORN": "ZC=F",
    "COTTON": "CT=F",
    "SUGAR": "SB=F",
    "COFFEE": "KC=F",
}
YF_METAL_SYMBOLS = {"gold": "GC=F", "silver": "SI=F"}


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_commodity_data(function, interval, display_name, metal_key):
    cache_key = f"COMMODITY:{function}:{metal_key or ''}"
    cached = get_cached_market_data(cache_key)
    if cached:
        return cached

    def yf_fallback():
        yf_symbol = YF_METAL_SYMBOLS.get(metal_key) if metal_key else YF_COMMODITY_SYMBOLS.get(function)
        if not yf_symbol:
            return None
        data = fetch_market_data_yf(yf_symbol)
        if data:
            data["display_name"] = display_name
            set_cached_market_data(cache_key, data, "yf")
        return data

    try:
        params = {"function": function}
        if interval:
            params["interval"] = interval
        payload = av_get(params)
        if not payload:
            return yf_fallback()

        latest_value, previous_value, as_of = None, None, None

        if function == "GOLD_SILVER_SPOT":
            metal_block = payload.get(metal_key) or payload.get(metal_key.capitalize()) if metal_key else None
            if isinstance(metal_block, dict):
                latest_value = _safe_float(metal_block.get("price") or metal_block.get("value"))
                as_of = metal_block.get("date") or metal_block.get("timestamp")
            elif "data" in payload and payload["data"]:
                series = payload["data"]
                latest_value = _safe_float(series[0].get("value"))
                previous_value = _safe_float(series[1].get("value")) if len(series) > 1 else None
                as_of = series[0].get("date")
        else:
            series = payload.get("data", [])
            if series:
                latest_value = _safe_float(series[0].get("value"))
                previous_value = _safe_float(series[1].get("value")) if len(series) > 1 else None
                as_of = series[0].get("date")

        if latest_value is None:
            return yf_fallback()

        change_percent = None
        if previous_value:
            change_percent = f"{((latest_value - previous_value) / previous_value) * 100:.2f}%"

        data = {
            "price": latest_value,
            "change_percent": change_percent,
            "rsi": None,
            "as_of": as_of,
            "display_name": display_name
        }
        set_cached_market_data(cache_key, data, "av")
        return data
    except Exception:
        return yf_fallback()


def run_deep_analysis(headline, sector, resolved_ticker, ripple_effects):
    try:
        index_match = detect_index(headline)
        commodity_match = detect_commodity(headline)
        if index_match:
            yf_symbol, display_name = index_match
            market_data = fetch_market_data_yf(yf_symbol)
            resolved_ticker = display_name
        elif commodity_match:
            function, interval, display_name, metal_key = commodity_match
            market_data = fetch_commodity_data(function, interval, display_name, metal_key)
            resolved_ticker = display_name
        else:
            market_data = fetch_market_data(resolved_ticker) if resolved_ticker else None
    except Exception as e:
        report_error("Market data grounding", e)
        market_data = None

    if market_data:
        detail_parts = [f"price ${market_data['price']}"]
        if market_data.get("change_percent"):
            detail_parts.append(f"change {market_data['change_percent']}")
        if market_data.get("rsi"):
            detail_parts.append(f"RSI {market_data['rsi']}")
        if market_data.get("as_of"):
            detail_parts.append(f"as of {market_data['as_of']}")
        grounding = (
            f"Real market data for {resolved_ticker}: {', '.join(detail_parts)}. "
            "Base your strategy on this real data, not assumptions."
        )
    else:
        grounding = "No real-time market data was available for this ticker right now — reason from the headline alone and say so in the strategy."

    ripple_note = ""
    if ripple_effects:
        ripple_summary = ", ".join(f"{r.get('name')} ({r.get('direction')})" for r in ripple_effects if r.get("name"))
        ripple_note = f"Related unverified ripple plays identified separately: {ripple_summary}. You may reference these in your strategy but do not treat them as fact-checked."

    prompt = f"""
    Analyze this high-impact market headline: "{headline}"
    Likely affected sector: {sector}. Verified ticker for grounding: {resolved_ticker or "none found"}.
    {grounding}
    {ripple_note}
    Provide an institutional-grade trading setup across US Stocks, Indian Markets, Global Stocks, Crypto, and Commodities.
    Keep the strategy field to 2-3 concise sentences (under 60 words) — no filler, no repetition.
    Also write a one-sentence plain-language summary of what happened and why it matters (under 30 words,
    distinct from both the headline and the key takeaways — this is the expanded explanation shown under the headline).
    Also classify this headline into exactly one category: "Signal" (a clear tradeable setup),
    "Market News" (routine price/market movement), "Analysis" (broader trend commentary),
    "Alert" (urgent/breaking risk event), "Trade Idea" (a specific actionable position), or
    "Macro View" (economic/policy-level story). And give up to 3 short key takeaways (under 12 words each).
    Return your response strictly as a valid JSON object with this exact structure, and nothing else (no markdown, no code fences):
    {{
        "sentiment": "STRONG_BULLISH / WEAK_BULLISH / BEARISH",
        "sector": "Sector name",
        "summary": "One-sentence plain-language explanation of the news and its market relevance.",
        "buy_targets": ["Ticker1", "Ticker2"],
        "sell_targets": ["Ticker1", "Ticker2"],
        "strategy": "Concise action to take before market open, referencing the real data if provided.",
        "category": "Signal / Market News / Analysis / Alert / Trade Idea / Macro View",
        "key_takeaways": ["Short takeaway 1", "Short takeaway 2", "Short takeaway 3"]
    }}
    """
    raw_text = call_llm(prompt)
    if raw_text is None:
        return {"sentiment": "ERROR", "sector": "None", "buy_targets": [], "sell_targets": [],
                "strategy": "Both Groq and the OpenRouter fallback failed — see the Last Error banner for details.",
                "market_data": market_data, "grounded_ticker": resolved_ticker}
    try:
        cleaned = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(cleaned)
        result["market_data"] = market_data
        result["grounded_ticker"] = resolved_ticker
        return result
    except Exception as e:
        report_error("Deep analysis (parse)", e)
        return {"sentiment": "ERROR", "sector": "None", "buy_targets": [], "sell_targets": [], "strategy": str(e), "market_data": market_data, "grounded_ticker": resolved_ticker}


RSS_SOURCE_NAMES = {
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US": "Yahoo Finance",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html": "CNBC",
    "https://www.investing.com/rss/news_25.rss": "Investing.com",
    "https://feeds.bloomberg.com/markets/news.rss": "Bloomberg",
    "https://feeds.bloomberg.com/economics/news.rss": "Bloomberg Economics",
}


def fetch_live_financial_news():
    rss_urls = list(RSS_SOURCE_NAMES.keys())

    new_headlines_found = []
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                if entry.title not in state.processed_headlines:
                    new_headlines_found.append(entry.title)
                    state.processed_headlines.add(entry.title)
                    state.headline_metadata[entry.title] = {
                        "link": getattr(entry, "link", None),
                        "source": RSS_SOURCE_NAMES.get(url, "Unknown source")
                    }
        except Exception:
            continue
    if new_headlines_found:
        save_processed_headlines(state.processed_headlines)
    return new_headlines_found


def sentiment_confidence(sentiment):
    s = (sentiment or "").upper()
    if "STRONG" in s:
        return 91
    if "WEAK" in s:
        return 58
    if "BULLISH" in s or "BEARISH" in s:
        return 76
    return 0


def sentiment_style(sentiment):
    s = (sentiment or "").upper()
    if "BULLISH" in s:
        return {"bg": "#eafaf1", "border": "#1ea672", "text": "#1ea672"}
    if "BEARISH" in s:
        return {"bg": "#fbebec", "border": "#d64545", "text": "#d64545"}
    return {"bg": "#fdf3e2", "border": "#c98a1f", "text": "#c98a1f"}


CATEGORY_STYLE = {
    "Signal": {"icon": "✨", "color": "#7c5cff"},
    "Market News": {"icon": "🌐", "color": "#3b82f6"},
    "Analysis": {"icon": "📄", "color": "#1ea672"},
    "Alert": {"icon": "🚨", "color": "#d64545"},
    "Trade Idea": {"icon": "💼", "color": "#b8791a"},
    "Macro View": {"icon": "⚡", "color": "#c98a1f"},
}


def category_style(category):
    return CATEGORY_STYLE.get(category, {"icon": "📊", "color": "#6b7280"})


def run_pipeline_cycle():
    """One tick of the pipeline: scan feeds if the interval elapsed and the queue is
    empty, then batch-filter or deep-analyze whatever is in the queue. This is the same
    logic that used to live inline in app.py's pipeline_fragment(), just without any
    Streamlit UI calls — the scheduler calls this repeatedly in the background.
    Reads refresh_interval_seconds/active from state so PATCH /api/settings can change
    behavior at runtime without restarting the scheduler."""
    state.roll_daily_usage_if_needed()
    if not state.active:
        return None
    now = time.time()

    if (not state.pending_headlines and not state.qualified_headlines
            and now - state.last_scan_time >= state.refresh_interval_seconds):
        new_headlines = fetch_live_financial_news()
        state.last_scan_time = now
        if new_headlines:
            state.pending_headlines.extend(new_headlines)

    qualifying_headline = None
    if state.qualified_headlines:
        qualifying_headline = state.qualified_headlines.pop(0)
    elif state.pending_headlines and time.time() >= state.ai_unavailable_until:
        batch = state.pending_headlines[:BATCH_FILTER_SIZE]
        state.pending_headlines = state.pending_headlines[len(batch):]
        qualifying = run_groq_filter_batch(batch)
        if qualifying:
            qualifying_headline = qualifying[0]
            state.qualified_headlines.extend(qualifying[1:])

    if not qualifying_headline:
        return None

    headline = qualifying_headline
    sector, company_guess, ripple_effects = analyze_ripple_effect(headline)
    resolved_ticker = None if (detect_commodity(headline) or detect_index(headline)) else resolve_ticker(company_guess)
    ai_blueprint = run_deep_analysis(headline, sector, resolved_ticker, ripple_effects)

    market_data = ai_blueprint.get("market_data")
    grounded_reading = "No live data available"
    if market_data:
        detail_bits = [f"${market_data.get('price')}"]
        if market_data.get("change_percent"):
            detail_bits.append(f"({market_data['change_percent']})")
        if market_data.get("rsi"):
            detail_bits.append(f"| RSI {market_data['rsi']}")
        if market_data.get("as_of"):
            detail_bits.append(f"| as of {market_data['as_of']}")
        grounded_reading = f"{ai_blueprint.get('grounded_ticker')} {' '.join(detail_bits)}"

    ripple_display = "; ".join(f"{r.get('name')} ({r.get('direction')})" for r in ripple_effects if r.get("name")) or "None identified"
    source_meta = state.headline_metadata.get(headline, {})

    new_alert = {
        "Timestamp": datetime.now().strftime("%I:%M:%S %p"),
        "Headline": headline,
        "Summary": ai_blueprint.get("summary"),
        "Sentiment": ai_blueprint.get("sentiment"),
        "Sector": ai_blueprint.get("sector"),
        "Buy Tickers": ", ".join(ai_blueprint.get("buy_targets", [])),
        "Sell Tickers": ", ".join(ai_blueprint.get("sell_targets", [])),
        "Execution Blueprint": ai_blueprint.get("strategy"),
        "Grounded Data": grounded_reading,
        "Ripple Effects (AI-inferred, unverified)": ripple_display,
        "Category": ai_blueprint.get("category", "Signal"),
        "Key Takeaways": ai_blueprint.get("key_takeaways", []),
        "Source": source_meta.get("source"),
        "Article Link": source_meta.get("link")
    }
    state.trade_history.insert(0, new_alert)
    save_trade_history(state.trade_history)
    return new_alert


# --- MARKET REGIME ---
def compute_market_regime(lookback=20):
    """A deterministic Risk-On/Risk-Off/Neutral read derived from the bullish-vs-bearish
    split of our own most recent AI signals — not a separate AI call, since we already have
    real, grounded sentiment data sitting in trade_history. Avoids inventing a new opaque
    "vibes" metric on top of data that's already there."""
    recent = state.trade_history[:lookback]
    if not recent:
        return {"regime": "Neutral", "bullish": 0, "bearish": 0, "score": 0}
    bullish = sum(1 for a in recent if "BULLISH" in (a.get("Sentiment") or "").upper())
    bearish = sum(1 for a in recent if "BEARISH" in (a.get("Sentiment") or "").upper())
    total = bullish + bearish
    score = round(((bullish - bearish) / total) * 100, 1) if total else 0
    if score > 20:
        regime = "Risk-On"
    elif score < -20:
        regime = "Risk-Off"
    else:
        regime = "Neutral"
    return {"regime": regime, "bullish": bullish, "bearish": bearish, "score": score}


# --- PRICE HISTORY (for sparkline charts) ---
def fetch_price_history(ticker, points=20):
    """Last `points` daily closes for a symbol — used for the small sparkline charts on
    the Market Data screen. Cached alongside the regular market data cache."""
    cache_key = f"HISTORY:{ticker}:{points}"
    cached = get_cached_market_data(cache_key)
    if cached:
        return cached
    try:
        hist = yf.Ticker(ticker).history(period="3mo", timeout=10)
        if hist.empty:
            return None
        closes = [round(float(c), 2) for c in hist["Close"].tolist()[-points:]]
        set_cached_market_data(cache_key, closes, "yf")
        return closes
    except Exception:
        return None


# --- CATEGORIZED MARKET DATA (Indices / Commodities / Forex / Bonds) ---
MARKET_DATA_CATEGORIES = {
    "indices": [
        ("^GSPC", "S&P 500"),
        ("^IXIC", "NASDAQ"),
        ("^DJI", "DOW"),
        ("^NSEI", "NIFTY 50"),
        ("^BSESN", "SENSEX"),
        ("^NSEBANK", "BANK NIFTY"),
    ],
    "commodities": [
        ("GC=F", "GOLD"),
        ("SI=F", "SILVER"),
        ("CL=F", "OIL (WTI)"),
        ("BZ=F", "BRENT"),
        ("HG=F", "COPPER"),
        ("NG=F", "NATURAL GAS"),
    ],
    "forex": [
        ("DX-Y.NYB", "DXY"),
        ("EURUSD=X", "EUR/USD"),
        ("GBPUSD=X", "GBP/USD"),
        ("USDJPY=X", "USD/JPY"),
        ("USDINR=X", "USD/INR"),
    ],
    "bonds": [
        ("^IRX", "US 3M YIELD"),
        ("^FVX", "US 5Y YIELD"),
        ("^TNX", "US 10Y YIELD"),
        ("^TYX", "US 30Y YIELD"),
    ],
}


def fetch_market_data_category(category, with_history=True, history_points=20):
    """Returns [{symbol, label, price, change_percent, rsi, as_of, history:[...]}] for one
    of indices/commodities/forex/bonds — the data source for the Market Data screen's tabs
    and sparklines."""
    symbols = MARKET_DATA_CATEGORIES.get(category)
    if symbols is None:
        return None
    rows = []
    for symbol, label in symbols:
        data = fetch_ticker_bar_data(symbol) or {}
        row = {"symbol": symbol, "label": label, **data}
        if with_history:
            row["history"] = fetch_price_history(symbol, history_points)
        rows.append(row)
    return rows


# --- SECTOR PERFORMANCE (real ETF price data, not just signal counts) ---
SECTOR_ETF_MAP = {
    "Energy": "XLE",
    "Financials": "XLF",
    "Technology": "XLK",
    "Semiconductors": "SMH",
    "Consumer Staples": "XLP",
    "Consumer Discretionary": "XLY",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

TIMEFRAME_LOOKBACK_TRADING_DAYS = {"1D": 1, "1W": 5, "1M": 21, "1Y": 252}


def _sector_signal_count(sector_label):
    """Loose case-insensitive substring match against the AI's free-text sector field —
    the AI doesn't pick from SECTOR_ETF_MAP's fixed list, so exact matching would undercount."""
    label_lower = sector_label.lower()
    count = 0
    for alert in state.trade_history:
        alert_sector = (alert.get("Sector") or "").lower()
        if label_lower in alert_sector or alert_sector in label_lower:
            count += 1
    return count


def fetch_sector_performance(timeframe="1D"):
    """Real % price change per sector ETF over the requested timeframe, plus how many of
    our own AI signals have touched that sector. One 1-year history fetch per ETF (cached)
    covers every timeframe by just changing how far back we look for the comparison close."""
    lookback = TIMEFRAME_LOOKBACK_TRADING_DAYS.get(timeframe, 1)
    rows = []
    for sector, etf_symbol in SECTOR_ETF_MAP.items():
        cache_key = f"SECTORHIST:{etf_symbol}"
        closes = get_cached_market_data(cache_key)
        if not closes:
            try:
                hist = yf.Ticker(etf_symbol).history(period="1y", timeout=10)
                closes = hist["Close"].tolist() if not hist.empty else None
                if closes:
                    set_cached_market_data(cache_key, closes, "yf")
            except Exception:
                closes = None
        if not closes:
            rows.append({"sector": sector, "symbol": etf_symbol, "change_percent": None, "signal_count": _sector_signal_count(sector)})
            continue
        latest = closes[-1]
        base_idx = max(0, len(closes) - 1 - lookback)
        base = closes[base_idx]
        change_percent = round(((latest - base) / base) * 100, 2) if base else None
        rows.append({
            "sector": sector,
            "symbol": etf_symbol,
            "change_percent": change_percent,
            "signal_count": _sector_signal_count(sector)
        })
    return rows


# --- WATCHLIST ---
def add_to_watchlist(symbol, label=None):
    symbol = symbol.strip().upper()
    if any(w["symbol"] == symbol for w in state.watchlist):
        return state.watchlist
    state.watchlist.append({"symbol": symbol, "label": label or symbol})
    save_watchlist(state.watchlist)
    return state.watchlist


def remove_from_watchlist(symbol):
    symbol = symbol.strip().upper()
    state.watchlist = [w for w in state.watchlist if w["symbol"] != symbol]
    save_watchlist(state.watchlist)
    return state.watchlist


def fetch_watchlist_quotes():
    """Live price/change for every symbol on the watchlist — reuses the same yfinance
    fetch + cache path as everything else, just for user-picked symbols instead of a
    fixed list."""
    rows = []
    for entry in state.watchlist:
        data = fetch_ticker_bar_data(entry["symbol"]) or {}
        rows.append({"symbol": entry["symbol"], "label": entry["label"], **data})
    return rows


# --- CACHE WARMING ---
def refresh_watchlist_cache():
    """Synchronous, on-demand refresh of just the watchlist cache — called right after a
    POST/DELETE mutation so the change is visible immediately instead of waiting for the
    next scheduled warm cycle (up to MARKET_DATA_WARM_SECONDS later)."""
    state.cached_watchlist = fetch_watchlist_quotes()
    return state.cached_watchlist


def prefetch_all_market_data():
    """Proactively refreshes every market-data cache (ticker bar, all 4 categories, all
    4 sector timeframes, watchlist) in parallel threads and writes the FINAL, request-ready
    results into state.cached_* — GET endpoints read only from these fields and never fall
    back to a live fetch themselves. That distinction matters: the underlying fetch_*
    helpers each have their own short TTL cache (get_cached_market_data), so if request
    handlers called them directly, a slow/throttled network call on a free-tier shared vCPU
    could still occasionally block a request for many seconds once that TTL expires mid-cycle
    (observed live: a 46s spike). Serving strictly from state.cached_* means a request always
    gets an instant answer — worst case, an answer that's one warm cycle old, never a hang.
    yfinance/requests calls are blocking network I/O, so run them concurrently instead of
    one-by-one — otherwise warming ~30 symbols sequentially could take 20-30s instead of a
    couple of seconds."""
    # Kept modest (not e.g. 10+) because free-tier hosts (Render's free plan) give a single
    # shared vCPU — too many concurrent threads contend for the GIL hard enough to visibly
    # delay request-handling on the main thread, even though each individual call is I/O-bound.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        ticker_futures = {executor.submit(fetch_ticker_bar_data, symbol): (symbol, label) for symbol, label in TICKER_BAR_SYMBOLS}
        category_futures = {executor.submit(fetch_market_data_category, category): category for category in MARKET_DATA_CATEGORIES}
        sector_futures = {executor.submit(fetch_sector_performance, tf): tf for tf in TIMEFRAME_LOOKBACK_TRADING_DAYS}
        watchlist_future = executor.submit(fetch_watchlist_quotes)

        new_ticker_bar = []
        for future in concurrent.futures.as_completed(ticker_futures):
            symbol, label = ticker_futures[future]
            try:
                data = future.result()
            except Exception as e:
                report_error("Market data prefetch (ticker bar)", e)
                data = None
            new_ticker_bar.append({"symbol": symbol, "label": label, "data": data})
        symbol_order = {symbol: i for i, (symbol, _) in enumerate(TICKER_BAR_SYMBOLS)}
        new_ticker_bar.sort(key=lambda row: symbol_order[row["symbol"]])
        state.cached_ticker_bar = new_ticker_bar

        for future in concurrent.futures.as_completed(category_futures):
            category = category_futures[future]
            try:
                state.cached_market_data[category] = future.result()
            except Exception as e:
                report_error("Market data prefetch (category)", e)

        for future in concurrent.futures.as_completed(sector_futures):
            timeframe = sector_futures[future]
            try:
                state.cached_sectors[timeframe] = future.result()
            except Exception as e:
                report_error("Market data prefetch (sectors)", e)

        try:
            state.cached_watchlist = watchlist_future.result()
        except Exception as e:
            report_error("Market data prefetch (watchlist)", e)

    state.cache_last_updated = time.time()
