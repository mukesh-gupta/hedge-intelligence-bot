import collections
import concurrent.futures
import gc
import gzip
import hashlib
import html
import itertools
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import certifi
import feedparser
import requests
import yfinance as yf
from dotenv import load_dotenv
from groq import Groq

from backend import storage

load_dotenv()

# No SDK retries: a 429 should fall straight through to the next model in the chain
# rather than sleep and retry a model whose daily allowance is already spent.
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=15.0, max_retries=0)
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash-lite"
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Everything runs on free tiers. Groq's free tier gives each model its own daily allowance
# (200K tokens/day each — checked live: 120b returned 429 while 20b kept answering), so each
# pipeline stage gets its own model instead of every stage draining one shared pool.
GROQ_FAST_MODEL = "openai/gpt-oss-20b"
GROQ_DEEP_MODEL = "openai/gpt-oss-120b"
GROQ_BACKUP_MODEL = "qwen/qwen3.8-27b"  # no hidden reasoning, so very few tokens per call
GROQ_DAILY_TOKEN_LIMIT = 200_000
# qwen's free tier also caps output at 1,000 tokens/minute and rejects any request whose
# expected output exceeds that (seen live on deep analyses), so its output is capped. It
# writes no hidden reasoning, so a cap is safe for it, unlike the gpt-oss models.
QWEN_PARAMS = {"max_tokens": 900}
# Each tier's chain avoids the other tier's primary model, so running out in one stage
# can't drain the other's daily allowance (seen live: deep analysis fell back onto 20b and
# used up triage's tokens). Gemini and OpenRouter are separate free quotas.
AI_TIERS = {
    # Triage: scores up to 20 headlines per call. ~1.1K tokens per batch on 20b at low effort.
    "fast": [(GROQ_FAST_MODEL, {"reasoning_effort": "low"}), (GROQ_BACKUP_MODEL, QWEN_PARAMS), "gemini", "openrouter"],
    # Deep analysis, only for high-impact headlines. Gemini before qwen: a deep answer is
    # close to qwen's per-minute output cap.
    "deep": [(GROQ_DEEP_MODEL, {}), "gemini", (GROQ_BACKUP_MODEL, QWEN_PARAMS), "openrouter"],
}
# Triage waits until it has this many headlines, or until the oldest has waited this long.
# Each call carries ~1K tokens of fixed cost (instructions plus hidden reasoning), and
# triaging the 1-3 headlines each 20s scan brings spent 20b's daily allowance in hours.
TRIAGE_MIN_BATCH = 10
TRIAGE_MAX_WAIT_SECONDS = 120
# Triage impact scores (1-10): below QUICK the headline is dropped; from QUICK up it becomes
# a quick signal straight from the triage answer (no extra AI call); from DEEP up it gets a
# full deep analysis. DEEP rises to DEEP_IMPACT_WHEN_LOW_BUDGET once the deep model has used
# most of its daily allowance, so what's left goes to the biggest news.
QUICK_SIGNAL_MIN_IMPACT = 5
DEEP_ANALYSIS_MIN_IMPACT = 7
DEEP_IMPACT_WHEN_LOW_BUDGET = 9
LOW_BUDGET_FRACTION = 0.8

AI_COOLDOWN_SECONDS = 90
AV_MIN_SECONDS_BETWEEN_CALLS = 13
YF_CACHE_SECONDS = 60
BATCH_FILTER_SIZE = 20

# ~1.1KB/entry observed in practice, so 2000 entries is ~2.2MB. Stored as a Redis list so a
# new signal is one small LPUSH, not a re-upload of the whole history — at a few hundred
# signals a day, full re-uploads alone would exceed Upstash's free 10GB/month bandwidth.
SIGNALS_KEY = "signals"
MAX_STORED_ALERTS = 2000
# ~70 bytes/entry observed, so 5000 entries is ~350KB.
MAX_STORED_HEADLINES = 5000
# Metadata is only needed until a headline is analyzed; ones the filter rejects would
# otherwise linger forever.
MAX_HEADLINE_METADATA = 1000
# News older than this is already priced in — skipped at ingestion, and dropped from the
# queues if it ages past this while waiting for analysis.
MAX_HEADLINE_AGE_MINUTES = 60
# Newest-first cap on the unfiltered queue, so a long AI outage can't grow it without bound.
MAX_PENDING_HEADLINES = 300
FEED_TIMEOUT_SECONDS = 8
FEED_FETCH_WORKERS = 10
# Seen-headline markers change on nearly every scan; re-uploading the whole set each time
# would cost far more Upstash bandwidth than it's worth. Losing a few minutes of markers on
# a restart is harmless — MAX_HEADLINE_AGE_MINUTES already stops old news being re-analyzed.
HEADLINES_SAVE_INTERVAL_SECONDS = 300
# Same story syndicated by several outlets with reworded titles: treated as a duplicate when
# this share of meaningful words overlaps with a story already accepted in the window.
DEDUP_SIMILARITY = 0.5
DEDUP_WINDOW_SECONDS = 6 * 3600
# Signals and seen-headlines older than this are deleted from memory and Upstash. Checked
# once a day (and at startup), so data is gone within a day of turning 7 days old.
RETENTION_DAYS = 7
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
        self.model_tokens_today = {}
        self.av_calls_today = 0
        self.av_exhausted = False
        self.market_data_cache = {}
        self.symbol_cache = {}
        self.last_av_call_time = 0

        self.last_error = None
        self.recent_errors = collections.deque(maxlen=50)  # newest last; served by /api/errors
        self.ai_unavailable_until = 0
        self.model_unavailable_until = {}  # model/provider -> epoch when its rate limit lifts
        self.last_scan_time = 0
        self.headlines_saved_at = 0

        # Runtime-adjustable settings — mutable via PATCH /api/settings instead of being
        # fixed constants, so the React Settings screen can actually control the scheduler.
        self.refresh_interval_seconds = 20
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
            self.model_tokens_today = {}
            self.av_calls_today = 0
            self.av_exhausted = False
            self.market_data_cache = {}
            self.symbol_cache = {}
            prune_old_data()


def _alert_time(alert):
    """Parses an alert's UTC ISO Timestamp. Alerts from before the switch to ISO timestamps
    stored only a time-of-day with no date — they return None and are treated as expired."""
    try:
        return datetime.fromisoformat(alert.get("Timestamp", "").removesuffix("Z"))
    except (TypeError, ValueError):
        return None


def prune_old_data():
    """Deletes signals and seen-headline markers older than RETENTION_DAYS, in memory and
    in Upstash, so neither grows indefinitely."""
    cutoff = datetime.utcnow() - timedelta(days=RETENTION_DAYS)
    kept_alerts = [a for a in state.trade_history if (_alert_time(a) or datetime.min) >= cutoff]
    if len(kept_alerts) != len(state.trade_history):
        removed = len(state.trade_history) - len(kept_alerts)
        state.trade_history = kept_alerts
        replace_trade_history(kept_alerts)
        print(f"[pipeline] Pruned {removed} signals older than {RETENTION_DAYS} days")

    cutoff_epoch = time.time() - RETENTION_DAYS * 86400
    kept_headlines = {t: seen for t, seen in state.processed_headlines.items() if seen >= cutoff_epoch}
    if len(kept_headlines) != len(state.processed_headlines):
        removed = len(state.processed_headlines) - len(kept_headlines)
        state.processed_headlines = kept_headlines
        save_processed_headlines(kept_headlines)
        print(f"[pipeline] Pruned {removed} seen-headlines older than {RETENTION_DAYS} days")


def load_trade_history():
    """Signals, newest first, from the Redis list. On the first run after the switch to a
    list, copies them over from the old single-JSON key (which is left in place, untouched)."""
    replies = storage.redis_pipeline([["LRANGE", SIGNALS_KEY, 0, -1]])
    if replies is None:
        return []  # Upstash not configured or unreachable: behave like a cold start
    if replies[0]:
        return [json.loads(item) for item in replies[0]]
    legacy = storage.redis_get_json("trade_history", [])[:MAX_STORED_ALERTS]
    if legacy and storage.redis_pipeline([["RPUSH", SIGNALS_KEY, *[json.dumps(a) for a in legacy]]]) is None:
        print("[pipeline] Copying signals to the Redis list failed; will retry next start")
    return legacy


def append_signal(alert):
    if storage.redis_pipeline([
        ["LPUSH", SIGNALS_KEY, json.dumps(alert)],
        ["LTRIM", SIGNALS_KEY, 0, MAX_STORED_ALERTS - 1],
    ]) is None:
        report_error("Persisting signal", "Upstash write failed")


def replace_trade_history(history):
    """Full rewrite — only for the once-a-day retention prune, never per signal."""
    commands = [["DEL", SIGNALS_KEY]]
    if history:
        commands.append(["RPUSH", SIGNALS_KEY, *[json.dumps(a) for a in history[:MAX_STORED_ALERTS]]])
    if storage.redis_pipeline(commands) is None:
        report_error("Persisting trade history", "Upstash write failed")


def load_processed_headlines():
    """title -> epoch seconds first seen. Insertion-ordered dict so trimming keeps the NEWEST
    headlines, and timestamped so prune_old_data() can expire them. Older deployments stored
    a plain list of titles; those get stamped "now" and age out normally from there."""
    stored = storage.redis_get_json("processed_headlines", {})
    if isinstance(stored, list):
        now = time.time()
        return {title: now for title in stored}
    return stored


def save_processed_headlines(headlines):
    """Persists only markers from the last two freshness windows: anything older is skipped
    by the MAX_HEADLINE_AGE_MINUTES check anyway, and re-uploading days of markers every few
    minutes would use up a large share of Upstash's free 10GB/month bandwidth."""
    cutoff = time.time() - 2 * MAX_HEADLINE_AGE_MINUTES * 60
    recent = [(title, seen) for title, seen in headlines.items() if seen >= cutoff]
    trimmed = dict(recent[-MAX_STORED_HEADLINES:])
    if not storage.redis_set_json("processed_headlines", trimmed):
        report_error("Persisting seen-headlines", "Upstash write failed")


def load_watchlist():
    return storage.redis_get_json("watchlist", list(DEFAULT_WATCHLIST))


def save_watchlist(watchlist):
    if not storage.redis_set_json("watchlist", watchlist):
        report_error("Persisting watchlist", "Upstash write failed")


def report_error(source, message):
    """Record the failure on shared state instead of firing a Streamlit toast —
    the API layer surfaces this via GET /api/status."""
    state.last_error = {"source": source, "message": str(message)[:300], "time": datetime.now().strftime("%I:%M:%S %p")}
    state.recent_errors.append({**state.last_error, "at": datetime.utcnow().isoformat() + "Z"})
    print(f"[pipeline] {source} error: {str(message)[:200]}")


def process_memory_mb():
    """Resident memory of this process, read from /proc (Linux, i.e. Render) — the number
    that matters against the free plan's 512MB limit. None where /proc isn't available."""
    try:
        with open("/proc/self/statm") as f:
            return round(int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024 / 1024, 1)
    except (OSError, ValueError, AttributeError):
        return None


state = PipelineState()
prune_old_data()


def _retry_after_seconds(error, default=60):
    """Seconds until a rate limit lifts, from Groq's 429 text ("Please try again in 7m12.5s")."""
    match = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", str(error))
    if not match or not any(match.groups()):
        return default
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds or 0)


def _provider_available(name):
    return time.time() >= state.model_unavailable_until.get(name, 0)


def _mark_unavailable(name, error, status_code):
    """Rate-limited (429): skip this model/provider until its limit lifts. Any other failure
    (timeout, 5xx): skip it briefly so one flaky provider doesn't slow every call."""
    cooldown = _retry_after_seconds(error, default=600) if status_code == 429 else 30
    state.model_unavailable_until[name] = time.time() + cooldown


def deep_impact_threshold():
    """Impact score a headline needs for a full deep analysis. Raised once the deep model
    has used most of its free daily allowance (or is rate-limited), so the remaining quota
    goes to the biggest news instead of running out on routine stories."""
    used = state.model_tokens_today.get(GROQ_DEEP_MODEL, 0)
    if used >= LOW_BUDGET_FRACTION * GROQ_DAILY_TOKEN_LIMIT or not _provider_available(GROQ_DEEP_MODEL):
        return DEEP_IMPACT_WHEN_LOW_BUDGET
    return DEEP_ANALYSIS_MIN_IMPACT


def _call_groq(model, extra_params, prompt, temperature):
    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            **extra_params,
        )
    except Exception as e:
        report_error(f"Groq {model}", e)
        _mark_unavailable(model, e, getattr(e, "status_code", None))
        return None
    tokens = completion.usage.total_tokens if completion.usage else 0
    state.groq_tokens_today += tokens
    state.model_tokens_today[model] = state.model_tokens_today.get(model, 0) + tokens
    return (completion.choices[0].message.content or "").strip() or None


def _call_gemini(prompt, temperature):
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            # Key in a header, not ?key=: requests puts the URL in its error messages, and
            # errors are served by the public /api/errors and /api/usage endpoints.
            headers={"x-goog-api-key": GEMINI_API_KEY},
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
        return payload["candidates"][0]["content"]["parts"][0]["text"].strip() or None
    except Exception as e:
        report_error("Gemini", e)
        _mark_unavailable("gemini", e, getattr(getattr(e, "response", None), "status_code", None))
        return None


def _call_openrouter(prompt, temperature):
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
        return payload["choices"][0]["message"]["content"].strip() or None
    except Exception as e:
        report_error("OpenRouter", e)
        _mark_unavailable("openrouter", e, getattr(getattr(e, "response", None), "status_code", None))
        return None


def call_llm(prompt, tier="deep", temperature=0.0):
    """Tries the tier's providers in order (see AI_TIERS). A model that hits its rate limit
    is skipped until the limit lifts instead of being retried on every call. Returns None
    only if all are unavailable. No max_tokens cap on gpt-oss models — they spend a large,
    variable amount of their budget on hidden reasoning before the visible answer; capping
    output length caused them to hit the limit mid-thought and return empty responses."""
    if time.time() < state.ai_unavailable_until:
        return None

    for provider in AI_TIERS[tier]:
        name = provider[0] if isinstance(provider, tuple) else provider
        if not _provider_available(name):
            continue
        if name == "gemini":
            content = _call_gemini(prompt, temperature) if GEMINI_API_KEY else None
        elif name == "openrouter":
            content = _call_openrouter(prompt, temperature) if OPENROUTER_API_KEY else None
        else:
            content = _call_groq(name, provider[1], prompt, temperature)
        if content:
            return content

    state.ai_unavailable_until = time.time() + AI_COOLDOWN_SECONDS
    print(f"[pipeline] All AI providers unavailable — pausing analysis for {AI_COOLDOWN_SECONDS}s")
    return None


def _word_regex(fragments):
    """Case-insensitive whole-word matcher for a list of regex fragments, each allowed a
    trailing plural "s"/"es". Whole-word matters: plain substring checks let "rate" match
    "corporate", "war" match "software", "nse" match "response" and "gold" match "Goldman"."""
    alternation = "|".join(f"(?:{f})" for f in fragments)
    return re.compile(rf"(?<![a-z0-9])(?:{alternation})(?:e?s)?(?![a-z0-9])", re.IGNORECASE)


# Regex fragments, matched as whole words (see _word_regex). Deliberately broad — this is
# only the cheap first gate; the AI batch filter makes the final market-moving call.
MARKET_RELEVANCE_KEYWORDS = [
    # Markets and price moves
    "stock", "share", r"equit(?:y|ies)", "market", "index", "indices", "nasdaq", r"s&p", "dow",
    "dow jones", "wall street", "ftse", "dax", "cac", "stoxx", "nikkei", "topix", "hang seng",
    "kospi", "shanghai composite", "csi 300", "asx", "msci", "futures", "etf", "ipo", "listing",
    r"delist(?:ed|ing)?", r"short[- ]sell(?:er|ing)?", "volatility", "vix", "bull market",
    "bear market", "correction", "record high", "all-time high", "52-week", "investor", "trader",
    "trading", "hedge fund", "fund", "valuation", "market cap", r"rall(?:y|ies|ied|ying)",
    r"surg(?:e|ed|ing)", r"plung(?:e|ed|ing)", r"slump(?:ed|ing)?", r"crash(?:ed|ing)?",
    r"tumbl(?:e|ed|ing)", r"soar(?:ed|ing)?", r"jump(?:ed|ing)?", r"sink(?:ing)?", "sank",
    r"slid(?:e|ing)?", r"sell-?off", "rout", r"rebound(?:ed|ing)?",
    # Macro, rates and central banks
    "fed", "federal reserve", "fomc", "powell", "central bank", "ecb", "lagarde", "boj",
    "bank of japan", "ueda", "boe", "bank of england", "pboc", "snb", "rba", "bank of canada",
    "rbi", "monetary policy", "mpc", "rate", "repo rate", "interest rate", "rate cut", "rate hike",
    "hike", "inflation", "deflation", "disinflation", "cpi", "ppi", "pce", "gdp", "payroll",
    "nonfarm", "jobs report", "jobs data", "jobless", "unemployment", "job openings", "jolts",
    "pmi", "ism", "retail sales", "consumer spending", "consumer confidence", "consumer sentiment",
    "industrial output", "factory", "manufacturing", "recession", "slowdown", "stimulus",
    "treasury", "treasuries", "bond", "yield", "gilt", "bund", "jgb", "debt", "deficit", "fiscal",
    "budget", r"tax(?:es)?", "tariff", "trade war", "trade deal", "trade talks", "export", "import",
    "sanction", "embargo", "shutdown", "debt ceiling", "imf", "world bank", "economy", "economic",
    "growth",
    # Company events
    r"earning", "revenue", "sales", "profit", "net loss", "quarterly", r"q[1-4]", r"fy\d{2,4}",
    "results", "guidance", "forecast", "outlook", "estimate", "expectation", "consensus",
    "merger", "acquisition", r"acquir(?:e|ed|ing)", "takeover", "buyout", "deal", "stake", "bid",
    "buyback", "share repurchase", "dividend", "stock split", r"spin-?off", "demerger",
    r"bankrupt(?:cy)?", "chapter 11", "default", "insolvency", "layoff", "job cuts",
    r"restructur(?:e|ed|ing)", "ceo", "cfo", r"downgrad(?:e|ed|ing)", r"upgrad(?:e|ed|ing)",
    "price target", "credit rating", "moody's", "fitch", r"halt(?:ed|ing)?", "recall", "antitrust",
    "lawsuit", "probe", "investigation", "fined", "penalty", "sec", "ftc", "doj", "regulator",
    # India
    "sensex", "nifty", "bank nifty", "banknifty", "nifty bank", "gift nifty", "nse", "bse", "sebi",
    "dalal street", "d-street", "rupee", "fii", "fpi", "dii", r"f&o", "futures and options",
    "mutual fund", "amfi", "crore", "lakh crore", "gst", "union budget", "monsoon", "mcx",
    # Commodities
    r"commodit(?:y|ies)", "oil", "crude", "brent", "wti", r"opec\+?", "natural gas", "gas", "lng",
    "gasoline", "diesel", "fuel", r"refiner(?:y|ies)?", "pipeline", "gold", "silver", "platinum",
    "palladium", "bullion", "precious metal", "copper", r"alumin(?:um|ium)", "zinc", "nickel",
    "iron ore", "steel", "coal", "lithium", "cobalt", "uranium", "rare earth", "metal", "mining",
    "miner", "wheat", "corn", "soybean", "soy", "rice", "sugar", "coffee", "cocoa", "cotton",
    "palm oil", "edible oil", "grain", "crop", "harvest", r"fertili[sz]er", "livestock", "cattle",
    "comex", "nymex", "lme", "cbot",
    # Forex and crypto
    "dollar", "dxy", "greenback", "euro", "yen", "yuan", "renminbi", "pound", "sterling", "franc",
    "peso", "lira", "rupiah", "forex", "fx", r"currenc(?:y|ies)", r"devalu(?:e|ed|ation)", "crypto",
    r"cryptocurrenc(?:y|ies)", "bitcoin", "btc", "ether", "ethereum", "stablecoin", "solana", "xrp",
    "binance", "coinbase", "tether", "blockchain",
    # Geopolitics
    "war", "conflict", "military", "missile", "airstrike", "strike", "invasion", "attack", "troops",
    "ceasefire", "truce", "nuclear", "hormuz", "red sea", "houthi", "strait", r"geopolitic(?:s|al)",
    "coup", "election", "blockade", "drone",
    # Sectors and technology
    "semiconductor", "chip", "chipmaker", "gpu", "cpu", "foundry", "fab", "dram", "nand", "hbm",
    "memory chip", "ai", "artificial intelligence", r"data cent(?:er|re)", "datacenter", "cloud",
    "server", "ev", "electric vehicle", "automaker", "carmaker", "battery", "solar", "renewable",
    "wind power", "power grid", "electricity", r"utilit(?:y|ies)", "pharma", "biotech", "drugmaker",
    "drug", "fda", "vaccine", "clinical trial", "housing", "mortgage", "home sales", "real estate",
    "homebuilder", "airline", "telecom", "5g", "spectrum", "broadband", "bank", "lender", "insurer",
    "e-commerce", "retailer", "retail",
    # Market-moving companies (US / global)
    "apple", "microsoft", "nvidia", "amazon", "alphabet", "google", "meta", "tesla", "broadcom",
    "berkshire", "jpmorgan", "goldman", "goldman sachs", "morgan stanley", r"citi(?:group)?",
    "wells fargo", "bank of america", "netflix", "intel", "amd", "micron", "qualcomm", "oracle",
    "salesforce", "palantir", "boeing", "exxon", "chevron", "walmart", "costco", "pfizer",
    "eli lilly", "novo nordisk", "unitedhealth", "openai", "anthropic", "tsmc", "samsung",
    "sk hynix", "asml", "alibaba", "tencent", "byd", "toyota", "sony", "softbank", "aramco",
    "shell", "bp", "lvmh", "nestle", "hsbc", "ubs",
    # Market-moving companies (India)
    "reliance", "tcs", "infosys", "wipro", "hcl tech", "hdfc", "icici", "sbi", "kotak",
    "axis bank", "adani", "tata", "bajaj", "mahindra", "maruti", "airtel", "itc", r"l&t", "larsen",
    "ongc", "coal india", "ntpc", "vedanta", "zomato", "paytm", "lic", "hindustan unilever",
    "sun pharma", "jio",
]
MARKET_KEYWORD_REGEX = _word_regex(MARKET_RELEVANCE_KEYWORDS)

# Headlines that mention market words but are never market-moving news: commentary,
# listicles, explainers, previews, live blogs and personal-finance content.
NOISE_REGEX = re.compile("|".join([
    r"\?\s*$",  # "Why Did X Stock Jump?", "Will the S&P Open Up?" — commentary, not news
    r"^how to\b",
    r"\bstocks? to (?:buy|watch|sell|avoid|own)\b",
    r"\b(?:best|top \d+) (?:stocks|shares|etfs|funds|picks)\b",
    r"\b\d+ (?:\w+ )?(?:stocks|etfs|shares|funds) (?:to|that|for|with)\b",
    r"\bwhat to (?:know|expect|watch)\b",
    r"\b(?:things|need) to know\b",
    r"\bhere['’]?s (?:why|what|how|the)\b",
    r"\bcramer\b",
    r"\bmotley fool\b",
    r"\((?:video|podcast|audio)\)",
    r"\bpodcast\b",
    r"\bnewsletter\b",
    r"\bopinion\b",
    r"\bexplain(?:ed|er)\b",
    r"\bexplains (?:why|how|what)\b",
    # Daily local price-list pages ("Gold Rate Today in Katni", "Gold price in Pakistan for today")
    r"\b(?:rates?|prices?) today\b",
    r"\bprices? in [\w ]+ for today\b",
    r"\bquiz\b",
    r"\bweek ahead\b",
    r"\blive(?: updates| blog)?:",
    r"\blive updates\b",
    r"\bmorning bid\b",
    r"\b(?:credit cards?|savings accounts?|cd rates?|personal loans?|mortgage rates today)\b",
    r"\bretirement (?:savings|plans?|accounts?)\b",
    r"\bhoroscope\b",
    r"\bsponsored\b",
    r"\breview & preview\b",
]), re.IGNORECASE)


def is_market_relevant(headline):
    if NOISE_REGEX.search(headline):
        return False
    if MARKET_KEYWORD_REGEX.search(headline):
        return True
    # Watchlist tickers count too, matched case-sensitively so a symbol like "ICE" only hits
    # the ticker, not the word "ice".
    return any(
        re.search(rf"(?<![A-Za-z0-9]){re.escape(w['symbol'])}(?![A-Za-z0-9])", headline)
        for w in state.watchlist
        if len(w["symbol"]) >= 3 and w["symbol"].isalpha()
    )


def triage_headlines(headlines):
    """One cheap AI call scores a whole batch. Returns {headline: {"impact", "direction",
    "tickers", "sector"}} for headlines scoring at least QUICK_SIGNAL_MIN_IMPACT ({} if none
    do), or None if every AI provider is unavailable — so the caller can requeue the batch."""
    if not headlines:
        return {}
    numbered = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines))
    prompt = f"""Below are {len(headlines)} news headlines from global, Indian, commodity, forex and
crypto news feeds. Score each one for how much it is likely to move a stock, sector, index,
currency, bond yield, commodity or crypto price today. Indian market news (Sensex/Nifty, RBI,
SEBI, the rupee, large Indian companies) counts just as much as US news.

impact (1-10): 10 = market-wide shock (surprise central bank move, war escalation, crash);
8-9 = major move for a large company, sector, commodity or currency (earnings/guidance surprise,
big M&A, regulatory action, supply shock); 6-7 = clear but limited price impact; 4-5 = minor.
Opinion/commentary, personal-finance advice, listicles, previews of scheduled events with no new
information, recaps of moves with no new cause, and news with no plausible price impact score 1-3.

{numbered}

For each headline scoring {QUICK_SIGNAL_MIN_IMPACT} or more, output one object:
{{"n": <headline number>, "impact": <1-10>, "direction": "BULLISH" or "BEARISH" or "MIXED",
"tickers": [up to 3 exchange-listed symbols most affected, in Yahoo Finance format such as AAPL,
RELIANCE.NS, ^NSEI, GC=F, CL=F, EURUSD=X, BTC-USD — use [] if the company is private or you are
not sure of the symbol; never guess], "sector": "<short sector name>"}}
Respond with ONLY a JSON array of these objects and nothing else. If none qualify, respond with []."""
    raw_text = call_llm(prompt, tier="fast")
    if raw_text is None:
        return None
    try:
        start, end = raw_text.find("["), raw_text.rfind("]")
        items = json.loads(raw_text[start:end + 1]) if start != -1 and end > start else []
    except ValueError as e:
        report_error("Triage (parse)", e)
        return {}
    triaged = {}
    for item in items:
        try:
            n, impact = int(item["n"]), int(item["impact"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 1 <= n <= len(headlines) or impact < QUICK_SIGNAL_MIN_IMPACT:
            continue
        direction = str(item.get("direction", "")).upper()
        triaged[headlines[n - 1]] = {
            "impact": min(impact, 10),
            "direction": direction if direction in ("BULLISH", "BEARISH") else "MIXED",
            "tickers": [str(t).strip() for t in (item.get("tickers") or []) if str(t).strip()][:3],
            "sector": item.get("sector") or "General Markets",
        }
    return triaged


def av_get(params):
    if not ALPHAVANTAGE_API_KEY or state.av_exhausted:
        return None
    elapsed = time.time() - state.last_av_call_time
    if elapsed < AV_MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(AV_MIN_SECONDS_BETWEEN_CALLS - elapsed)
    resp = requests.get("https://www.alphavantage.co/query", params={**params, "apikey": ALPHAVANTAGE_API_KEY}, timeout=15)
    state.last_av_call_time = time.time()
    state.av_calls_today += 1
    payload = resp.json()
    if "Note" in payload or "Information" in payload:
        # The free tier allows only a few dozen calls a day. Once it says so, every further
        # call today would also fail — after paying the 13s pacing sleep each time, which
        # stalled every headline's analysis. Skip straight to yfinance until the daily reset.
        state.av_exhausted = True
        return None
    return payload



def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    deltas = [b - a for a, b in zip(closes[-period - 1:], closes[-period:])]
    avg_gain = sum(d for d in deltas if d > 0) / period
    avg_loss = sum(-d for d in deltas if d < 0) / period
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


def fetch_closes_yf(ticker):
    """One cached 3-month daily-close download per symbol, shared by the quote/RSI path and
    the sparkline path (which previously downloaded 2mo and 3mo separately). Converted to a
    plain list right away so the pandas DataFrame doesn't outlive this call."""
    cache_key = f"CLOSES:{ticker}"
    cached = get_cached_market_data(cache_key)
    if cached:
        return cached
    try:
        hist = yf.Ticker(ticker).history(period="3mo", timeout=10)
        if hist.empty:
            return None
        result = {"closes": [float(c) for c in hist["Close"].tolist()], "as_of": str(hist.index[-1].date())}
        del hist
        set_cached_market_data(cache_key, result, "yf")
        return result
    except Exception:
        return None


def fetch_market_data_yf(ticker):
    history = fetch_closes_yf(ticker)
    if not history or len(history["closes"]) < 2:
        return None
    closes = history["closes"]
    latest_price = closes[-1]
    prev_price = closes[-2]
    change_percent = f"{((latest_price - prev_price) / prev_price) * 100:.2f}%" if prev_price else None
    return {
        "price": round(latest_price, 2),
        "change_percent": change_percent,
        "rsi": compute_rsi(closes),
        "as_of": history["as_of"]
    }


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


_COMPANY_SUFFIXES = re.compile(r"\b(?:inc|corp|corporation|co|company|ltd|limited|plc|ag|sa|nv|holdings?|group)\b\.?", re.IGNORECASE)
# Single-stock and thematic funds carry the company's name and are sometimes mislabeled as
# EQUITY (live example: "Direxion Daily SpaceX Bull 2X ETF" for "SpaceX").
_FUND_NAME = re.compile(r"\b(?:etf|etn|fund|[23]x|leveraged|inverse|ultra|bull|bear)\b", re.IGNORECASE)


def _normalize_company(name):
    return " ".join(_COMPANY_SUFFIXES.sub(" ", re.sub(r"[^\w&\s]", " ", name.lower())).split())


def resolve_ticker(company_or_ticker):
    """Maps an AI-supplied ticker or company name to a real listed symbol, or None. Search
    always returns its closest matches even for companies that aren't listed (live example:
    "Anthropic", which is private, came back as ANTW — a thematic "Anthropic AI Lab
    Ecosystem ETF" — and a trade was written on it), so a result only counts if it is
    exactly the symbol given, or a stock (never an ETF) whose listed name contains the
    company name given."""
    if not company_or_ticker:
        return None
    query = company_or_ticker.strip()
    cache = state.symbol_cache
    if query in cache:
        return cache[query]
    try:
        quotes = yf.Search(query, timeout=10).quotes
    except Exception:
        return None  # not cached, so a transient failure gets retried next time
    resolved = next((q["symbol"] for q in quotes if q.get("symbol", "").upper() == query.upper()), None)
    if not resolved:
        wanted = _normalize_company(query)
        for q in quotes:
            listed_name = _normalize_company(q.get("longname") or q.get("shortname") or "")
            if (wanted and q.get("quoteType") == "EQUITY" and not _FUND_NAME.search(listed_name)
                    and re.search(rf"\b{re.escape(wanted)}\b", listed_name)):
                resolved = q["symbol"]
                break
    cache[query] = resolved
    return resolved


def fetch_market_data(ticker):
    """yfinance first — free with no daily cap and no pacing sleep. Alpha Vantage's ~25
    free calls/day are only a fallback for symbols yfinance can't price."""
    cached = get_cached_market_data(ticker)
    if cached:
        return cached
    data = fetch_market_data_yf(ticker)
    if data:
        set_cached_market_data(ticker, data, "yf")
        return data
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
    return None


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


_COMMODITY_PATTERNS = [(_word_regex([re.escape(k.strip()) for k in keywords]), rest) for keywords, *rest in COMMODITY_MAP]


def detect_commodity(headline):
    for pattern, (function, interval, display_name, metal_key) in _COMMODITY_PATTERNS:
        if pattern.search(headline):
            return function, interval, display_name, metal_key
    return None


INDEX_MAP = [
    (["bank nifty", "banknifty", "nifty bank"], "^NSEBANK", "Bank Nifty"),
    (["sensex", "bse"], "^BSESN", "Sensex"),
    (["nifty", "nse", "f&o", "futures and options", "futures & options"], "^NSEI", "Nifty 50"),
]


_INDEX_PATTERNS = [(_word_regex([re.escape(k) for k in keywords]), yf_symbol, display_name) for keywords, yf_symbol, display_name in INDEX_MAP]


def detect_index(headline):
    for pattern, yf_symbol, display_name in _INDEX_PATTERNS:
        if pattern.search(headline):
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

    # Futures prices from yfinance first: daily, free, no cap and no pacing sleep. Alpha
    # Vantage (many of its commodity series are only monthly) is the fallback for
    # commodities yfinance has no symbol for, e.g. aluminum.
    yf_symbol = YF_METAL_SYMBOLS.get(metal_key) if metal_key else YF_COMMODITY_SYMBOLS.get(function)
    data = fetch_market_data_yf(yf_symbol) if yf_symbol else None
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
            return None

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
            return None

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
        return None


def run_deep_analysis(headline, sector, resolved_ticker):
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

    # Ripple plays come back from this same call — previously a separate AI call per
    # headline, which paid for the prompt and the model's hidden reasoning twice.
    prompt = f"""
    Analyze this high-impact market headline: "{headline}"
    Likely affected sector: {sector}. Verified ticker for grounding: {resolved_ticker or "none found"}.
    {grounding}
    Provide an institutional-grade trading setup across US Stocks, Indian Markets, Global Stocks, Crypto, and Commodities.
    Only name tickers you are confident are real, exchange-listed symbols, in Yahoo Finance format (AAPL, RELIANCE.NS,
    KGX.DE, 7203.T); leave a list empty rather than guess.
    Also list up to 4 related plays that could ripple from this news across the supply chain, competitors,
    commodities or currencies — specific companies preferred over broad ETFs — each with a direction.
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
        "key_takeaways": ["Short takeaway 1", "Short takeaway 2", "Short takeaway 3"],
        "ripple_effects": [{{"name": "Company or ticker", "direction": "BULLISH or BEARISH"}}]
    }}
    """
    raw_text = call_llm(prompt, tier="deep")
    if raw_text is None:
        return {"sentiment": "ERROR", "sector": "None", "buy_targets": [], "sell_targets": [],
                "strategy": "All AI providers failed — see the Last Error banner for details.",
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


# "every" = minimum seconds between polls of that source (0 = every scan). Slow-moving or
# rate-limit-sensitive sources (Google News, central banks) are polled less often.
# Checked live: Moneycontrol's feeds are years stale, Kitco's 404s, so neither is listed.
NEWS_SOURCES = [
    # Global markets and economy
    {"name": "Bloomberg Markets", "url": "https://feeds.bloomberg.com/markets/news.rss", "region": "Global"},
    {"name": "Bloomberg Economics", "url": "https://feeds.bloomberg.com/economics/news.rss", "region": "Global"},
    {"name": "Bloomberg Politics", "url": "https://feeds.bloomberg.com/politics/news.rss", "region": "Global", "every": 60},
    {"name": "Bloomberg Technology", "url": "https://feeds.bloomberg.com/technology/news.rss", "region": "Global", "every": 60},
    {"name": "CNBC", "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html", "region": "Global"},
    {"name": "CNBC World", "url": "https://www.cnbc.com/id/100727362/device/rss/rss.html", "region": "Global"},
    {"name": "CNBC Finance", "url": "https://www.cnbc.com/id/10000664/device/rss/rss.html", "region": "Global"},
    {"name": "CNBC Earnings", "url": "https://www.cnbc.com/id/15839135/device/rss/rss.html", "region": "Global", "every": 60},
    {"name": "MarketWatch", "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories", "region": "Global"},
    {"name": "MarketWatch Bulletins", "url": "https://feeds.content.dowjones.io/public/rss/mw_bulletins", "region": "Global"},
    {"name": "WSJ Markets", "url": "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain", "region": "Global"},
    {"name": "WSJ World", "url": "https://feeds.content.dowjones.io/public/rss/RSSWorldNews", "region": "Global", "every": 60},
    {"name": "Financial Times", "url": "https://www.ft.com/markets?format=rss", "region": "Global"},
    {"name": "Yahoo Finance", "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US", "region": "Global"},
    {"name": "Investing.com", "url": "https://www.investing.com/rss/news_25.rss", "region": "Global"},
    {"name": "Investing.com Economy", "url": "https://www.investing.com/rss/news_14.rss", "region": "Global"},
    {"name": "Investing.com Indicators", "url": "https://www.investing.com/rss/news_95.rss", "region": "Global"},
    {"name": "Seeking Alpha", "url": "https://seekingalpha.com/market_currents.xml", "region": "Global"},
    {"name": "Benzinga", "url": "https://www.benzinga.com/feed", "region": "Global", "every": 60},
    {"name": "Reuters (Google News)", "url": "https://news.google.com/rss/search?q=site:reuters.com+markets+when:1h&hl=en-US&gl=US&ceid=US:en", "region": "Global", "every": 180},
    {"name": "Reuters Business (Google News)", "url": "https://news.google.com/rss/search?q=site:reuters.com+business+when:1h&hl=en-US&gl=US&ceid=US:en", "region": "Global", "every": 180},
    {"name": "Federal Reserve", "url": "https://www.federalreserve.gov/feeds/press_all.xml", "region": "Global", "every": 300},
    {"name": "ECB", "url": "https://www.ecb.europa.eu/rss/press.html", "region": "Global", "every": 300},
    # India
    {"name": "Economic Times Markets", "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "region": "India"},
    {"name": "Economic Times Stocks", "url": "https://economictimes.indiatimes.com/markets/stocks/news/rssfeeds/2146842.cms", "region": "India"},
    {"name": "Economic Times Economy", "url": "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms", "region": "India"},
    {"name": "LiveMint Markets", "url": "https://www.livemint.com/rss/markets", "region": "India"},
    {"name": "LiveMint Economy", "url": "https://www.livemint.com/rss/economy", "region": "India", "every": 60},
    {"name": "Business Standard Markets", "url": "https://www.business-standard.com/rss/markets-106.rss", "region": "India"},
    {"name": "Business Standard Economy", "url": "https://www.business-standard.com/rss/economy-102.rss", "region": "India", "every": 60},
    {"name": "BusinessLine Markets", "url": "https://www.thehindubusinessline.com/markets/feeder/default.rss", "region": "India"},
    {"name": "NDTV Profit", "url": "https://feeds.feedburner.com/ndtvprofit-latest", "region": "India"},
    {"name": "India Markets (Google News)", "url": "https://news.google.com/rss/search?q=sensex+OR+nifty+OR+sebi+OR+rbi+when:1h&hl=en-IN&gl=IN&ceid=IN:en", "region": "India", "every": 180},
    {"name": "RBI", "url": "https://www.rbi.org.in/pressreleases_rss.xml", "region": "India", "every": 300},
    # Commodities
    {"name": "Investing.com Commodities", "url": "https://www.investing.com/rss/news_11.rss", "region": "Commodities"},
    {"name": "Economic Times Commodities", "url": "https://economictimes.indiatimes.com/markets/commodities/rssfeeds/1808152121.cms", "region": "Commodities"},
    {"name": "OilPrice.com", "url": "https://oilprice.com/rss/main", "region": "Commodities", "every": 60},
    {"name": "Mining.com", "url": "https://www.mining.com/feed/", "region": "Commodities", "every": 120},
    {"name": "Commodities (Google News)", "url": "https://news.google.com/rss/search?q=crude+OR+gold+OR+copper+OR+opec+prices+when:1h&hl=en-US&gl=US&ceid=US:en", "region": "Commodities", "every": 180},
    # Forex
    {"name": "Investing.com Forex", "url": "https://www.investing.com/rss/news_1.rss", "region": "Forex"},
    {"name": "FXStreet", "url": "https://www.fxstreet.com/rss/news", "region": "Forex"},
    # Crypto
    {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/", "region": "Crypto", "every": 60},
    {"name": "Cointelegraph", "url": "https://cointelegraph.com/rss", "region": "Crypto", "every": 60},
    # Finnhub news API — only polled when FINNHUB_API_KEY is set.
    {"name": "Finnhub General", "finnhub_category": "general", "region": "Global", "every": 60},
    {"name": "Finnhub Mergers", "finnhub_category": "merger", "region": "Global", "every": 60},
    {"name": "Finnhub Forex", "finnhub_category": "forex", "region": "Forex", "every": 60},
    {"name": "Finnhub Crypto", "finnhub_category": "crypto", "region": "Crypto", "every": 60},
]

_FEED_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
# One TLS context for every feed request. requests re-loads its CA bundle for each new
# connection, and news servers close idle connections between scans, so ~43 polls per
# scan each paid that cost — measured ~70x more CPU than urllib with this shared context,
# enough to starve request handling on Render's fractional free-tier CPU.
# certifi's bundle (what requests used) rather than the OS store, which lacks some feeds'
# roots on some platforms (ECB failed verification against the Windows store).
_FEED_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
_source_last_polled = {}
_feed_validators = {}  # url -> ETag / Last-Modified, for conditional GETs
_feed_body_hashes = {}  # url -> hash of the last body parsed, for feeds without validators
_undated_titles = {}  # source name -> undated titles seen on its previous poll
_recent_story_tokens = collections.deque(maxlen=2000)  # (accepted_at, word set) for dedup
_DEDUP_STOPWORDS = frozenset(
    "the a an and or of to in on for with as at by from after amid over its it is are be was "
    "were has have had says said new than this that into up down".split()
)


def _xml_name(tag):
    return tag.rsplit("}", 1)[-1]  # drop any XML namespace


def _parse_utc(text):
    """RFC 822 (RSS) or ISO 8601 (Atom) date -> naive UTC. None if unparseable or if it
    has no timezone, since a zone-less time can't be placed (RBI's are IST, unlabeled)."""
    try:
        dt = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_feed_fast(body):
    """Extracts just (title, link, published, publisher) per item with the C-accelerated
    XML parser — measured 14-37x less CPU than feedparser, which sanitizes every field of
    every entry. That matters on Render's free fractional CPU, where feedparser parsing all
    feeds every scan starved request handling (observed: /api/status timing out at 30s).
    Returns None — meaning "use feedparser" — for anything it can't handle with certainty."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return None
    items = []
    for element in root.iter():
        if _xml_name(element.tag) not in ("item", "entry"):
            continue
        title = link = date_text = publisher = None
        for child in element:
            name = _xml_name(child.tag)
            if name == "title":
                title = "".join(child.itertext())
            elif name == "link":
                link = child.get("href") or (child.text or "").strip() or link
            elif name in ("pubDate", "published", "date") or (name == "updated" and not date_text):
                date_text = (child.text or "").strip() or date_text
            elif name == "source":
                publisher = (child.text or "").strip() or None
        published = _parse_utc(date_text) if date_text else None
        if date_text and published is None:
            return None  # a date we can't place with certainty: let feedparser decide
        items.append((title or "", link, published, publisher))
    return items


def _parse_feed_slow(body):
    items = []
    for entry in feedparser.parse(body).entries:
        # published_parsed is UTC — the article's real publish time, not when we saw it.
        published = entry.get("published_parsed") or entry.get("updated_parsed")
        items.append((entry.get("title", ""), entry.get("link"), datetime(*published[:6]) if published else None,
                      entry.get("source", {}).get("title")))
    return items


def _http_get(url, headers=None):
    """GET -> (body, response headers), or (None, None) on 304 Not Modified. Raises on
    other HTTP errors and network failures."""
    request = urllib.request.Request(url, headers={"User-Agent": _FEED_USER_AGENT, **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=FEED_TIMEOUT_SECONDS, context=_FEED_SSL_CONTEXT) as resp:
            body = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            return body, resp.headers
    except urllib.error.HTTPError as e:
        if e.code == 304:
            return None, None
        raise


def _fetch_rss(source):
    """Returns [(title, link, published_utc_or_None, publisher_or_None)], or None if the
    feed is unchanged since the last poll — either a 304 from a conditional GET, or a body
    byte-identical to the last one parsed (for feeds that don't support conditional GETs)."""
    url = source["url"]
    validators = _feed_validators.get(url, {})
    headers = {}
    if validators.get("etag"):
        headers["If-None-Match"] = validators["etag"]
    if validators.get("modified"):
        headers["If-Modified-Since"] = validators["modified"]
    body, resp_headers = _http_get(url, headers)
    if body is None:
        return None
    _feed_validators[url] = {"etag": resp_headers.get("ETag"), "modified": resp_headers.get("Last-Modified")}
    body_hash = hashlib.blake2b(body, digest_size=16).digest()
    if _feed_body_hashes.get(url) == body_hash:
        return None
    _feed_body_hashes[url] = body_hash

    raw_items = _parse_feed_fast(body)
    if raw_items is None:
        raw_items = _parse_feed_slow(body)
    items = []
    for title, link, published, publisher in raw_items:
        # Some feeds double-escape entities ("M&amp;M"); unescape once more and tidy spaces.
        title = " ".join(html.unescape(title).split())
        if not title:
            continue
        # Google News titles end in " - Reuters" etc.; keep the publisher, drop the suffix.
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(publisher) - 3]
        items.append((title, link, published, publisher))
    return items


def _fetch_finnhub(source):
    query = urllib.parse.urlencode({"category": source["finnhub_category"], "token": FINNHUB_API_KEY})
    body, _ = _http_get(f"https://finnhub.io/api/v1/news?{query}")
    items = []
    for article in json.loads(body or b"[]"):
        title = " ".join((article.get("headline") or "").split())
        if not title:
            continue
        published = datetime.fromtimestamp(article["datetime"], timezone.utc).replace(tzinfo=None) if article.get("datetime") else None
        items.append((title, article.get("url"), published, article.get("source")))
    return items


def _fetch_source(source):
    try:
        if "finnhub_category" in source:
            return source, _fetch_finnhub(source)
        return source, _fetch_rss(source)
    except Exception as e:
        print(f"[pipeline] Feed '{source['name']}' failed: {str(e)[:150]}")
        return source, None


def _story_tokens(title):
    return frozenset(w for w in re.findall(r"[a-z0-9$%&]+", title.lower()) if len(w) > 1 and w not in _DEDUP_STOPWORDS)


def _is_duplicate_story(tokens, now):
    """True if a story with mostly the same words was already accepted recently — the same
    news syndicated across outlets with reworded titles would otherwise become several
    separate signals."""
    if len(tokens) < 4:
        return False
    for accepted_at, seen in _recent_story_tokens:
        if now - accepted_at <= DEDUP_WINDOW_SECONDS and len(tokens & seen) / len(tokens | seen) >= DEDUP_SIMILARITY:
            return True
    return False


def fetch_live_financial_news():
    """Polls every due source in parallel and returns the new headlines that are fresh,
    market-relevant and not a duplicate of a story already queued."""
    now = time.time()
    due = [
        s for s in NEWS_SOURCES
        if ("finnhub_category" not in s or FINNHUB_API_KEY)
        and now - _source_last_polled.get(s["name"], 0) >= s.get("every", 0)
    ]
    for s in due:
        _source_last_polled[s["name"]] = now
    with concurrent.futures.ThreadPoolExecutor(max_workers=FEED_FETCH_WORKERS) as executor:
        results = list(executor.map(_fetch_source, due))

    cutoff = datetime.utcnow() - timedelta(minutes=MAX_HEADLINE_AGE_MINUTES)
    new_headlines_found = []
    marked_any = False
    for source, items in results:
        if items is None:
            continue
        # Undated entries (e.g. RBI press releases) have no publish time to check, so they
        # count as new only if they weren't in this source's previous poll — and nothing is
        # new on the first poll after startup, when there's no previous poll to compare to.
        undated_before = _undated_titles.get(source["name"])
        _undated_titles[source["name"]] = {title for title, _, published, _ in items if published is None}
        for title, link, published_dt, publisher in items:
            if title in state.processed_headlines:
                continue
            state.processed_headlines[title] = now
            marked_any = True
            if published_dt is None:
                if undated_before is None or title in undated_before:
                    continue
                published_dt = datetime.utcnow()
            if published_dt < cutoff or not is_market_relevant(title):
                continue
            tokens = _story_tokens(title)
            if _is_duplicate_story(tokens, now):
                continue
            _recent_story_tokens.append((now, tokens))
            new_headlines_found.append(title)
            state.headline_metadata[title] = {
                "link": link,
                "source": publisher if publisher and publisher != source["name"] else source["name"],
                "region": source["region"],
                "published_at": published_dt.isoformat() + "Z",
                "queued_at": now,
            }

    if marked_any:
        # Bound the in-memory copies too, not just what gets persisted — otherwise they grow
        # for the life of the process on Render's 512MB instance.
        _trim_oldest(state.processed_headlines, MAX_STORED_HEADLINES)
        _trim_oldest(state.headline_metadata, MAX_HEADLINE_METADATA)
        if now - state.headlines_saved_at >= HEADLINES_SAVE_INTERVAL_SECONDS:
            save_processed_headlines(state.processed_headlines)
            state.headlines_saved_at = now
    return new_headlines_found


def _trim_oldest(d, max_size):
    excess = len(d) - max_size
    if excess > 0:
        for key in list(itertools.islice(d, excess)):
            del d[key]


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


def _headline_published(headline):
    """A queued headline's publish time (naive UTC); datetime.min if its metadata is gone,
    so it sorts last and counts as stale."""
    published_at = state.headline_metadata.get(headline, {}).get("published_at")
    try:
        return datetime.fromisoformat(published_at.removesuffix("Z"))
    except (AttributeError, ValueError):
        return datetime.min


def run_pipeline_cycle():
    """One tick of the pipeline: scan feeds if the interval elapsed, then batch-filter or
    deep-analyze whatever is in the queue. This is the same
    logic that used to live inline in app.py's pipeline_fragment(), just without any
    Streamlit UI calls — the scheduler calls this repeatedly in the background.
    Reads refresh_interval_seconds/active from state so PATCH /api/settings can change
    behavior at runtime without restarting the scheduler."""
    state.roll_daily_usage_if_needed()
    if not state.active:
        return None
    now = time.time()

    # Scans run on schedule even while older headlines are still queued — waiting for the
    # queue to drain first meant breaking news sat unseen behind a backlog.
    if now - state.last_scan_time >= state.refresh_interval_seconds:
        state.last_scan_time = now
        new_headlines = fetch_live_financial_news()
        if new_headlines:
            state.pending_headlines.extend(new_headlines)
            del state.pending_headlines[:-MAX_PENDING_HEADLINES]

    cutoff = datetime.utcnow() - timedelta(minutes=MAX_HEADLINE_AGE_MINUTES)
    state.qualified_headlines = [h for h in state.qualified_headlines if _headline_published(h) >= cutoff]
    state.pending_headlines = [h for h in state.pending_headlines if _headline_published(h) >= cutoff]

    oldest_wait = max((now - state.headline_metadata.get(h, {}).get("queued_at", now) for h in state.pending_headlines), default=0)
    triage_due = len(state.pending_headlines) >= TRIAGE_MIN_BATCH or oldest_wait >= TRIAGE_MAX_WAIT_SECONDS
    if not state.qualified_headlines and triage_due and time.time() >= state.ai_unavailable_until:
        # Newest first: the freshest news is the most tradeable.
        state.pending_headlines.sort(key=_headline_published, reverse=True)
        batch = state.pending_headlines[:BATCH_FILTER_SIZE]
        state.pending_headlines = state.pending_headlines[len(batch):]
        triaged = triage_headlines(batch)
        if triaged is None:
            state.pending_headlines[:0] = batch  # every AI provider is down: retry after cooldown
        else:
            threshold = deep_impact_threshold()
            for headline, triage in triaged.items():
                state.headline_metadata.setdefault(headline, {})["triage"] = triage
                if triage["impact"] >= threshold:
                    state.qualified_headlines.append(headline)
                else:
                    _publish_alert(headline, _quick_alert(headline, triage))
            # Biggest news first, then newest.
            state.qualified_headlines.sort(key=lambda h: (_headline_triage(h).get("impact", 0), _headline_published(h)), reverse=True)

    if not state.qualified_headlines:
        return None

    headline = state.qualified_headlines.pop(0)
    triage = _headline_triage(headline)
    primary = (triage.get("tickers") or [None])[0]
    resolved_ticker = None if (detect_commodity(headline) or detect_index(headline)) else resolve_ticker(primary)
    ai_blueprint = run_deep_analysis(headline, triage.get("sector", "General Markets"), resolved_ticker)
    if ai_blueprint.get("sentiment") == "ERROR" and triage:
        # Deep analysis unavailable — still publish what triage already knows rather than
        # dropping a high-impact headline or showing an error card.
        new_alert = _quick_alert(headline, triage)
        _publish_alert(headline, new_alert)
        return new_alert

    ripple_effects = [r for r in (ai_blueprint.get("ripple_effects") or []) if isinstance(r, dict)]
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
        # The article's actual RSS publish time (UTC, ISO 8601) — not when our
        # pipeline finished analyzing it, which can lag by however long the
        # headline sat in the filter/analysis queue. The frontend renders
        # this in the viewer's own local timezone.
        "Timestamp": source_meta.get("published_at") or datetime.utcnow().isoformat() + "Z",
        "Headline": headline,
        "Summary": ai_blueprint.get("summary"),
        # Always a string: the frontend calls .toUpperCase() on it.
        "Sentiment": ai_blueprint.get("sentiment") or "NEUTRAL",
        "Sector": ai_blueprint.get("sector") or triage.get("sector", "General Markets"),
        "Buy Tickers": ", ".join(_valid_tickers(ai_blueprint.get("buy_targets"))),
        "Sell Tickers": ", ".join(_valid_tickers(ai_blueprint.get("sell_targets"))),
        "Execution Blueprint": ai_blueprint.get("strategy"),
        "Grounded Data": grounded_reading,
        "Ripple Effects (AI-inferred, unverified)": ripple_display,
        "Category": ai_blueprint.get("category", "Signal"),
        "Key Takeaways": ai_blueprint.get("key_takeaways", []),
        "Impact": triage.get("impact"),
        "Analysis": "Deep",
        "Region": source_meta.get("region"),
        "Source": source_meta.get("source"),
        "Article Link": source_meta.get("link")
    }
    _publish_alert(headline, new_alert)
    return new_alert


def _headline_triage(headline):
    return state.headline_metadata.get(headline, {}).get("triage", {})


def _valid_tickers(symbols):
    """AI-named tickers that resolve to real listed symbols; unverifiable ones are dropped
    rather than shown as tradeable."""
    resolved = (resolve_ticker(str(s)) for s in (symbols or []) if isinstance(s, str))
    return list(dict.fromkeys(t for t in resolved if t))


def _quick_alert(headline, triage):
    """A signal built from the triage answer alone — no extra AI call — for headlines that
    matter but don't clear the deep-analysis bar, or when deep analysis is unavailable."""
    meta = state.headline_metadata.get(headline, {})
    tickers = _valid_tickers(triage["tickers"])
    direction = triage["direction"]
    return {
        "Timestamp": meta.get("published_at") or datetime.utcnow().isoformat() + "Z",
        "Headline": headline,
        "Summary": None,
        "Sentiment": {"BULLISH": "BULLISH", "BEARISH": "BEARISH"}.get(direction, "NEUTRAL"),
        "Sector": triage["sector"],
        "Buy Tickers": ", ".join(tickers) if direction == "BULLISH" else "",
        "Sell Tickers": ", ".join(tickers) if direction == "BEARISH" else "",
        "Tickers": ", ".join(tickers),
        "Execution Blueprint": f"Quick signal (impact {triage['impact']}/10) — scored by AI triage, not deep-analyzed.",
        "Grounded Data": "Not fetched for quick signals",
        "Ripple Effects (AI-inferred, unverified)": "None identified",
        "Category": "Market News",
        "Key Takeaways": [],
        "Impact": triage["impact"],
        "Analysis": "Quick",
        "Region": meta.get("region"),
        "Source": meta.get("source"),
        "Article Link": meta.get("link"),
    }


def _publish_alert(headline, alert):
    state.trade_history.insert(0, alert)
    del state.trade_history[MAX_STORED_ALERTS:]
    state.headline_metadata.pop(headline, None)
    append_signal(alert)


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
    the Market Data screen. Reuses the same cached download as fetch_market_data_yf."""
    history = fetch_closes_yf(ticker)
    if not history:
        return None
    return [round(c, 2) for c in history["closes"][-points:]]


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
                closes = [float(c) for c in hist["Close"].tolist()] if not hist.empty else None
                del hist
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
    """Raises ValueError if the symbol doesn't resolve to real yfinance data — without
    this check, a bad symbol (e.g. a company name an AI ripple-effect guess produced that
    isn't an actual traded ticker, like "SPACEX" instead of its real SPAC ticker "SPCX")
    would sit on the watchlist forever, failing on every single warm cycle and flooding
    the logs with "No data found, symbol may be delisted" indefinitely."""
    symbol = symbol.strip().upper()
    if any(w["symbol"] == symbol for w in state.watchlist):
        return state.watchlist
    if fetch_market_data_yf(symbol) is None:
        raise ValueError(f"'{symbol}' does not resolve to a real, tradeable yfinance symbol")
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
        # One task for all timeframes, run sequentially: the first one downloads the 12 1-year
        # ETF histories and the rest hit that cache. Submitting one task per timeframe made all
        # four race on a cold cache and download the same histories up to 4x concurrently.
        sector_future = executor.submit(
            lambda: {tf: fetch_sector_performance(tf) for tf in TIMEFRAME_LOOKBACK_TRADING_DAYS}
        )
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

        try:
            state.cached_sectors.update(sector_future.result())
        except Exception as e:
            report_error("Market data prefetch (sectors)", e)

        try:
            state.cached_watchlist = watchlist_future.result()
        except Exception as e:
            report_error("Market data prefetch (watchlist)", e)

    state.cache_last_updated = time.time()
    # Each cycle churns through dozens of short-lived yfinance/pandas objects; collect them
    # now rather than letting cyclic garbage pile up between automatic GC passes.
    gc.collect()
