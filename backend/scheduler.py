import asyncio

from backend import pipeline

# Ticks fast (the same 5s cadence app.py's fragment used) so any backlog of already-queued
# headlines drains quickly, while fetch_live_financial_news() itself is only actually
# called once pipeline.state.refresh_interval_seconds has elapsed (checked inside
# run_pipeline_cycle). The interval and active/paused flag now live on pipeline.state
# instead of a module constant here, so PATCH /api/settings can change them live.
TICK_SECONDS = 5

# Separate, faster-than-cache-TTL cadence for warming market-data caches (ticker bar,
# categorized market data, sector performance, watchlist). YF_CACHE_SECONDS is 60, so
# warming every 45s guarantees the cache never goes cold between refreshes — every
# GET request hits an already-warm cache instead of blocking on a live yfinance call.
MARKET_DATA_WARM_SECONDS = 45

_pipeline_task = None
_market_data_task = None


async def _pipeline_loop():
    while True:
        try:
            # run_pipeline_cycle() is fully synchronous/blocking under the hood — feedparser,
            # requests.get/post for the RSS feeds and AI providers, yfinance calls, and even a
            # literal time.sleep() for Alpha Vantage's rate-limit pacing. FastAPI/uvicorn runs
            # on a single event loop, so calling it directly here froze request handling for
            # every connected client (including trivial in-memory endpoints like /api/status)
            # for as long as that blocking call took — observed live as a 35s hang on
            # /api/status while a scan/analysis cycle was in progress. asyncio.to_thread runs
            # it on a worker thread instead, so the event loop stays free to serve requests.
            await asyncio.to_thread(pipeline.run_pipeline_cycle)
        except Exception as e:
            pipeline.report_error("Scheduler loop", e)
        await asyncio.sleep(TICK_SECONDS)


async def _market_data_warm_loop():
    loop = asyncio.get_running_loop()
    while True:
        try:
            # prefetch_all_market_data() is blocking (thread-pooled network I/O internally),
            # so run it off the event loop thread rather than freezing request handling.
            await loop.run_in_executor(None, pipeline.prefetch_all_market_data)
        except Exception as e:
            pipeline.report_error("Market data warm loop", e)
        await asyncio.sleep(MARKET_DATA_WARM_SECONDS)


def start():
    """Launch both background loops on the running event loop. Runs independent of any
    connected HTTP client — this is the point of moving off Streamlit's run_every, which
    only ticked while a browser tab with an open script-run was present."""
    global _pipeline_task, _market_data_task
    if _pipeline_task is None or _pipeline_task.done():
        _pipeline_task = asyncio.create_task(_pipeline_loop())
    if _market_data_task is None or _market_data_task.done():
        _market_data_task = asyncio.create_task(_market_data_warm_loop())
    return _pipeline_task, _market_data_task


def stop():
    global _pipeline_task, _market_data_task
    if _pipeline_task is not None:
        _pipeline_task.cancel()
        _pipeline_task = None
    if _market_data_task is not None:
        _market_data_task.cancel()
        _market_data_task = None
