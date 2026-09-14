import asyncio

from backend import pipeline

# Default scan interval — mirrors app.py's sidebar slider default. Exposed as a mutable
# module-level value so a future POST /api/settings could adjust it without a restart.
REFRESH_INTERVAL_SECONDS = 30

# Ticks fast (the same 5s cadence app.py's fragment used) so any backlog of already-queued
# headlines drains quickly, while fetch_live_financial_news() itself is only actually
# called once REFRESH_INTERVAL_SECONDS has elapsed (checked inside run_pipeline_cycle).
TICK_SECONDS = 5

_task = None


async def _loop():
    while True:
        try:
            pipeline.run_pipeline_cycle(REFRESH_INTERVAL_SECONDS)
        except Exception as e:
            pipeline.report_error("Scheduler loop", e)
        await asyncio.sleep(TICK_SECONDS)


def start():
    """Launch the background pipeline loop on the running event loop. Runs independent of
    any connected HTTP client — this is the point of moving off Streamlit's run_every,
    which only ticked while a browser tab with an open script-run was present."""
    global _task
    if _task is None or _task.done():
        _task = asyncio.create_task(_loop())
    return _task


def stop():
    global _task
    if _task is not None:
        _task.cancel()
        _task = None
