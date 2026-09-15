import json
import os

import requests

UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")


def is_configured() -> bool:
    return bool(UPSTASH_URL and UPSTASH_TOKEN)


def _headers():
    return {"Authorization": f"Bearer {UPSTASH_TOKEN}"}


def redis_get_json(key, default):
    """Reads a JSON value from Upstash Redis. Falls back to `default` on any
    failure (not configured, network error, missing key) so a transient
    Upstash issue never crashes a request — it just behaves like a cold
    start, same as before this migration."""
    if not is_configured():
        return default
    try:
        resp = requests.get(f"{UPSTASH_URL}/get/{key}", headers=_headers(), timeout=10)
        resp.raise_for_status()
        result = resp.json().get("result")
        return default if result is None else json.loads(result)
    except Exception:
        return default


def redis_set_json(key, value) -> bool:
    """Writes a JSON value to Upstash Redis. Returns False on failure instead
    of raising, so callers can report it via the existing report_error()
    pattern without a try/except at every call site."""
    if not is_configured():
        return False
    try:
        resp = requests.post(
            f"{UPSTASH_URL}/set/{key}",
            headers=_headers(),
            data=json.dumps(value),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("result") == "OK"
    except Exception:
        return False
