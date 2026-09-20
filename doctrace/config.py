"""
Runtime settings read from the environment, in one place.

Nothing here has a side effect at import time, so tests and scripts can import it freely.
"""

import os
from functools import lru_cache

_DEFAULT_ORIGINS = "http://localhost:8501,http://127.0.0.1:8501"


@lru_cache(maxsize=1)
def get_device() -> str:
    """'cuda' when a GPU is usable, otherwise 'cpu' (slow, but it works)."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def allowed_origins() -> list[str]:
    """Browser origins that may call the HTTP service (DOCTRACE_ALLOWED_ORIGINS,
    comma separated). The default covers the local dashboard only."""
    raw = os.getenv("DOCTRACE_ALLOWED_ORIGINS", _DEFAULT_ORIGINS)
    return [o.strip() for o in raw.split(",") if o.strip()]


def api_key() -> str | None:
    """Shared secret for /v1/ask (DOCTRACE_API_KEY). Unset means the route is open,
    which is fine on localhost and not on a shared network."""
    return os.getenv("DOCTRACE_API_KEY") or None


def bind_host() -> str:
    """Interface the service listens on. Loopback unless told otherwise."""
    return os.getenv("DOCTRACE_HOST", "127.0.0.1")


def bind_port() -> int:
    return int(os.getenv("DOCTRACE_PORT", "8000"))
