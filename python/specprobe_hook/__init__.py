"""spec-probe hook package — rejection-sampler patch, JSONL sink, FastAPI bridge."""

from __future__ import annotations

from .event_to_trace import event_to_trace, events_to_strategy_bundle
from .jsonl_sink import JsonlSink, get_sink
from .writer import SpeculativeTraceHook, SpeculativeTraceWriter

try:
    from .vllm_patch import install, uninstall, try_install_from_env
except Exception:  # pragma: no cover
    install = None  # type: ignore[assignment]
    uninstall = None  # type: ignore[assignment]
    try_install_from_env = None  # type: ignore[assignment]

__all__ = [
    "JsonlSink",
    "get_sink",
    "event_to_trace",
    "events_to_strategy_bundle",
    "install",
    "uninstall",
    "try_install_from_env",
    "SpeculativeTraceHook",
    "SpeculativeTraceWriter",
]
