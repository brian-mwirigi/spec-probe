"""Append-only JSONL sink for rejection-sampler events."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlSink:
    """Thread-safe append of one JSON object per line."""

    def __init__(self, path: str | Path | None = None) -> None:
        env = os.environ.get("SPECPROBE_JSONL", "traces/live.jsonl")
        self.path = Path(path or env)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            self.path.touch()

    def append(self, event: dict[str, Any]) -> None:
        payload = {"ts": event.get("ts") or _utcnow(), **event}
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()

    def read_from(self, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        """Read events starting at byte offset. Returns (events, new_offset)."""
        events: list[dict[str, Any]] = []
        with self._lock:
            size = self.path.stat().st_size
            if offset > size:
                offset = 0
            with self.path.open("r", encoding="utf-8") as f:
                f.seek(offset)
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):
                        # Incomplete line — wait for next poll
                        return events, pos
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                return events, f.tell()

    def iter_all(self) -> Iterator[dict[str, Any]]:
        events, _ = self.read_from(0)
        yield from events

    def clear(self) -> None:
        with self._lock:
            self.path.write_text("", encoding="utf-8")


_default_sink: Optional[JsonlSink] = None


def get_sink() -> JsonlSink:
    global _default_sink
    if _default_sink is None:
        _default_sink = JsonlSink()
    return _default_sink
