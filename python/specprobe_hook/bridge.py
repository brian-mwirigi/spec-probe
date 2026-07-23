"""
Tiny FastAPI bridge: Vite polls / WebSockets this to stream JSONL events.

  uvicorn specprobe_hook.bridge:app --reload --port 8787
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .event_to_trace import event_to_trace, events_to_strategy_bundle
from .jsonl_sink import JsonlSink

JSONL_PATH = Path(os.environ.get("SPECPROBE_JSONL", "traces/live.jsonl"))

app = FastAPI(title="spec-probe bridge", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sink = JsonlSink(JSONL_PATH)


class PollResponse(BaseModel):
    events: list[dict[str, Any]]
    offset: int
    path: str


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "jsonl": str(_sink.path), "size": _sink.path.stat().st_size}


@app.get("/events")
def poll_events(offset: int = 0) -> PollResponse:
    events, new_offset = _sink.read_from(offset)
    return PollResponse(events=events, offset=new_offset, path=str(_sink.path))


@app.get("/traces")
def traces(offset: int = 0) -> dict[str, Any]:
    events, new_offset = _sink.read_from(offset)
    return {
        "offset": new_offset,
        "traces": [event_to_trace(e) for e in events],
    }


@app.get("/bundle")
def bundle() -> dict[str, Any]:
    events, _ = _sink.read_from(0)
    return events_to_strategy_bundle(events)


@app.post("/clear")
def clear() -> dict[str, str]:
    _sink.clear()
    return {"status": "cleared"}


@app.websocket("/ws")
async def ws_events(websocket: WebSocket) -> None:
    await websocket.accept()
    offset = 0
    try:
        while True:
            events, offset = _sink.read_from(offset)
            for ev in events:
                await websocket.send_json({"type": "event", "data": ev, "trace": event_to_trace(ev)})
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return


def main(host: str = "127.0.0.1", port: int = 8787) -> None:
    import uvicorn

    uvicorn.run("specprobe_hook.bridge:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
