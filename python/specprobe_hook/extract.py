"""
Extract artifact-grade rejection-sampler fields from vLLM tensors.

vLLM terminology (RejectionSampler docstring):
  accepted  — passed the raw draft/target probability check
  recovered — sampled from residual r(x) ∝ max(p_target - p_draft, 0)
  bonus     — extra target sample when the entire draft is accepted
  output    — accepted + recovered + bonus

GPU → CPU is non-blocking: hot path only schedules D2H; a background worker
synchronizes, builds events, and appends JSONL so disk I/O never stalls CUDA.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from dataclasses import dataclass
from typing import Any, Optional, Sequence

PLACEHOLDER_TOKEN_ID = -1
logger = logging.getLogger("specprobe.extract")


def _as_int_list(arr) -> list[int]:
    return [int(x) for x in arr.flatten().tolist()]


def _as_float_list(arr) -> list[float]:
    return [float(x) for x in arr.flatten().tolist()]


def gather_token_probs(probs, token_ids):
    """probs: [N, V], token_ids: [N] -> [N] probability of each drafted token."""
    idx = token_ids.long().view(-1, 1)
    return probs.gather(-1, idx).squeeze(-1)


def split_by_cu(flat, cu_num_draft_tokens: Sequence[int]) -> list:
    out = []
    start = 0
    for end in cu_num_draft_tokens:
        end_i = int(end)
        out.append(flat[start:end_i])
        start = end_i
    return out


def classify_output_tokens(
    draft_token_ids: list[int],
    output_row: list[int],
) -> dict[str, Any]:
    """
    Split one rejection-sampler output row into vLLM's three token classes.

    output = accepted_tokens + recovered_token? + bonus_token?
    """
    accepted: list[int] = []
    recovered: Optional[int] = None
    bonus: Optional[int] = None
    rejected_at: Optional[int] = None
    mask: list[bool] = []

    for i, draft_id in enumerate(draft_token_ids):
        out_id = output_row[i] if i < len(output_row) else PLACEHOLDER_TOKEN_ID
        if rejected_at is not None:
            mask.append(False)
            continue
        if out_id == draft_id:
            accepted.append(draft_id)
            mask.append(True)
        else:
            mask.append(False)
            rejected_at = i
            if out_id != PLACEHOLDER_TOKEN_ID:
                recovered = out_id

    if rejected_at is None and len(draft_token_ids) < len(output_row):
        b = output_row[len(draft_token_ids)]
        if b != PLACEHOLDER_TOKEN_ID:
            bonus = b

    return {
        "accepted_tokens": accepted,
        "recovered_token": recovered,
        "bonus_token": bonus,
        "rejected_at": rejected_at,
        "acceptance_mask": mask,
    }


def acceptance_from_output(
    draft_token_ids: list[int],
    output_row: list[int],
) -> tuple[list[bool], Optional[int], Optional[int], Optional[int]]:
    """Back-compat wrapper used by unit tests."""
    c = classify_output_tokens(draft_token_ids, output_row)
    return c["acceptance_mask"], c["rejected_at"], c["recovered_token"], c["bonus_token"]


def _to_cpu_nonblocking(t):
    """Detach + async D2H when CUDA; plain detach otherwise."""
    if t is None:
        return None
    x = t.detach()
    if x.is_cuda:
        return x.to("cpu", non_blocking=True)
    return x.cpu()


@dataclass
class _PendingSnapshot:
    draft_token_ids_cpu: Any
    cu_num_draft_tokens_cpu: Any
    draft_probs_cpu: Any  # None or tensor
    target_logits_cpu: Any
    bonus_token_ids_cpu: Any
    output_token_ids_cpu: Any
    uniform_probs_cpu: Any
    num_draft_tokens: list[int]
    run_id: str
    draft_strategy: str
    request_ids: Optional[list[str]]
    prompt: Optional[str]
    temperature: Optional[float]
    step: int
    cuda_event: Any  # torch.cuda.Event | None


def build_events_from_numpy(
    *,
    draft_token_ids,
    cu_num_draft_tokens,
    draft_probs,
    target_logits,
    bonus_token_ids,
    output_token_ids,
    uniform_probs,
    num_draft_tokens: list[int],
    run_id: str,
    draft_strategy: str,
    request_ids: Optional[list[str]],
    prompt: Optional[str],
    temperature: Optional[float],
    step: int,
) -> list[dict[str, Any]]:
    """CPU-only event build (safe to run on the extract worker thread)."""
    import numpy as np
    import torch

    cu = [int(x) for x in cu_num_draft_tokens.numpy().tolist()]
    drafts_flat = draft_token_ids
    if not isinstance(drafts_flat, torch.Tensor):
        drafts_flat = torch.as_tensor(drafts_flat)

    target_logits_t = (
        target_logits if isinstance(target_logits, torch.Tensor) else torch.as_tensor(target_logits)
    )
    target_probs = torch.softmax(target_logits_t.float(), dim=-1)
    p_target_flat = gather_token_probs(target_probs, drafts_flat.long())

    if draft_probs is not None:
        dp = draft_probs if isinstance(draft_probs, torch.Tensor) else torch.as_tensor(draft_probs)
        p_draft_flat = gather_token_probs(dp.float(), drafts_flat.long())
        draft_probs_available = True
    else:
        p_draft_flat = torch.ones_like(p_target_flat)
        draft_probs_available = False

    rolls_flat = None
    if uniform_probs is not None:
        rolls_flat = (
            uniform_probs if isinstance(uniform_probs, torch.Tensor) else torch.as_tensor(uniform_probs)
        )

    draft_chunks = split_by_cu(drafts_flat, cu)
    p_draft_chunks = split_by_cu(p_draft_flat, cu)
    p_target_chunks = split_by_cu(p_target_flat, cu)
    roll_chunks = split_by_cu(rolls_flat, cu) if rolls_flat is not None else [None] * len(cu)

    bonus_arr = (
        bonus_token_ids.numpy()
        if hasattr(bonus_token_ids, "numpy")
        else np.asarray(bonus_token_ids)
    )
    bonus_list = [int(x) for x in bonus_arr.flatten().tolist()]
    out_arr = (
        output_token_ids.numpy()
        if hasattr(output_token_ids, "numpy")
        else np.asarray(output_token_ids)
    )

    events: list[dict[str, Any]] = []
    for req_idx, n_draft in enumerate(num_draft_tokens):
        draft_ids = _as_int_list(draft_chunks[req_idx])
        p_draft = _as_float_list(p_draft_chunks[req_idx])
        p_target = _as_float_list(p_target_chunks[req_idx])
        rolls = _as_float_list(roll_chunks[req_idx]) if roll_chunks[req_idx] is not None else None
        output_row = [int(x) for x in out_arr[req_idx].tolist()]

        classified = classify_output_tokens(draft_ids, output_row)
        rejected_at = classified["rejected_at"]
        bonus_token = classified["bonus_token"]
        if rejected_at is None and req_idx < len(bonus_list):
            bonus_token = bonus_list[req_idx]

        accept_probs = [min(1.0, pt / max(pd, 1e-12)) for pd, pt in zip(p_draft, p_target)]

        events.append(
            {
                "schema_version": "specprobe.event.v1",
                "run_id": run_id,
                "request_id": (request_ids[req_idx] if request_ids else f"req_{req_idx}"),
                "step": step,
                "req_index": req_idx,
                "draft_strategy": draft_strategy,
                "prompt": prompt,
                "temperature": temperature,
                "draft_probs_available": draft_probs_available,
                "num_draft_tokens": int(n_draft),
                "draft_tokens": draft_ids,
                "p_draft": p_draft,
                "p_target": p_target,
                "accept_prob": accept_probs,
                "rolls": rolls,
                "acceptance_mask": classified["acceptance_mask"],
                "rejected_at": rejected_at,
                # vLLM three-way split
                "accepted_tokens": classified["accepted_tokens"],
                "recovered_token": classified["recovered_token"],
                "bonus_token": bonus_token if rejected_at is None else None,
                "output_token_ids": output_row,
                # Explicit composition note for the lane / docs
                "output_composition": "accepted + recovered + bonus",
            }
        )
    return events


class AsyncExtractWorker:
    """Background thread: wait for D2H → build events → JSONL append."""

    def __init__(self) -> None:
        self._q: queue.Queue[Optional[_PendingSnapshot]] = queue.Queue()
        self._thread = threading.Thread(
            target=self._loop,
            name="specprobe-extract",
            daemon=True,
        )
        self._started = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if not self._started:
                self._thread.start()
                self._started = True

    def submit(self, snap: _PendingSnapshot) -> None:
        self.start()
        self._q.put(snap)

    def drain(self, timeout: float = 2.0) -> None:
        """Best-effort wait until the pending queue is empty."""
        import time

        deadline = time.time() + timeout
        while time.time() < deadline and not self._q.empty():
            time.sleep(0.05)

    def _loop(self) -> None:
        from .jsonl_sink import get_sink

        while True:
            snap = self._q.get()
            if snap is None:
                return
            try:
                if snap.cuda_event is not None:
                    snap.cuda_event.synchronize()

                events = build_events_from_numpy(
                    draft_token_ids=snap.draft_token_ids_cpu,
                    cu_num_draft_tokens=snap.cu_num_draft_tokens_cpu,
                    draft_probs=snap.draft_probs_cpu,
                    target_logits=snap.target_logits_cpu,
                    bonus_token_ids=snap.bonus_token_ids_cpu,
                    output_token_ids=snap.output_token_ids_cpu,
                    uniform_probs=snap.uniform_probs_cpu,
                    num_draft_tokens=snap.num_draft_tokens,
                    run_id=snap.run_id,
                    draft_strategy=snap.draft_strategy,
                    request_ids=snap.request_ids,
                    prompt=snap.prompt,
                    temperature=snap.temperature,
                    step=snap.step,
                )
                sink = get_sink()
                for ev in events:
                    sink.append(ev)
            except Exception:
                logger.exception("SpecProbe async extract failed")


_worker: Optional[AsyncExtractWorker] = None
_worker_lock = threading.Lock()


def get_worker() -> AsyncExtractWorker:
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = AsyncExtractWorker()
            _worker.start()
        return _worker


def schedule_extract(
    *,
    draft_token_ids,
    num_draft_tokens: list[int],
    cu_num_draft_tokens,
    draft_probs,
    target_logits,
    bonus_token_ids,
    output_token_ids,
    uniform_probs=None,
    run_id: str,
    draft_strategy: str = "unknown",
    request_ids: Optional[list[str]] = None,
    prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    step: int = 0,
) -> None:
    """
    Hot-path entry: non-blocking D2H + enqueue. Never writes JSON on this thread.

    Call this from RejectionSampler.forward right after rejection_sample returns
    (output tokens = accepted + recovered + bonus).
    """
    import torch

    if os.environ.get("SPECPROBE_SYNC_EXTRACT", "").lower() in {"1", "true", "yes"}:
        # Debug path: blocking extract (unit tests / tiny CPU runs)
        events = extract_step_events(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens,
            draft_probs=draft_probs,
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            output_token_ids=output_token_ids,
            uniform_probs=uniform_probs,
            run_id=run_id,
            draft_strategy=draft_strategy,
            request_ids=request_ids,
            prompt=prompt,
            temperature=temperature,
            step=step,
        )
        from .jsonl_sink import get_sink

        for ev in events:
            get_sink().append(ev)
        return

    cuda_event = None
    if draft_token_ids.is_cuda or (
        isinstance(target_logits, torch.Tensor) and target_logits.is_cuda
    ):
        cuda_event = torch.cuda.Event()
        # Issue async copies first, then record
        draft_cpu = _to_cpu_nonblocking(draft_token_ids)
        cu_cpu = _to_cpu_nonblocking(cu_num_draft_tokens)
        dp_cpu = _to_cpu_nonblocking(draft_probs)
        tl_cpu = _to_cpu_nonblocking(target_logits)
        bonus_cpu = _to_cpu_nonblocking(bonus_token_ids)
        out_cpu = _to_cpu_nonblocking(output_token_ids)
        uni_cpu = _to_cpu_nonblocking(uniform_probs)
        cuda_event.record()
    else:
        draft_cpu = _to_cpu_nonblocking(draft_token_ids)
        cu_cpu = _to_cpu_nonblocking(cu_num_draft_tokens)
        dp_cpu = _to_cpu_nonblocking(draft_probs)
        tl_cpu = _to_cpu_nonblocking(target_logits)
        bonus_cpu = _to_cpu_nonblocking(bonus_token_ids)
        out_cpu = _to_cpu_nonblocking(output_token_ids)
        uni_cpu = _to_cpu_nonblocking(uniform_probs)

    get_worker().submit(
        _PendingSnapshot(
            draft_token_ids_cpu=draft_cpu,
            cu_num_draft_tokens_cpu=cu_cpu,
            draft_probs_cpu=dp_cpu,
            target_logits_cpu=tl_cpu,
            bonus_token_ids_cpu=bonus_cpu,
            output_token_ids_cpu=out_cpu,
            uniform_probs_cpu=uni_cpu,
            num_draft_tokens=list(num_draft_tokens),
            run_id=run_id,
            draft_strategy=draft_strategy,
            request_ids=request_ids,
            prompt=prompt,
            temperature=temperature,
            step=step,
            cuda_event=cuda_event,
        )
    )


def extract_step_events(
    *,
    draft_token_ids,
    num_draft_tokens: list[int],
    cu_num_draft_tokens,
    draft_probs,
    target_logits,
    bonus_token_ids,
    output_token_ids,
    uniform_probs=None,
    run_id: str,
    draft_strategy: str = "unknown",
    request_ids: Optional[list[str]] = None,
    prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    step: int = 0,
) -> list[dict[str, Any]]:
    """Synchronous extract (tests / SPECPROBE_SYNC_EXTRACT=1)."""
    return build_events_from_numpy(
        draft_token_ids=_to_cpu_nonblocking(draft_token_ids),
        cu_num_draft_tokens=_to_cpu_nonblocking(cu_num_draft_tokens),
        draft_probs=_to_cpu_nonblocking(draft_probs),
        target_logits=_to_cpu_nonblocking(target_logits),
        bonus_token_ids=_to_cpu_nonblocking(bonus_token_ids),
        output_token_ids=_to_cpu_nonblocking(output_token_ids),
        uniform_probs=_to_cpu_nonblocking(uniform_probs),
        num_draft_tokens=num_draft_tokens,
        run_id=run_id,
        draft_strategy=draft_strategy,
        request_ids=request_ids,
        prompt=prompt,
        temperature=temperature,
        step=step,
    )
