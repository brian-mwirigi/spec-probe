"""Convert JSONL rejection events → SpecProbe trace.v1 for the React lane."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def _display(token_id: int, decoder: Optional[Any] = None) -> str:
    if decoder is not None:
        try:
            text = decoder.decode([token_id])
            return (
                text.replace(" ", "␠").replace("\t", "⇥").replace("\n", "↵") or f"id:{token_id}"
            )
        except Exception:
            pass
    return f"id:{token_id}"


def event_to_trace(
    event: dict[str, Any],
    *,
    tokenizer: Optional[Any] = None,
) -> dict[str, Any]:
    """Lift one specprobe.event.v1 record into a single-block trace.v1 document."""
    draft_tokens = event.get("draft_tokens") or []
    p_draft = event.get("p_draft") or []
    p_target = event.get("p_target") or []
    accept_prob = event.get("accept_prob") or []
    rolls = event.get("rolls")
    mask = event.get("acceptance_mask") or []
    rejected_at = event.get("rejected_at")
    strategy = event.get("draft_strategy") or "unknown"

    tokens = []
    failures = []
    for i, tok_id in enumerate(draft_tokens):
        pd = float(p_draft[i]) if i < len(p_draft) else 0.0
        pt = float(p_target[i]) if i < len(p_target) else 0.0
        ap = float(accept_prob[i]) if i < len(accept_prob) else min(1.0, pt / max(pd, 1e-12))
        accepted = bool(mask[i]) if i < len(mask) else False
        if rejected_at is not None and i > rejected_at:
            outcome = "unverified"
            reason = "Unverified — block aborted after earlier rejection."
        elif accepted:
            outcome = "accepted"
            reason = None
        else:
            outcome = "rejected"
            reason = f"Rejected: q/p={pt / max(pd, 1e-12):.3f}"
            failures.append(
                {
                    "kind": "overconfidence",
                    "label": "Draft overconfidence",
                    "detail": f"p_draft={pd:.4f} p_target={pt:.4f} on token_id={tok_id}",
                    "token_index": i,
                }
            )

        tokens.append(
            {
                "index": i,
                "token_id": tok_id,
                "token": _display(tok_id, tokenizer),
                "display": _display(tok_id, tokenizer),
                "draft_prob": pd,
                "target_prob": pt,
                "ratio": pt / max(pd, 1e-12),
                "accept_prob": ap,
                "roll": (float(rolls[i]) if rolls and i < len(rolls) else None),
                "outcome": outcome,
                "tag": None,
                "reason": reason,
            }
        )

    bonus = event.get("bonus_token")
    bonus_token = None
    if bonus is not None and rejected_at is None:
        bonus_token = {
            "token_id": bonus,
            "token": _display(int(bonus), tokenizer),
            "display": "⟨+1 target⟩",
            "target_prob": None,
        }
        tokens.append(
            {
                "index": len(tokens),
                "token_id": bonus,
                "token": _display(int(bonus), tokenizer),
                "display": "⟨+1 target⟩",
                "draft_prob": 0.0,
                "target_prob": 0.0,
                "ratio": None,
                "accept_prob": 1.0,
                "roll": 0.0,
                "outcome": "bonus",
                "tag": None,
                "reason": "Full draft accepted — target samples one extra token.",
            }
        )

    accepted_n = sum(1 for t in tokens if t["outcome"] == "accepted")
    verified = sum(1 for t in tokens if t["outcome"] in {"accepted", "rejected"})

    return {
        "schema_version": "specprobe.trace.v1",
        "trace_id": f"{event.get('run_id', 'run')}_{event.get('step', 0)}_{event.get('req_index', 0)}",
        "created_at": event.get("ts") or datetime.now(timezone.utc).isoformat(),
        "engine": {
            "name": "vllm",
            "version": None,
            "draft_strategy": strategy if strategy in {"eagle", "medusa", "ngram", "draft_model", "unknown"} else "unknown",
            "target_model": event.get("target_model"),
            "draft_model": event.get("draft_model"),
            "temperature": event.get("temperature"),
            "extra": {
                "run_id": event.get("run_id"),
                "step": event.get("step"),
                "draft_probs_available": event.get("draft_probs_available"),
                "recovered_token": event.get("recovered_token"),
                "raw_strategy": strategy,
            },
        },
        "request": {
            "prompt": event.get("prompt") or "",
            "request_id": event.get("request_id"),
            "domain_hint": event.get("domain_hint"),
        },
        "blocks": [
            {
                "block_index": 0,
                "draft_len": len(draft_tokens),
                "rejected_at": rejected_at,
                "bonus_token": bonus_token,
                "tokens": tokens,
                "structural_failures": failures,
            }
        ],
        "metrics": {
            "wall_time_ms": None,
            "acceptance_rate": (accepted_n / verified) if verified else None,
            "tokens_accepted": accepted_n,
            "tokens_drafted": len(draft_tokens),
        },
    }


def events_to_strategy_bundle(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Group events by draft_strategy for side-by-side React compare."""
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        key = str(ev.get("draft_strategy") or "unknown")
        by_strategy.setdefault(key, []).append(event_to_trace(ev))
    return {
        "schema_version": "specprobe.bundle.v1",
        "strategies": by_strategy,
        "count": sum(len(v) for v in by_strategy.values()),
    }
