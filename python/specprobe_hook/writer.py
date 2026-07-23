"""Manual / simulator TraceWriter (no vLLM required)."""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

SCHEMA_VERSION = "specprobe.trace.v1"
Outcome = Literal["accepted", "rejected", "unverified", "bonus"]
Strategy = Literal["eagle", "medusa", "ngram", "draft_model", "unknown"]
Tag = Literal["indent", "syntax", "phrasing", "boundary", "overconfidence"]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def acceptance_probability(draft_prob: float, target_prob: float) -> float:
    if draft_prob <= 1e-12:
        return 1.0
    return min(1.0, target_prob / draft_prob)


def infer_tag(token: str) -> Optional[Tag]:
    if token in {"\n", "\t"} or token.isspace() or set(token) <= {"▁", "Ġ", " "}:
        return "indent"
    if token in set("(){}[]:;,."):
        return "syntax"
    if token in {"```", "def", "class", "import", "from", "return"}:
        return "boundary"
    return None


def display_token(token: str) -> str:
    return (
        token.replace(" ", "␠")
        .replace("\t", "⇥")
        .replace("\n", "↵")
        .replace("▁", "␠")
        .replace("Ġ", "␠")
    )


@dataclass
class TraceToken:
    index: int
    token: str
    draft_prob: float
    target_prob: float
    accept_prob: float
    outcome: Outcome
    token_id: Optional[int] = None
    display: Optional[str] = None
    ratio: Optional[float] = None
    roll: Optional[float] = None
    tag: Optional[Tag] = None
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["display"] is None:
            d["display"] = display_token(self.token)
        if d["ratio"] is None:
            d["ratio"] = self.target_prob / max(self.draft_prob, 1e-12)
        return d


@dataclass
class StructuralFailure:
    kind: Tag
    label: str
    detail: str
    token_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceBlock:
    block_index: int
    tokens: list[TraceToken] = field(default_factory=list)
    draft_len: int = 0
    rejected_at: Optional[int] = None
    bonus_token: Optional[dict[str, Any]] = None
    structural_failures: list[StructuralFailure] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "draft_len": self.draft_len or len(self.tokens),
            "rejected_at": self.rejected_at,
            "bonus_token": self.bonus_token,
            "tokens": [t.to_dict() for t in self.tokens],
            "structural_failures": [f.to_dict() for f in self.structural_failures],
        }


class SpeculativeTraceWriter:
    def __init__(
        self,
        *,
        prompt: str,
        draft_strategy: Strategy = "unknown",
        engine_name: str = "vllm",
        engine_version: Optional[str] = None,
        target_model: Optional[str] = None,
        draft_model: Optional[str] = None,
        temperature: float = 1.0,
        request_id: Optional[str] = None,
        domain_hint: Optional[str] = None,
        output_dir: str | Path = "traces",
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        self.prompt = prompt
        self.draft_strategy = draft_strategy
        self.engine_name = engine_name
        self.engine_version = engine_version
        self.target_model = target_model
        self.draft_model = draft_model
        self.temperature = temperature
        self.request_id = request_id
        self.domain_hint = domain_hint
        self.output_dir = Path(output_dir)
        self.extra = extra or {}
        self.trace_id = f"tr_{uuid.uuid4().hex[:12]}"
        self.created_at = _utcnow()
        self.blocks: list[TraceBlock] = []
        self._t0 = time.perf_counter()
        self._current: Optional[TraceBlock] = None

    def begin_block(self) -> TraceBlock:
        block = TraceBlock(block_index=len(self.blocks))
        self._current = block
        self.blocks.append(block)
        return block

    def record_token(
        self,
        *,
        token: str,
        draft_prob: float,
        target_prob: float,
        roll: Optional[float] = None,
        token_id: Optional[int] = None,
        display: Optional[str] = None,
        tag: Optional[Tag] = None,
        aborted: bool = False,
    ) -> TraceToken:
        if self._current is None:
            self.begin_block()
        assert self._current is not None

        block = self._current
        index = len(block.tokens)
        accept_prob = acceptance_probability(draft_prob, target_prob)
        ratio = target_prob / max(draft_prob, 1e-12)

        if aborted or block.rejected_at is not None:
            outcome: Outcome = "unverified"
            reason = "Unverified — block aborted after earlier rejection."
            resolved_tag = None
        else:
            accepted = True if roll is None else roll <= accept_prob
            if accepted:
                outcome = "accepted"
                reason = None
                resolved_tag = tag
            else:
                outcome = "rejected"
                block.rejected_at = index
                resolved_tag = tag or infer_tag(token)
                reason = (
                    f"Rejected: q/p={ratio:.3f} "
                    f"(p={draft_prob:.4f}, q={target_prob:.4f}, "
                    f"P(accept)={accept_prob:.4f}, roll={roll})"
                )
                block.structural_failures.append(
                    self._failure(
                        index,
                        display or display_token(token),
                        draft_prob,
                        target_prob,
                        resolved_tag,
                    )
                )

        tok = TraceToken(
            index=index,
            token=token,
            token_id=token_id,
            display=display or display_token(token),
            draft_prob=float(draft_prob),
            target_prob=float(target_prob),
            ratio=float(ratio),
            accept_prob=float(accept_prob),
            roll=None if roll is None else float(roll),
            outcome=outcome,
            tag=resolved_tag,
            reason=reason,
        )
        block.tokens.append(tok)
        block.draft_len = len([t for t in block.tokens if t.outcome != "bonus"])
        return tok

    def record_bonus(self, token: str, target_prob: float, token_id: Optional[int] = None) -> None:
        if self._current is None:
            self.begin_block()
        assert self._current is not None
        if self._current.rejected_at is not None:
            return
        self._current.bonus_token = {
            "token_id": token_id,
            "token": token,
            "display": "⟨+1 target⟩",
            "target_prob": float(target_prob),
        }
        self._current.tokens.append(
            TraceToken(
                index=len(self._current.tokens),
                token=token,
                token_id=token_id,
                display="⟨+1 target⟩",
                draft_prob=0.0,
                target_prob=float(target_prob),
                ratio=math.inf,
                accept_prob=1.0,
                roll=0.0,
                outcome="bonus",
                reason="Full draft accepted — target samples one extra token.",
            )
        )

    def end_block(self) -> None:
        self._current = None

    def _failure(
        self,
        index: int,
        display: str,
        draft_prob: float,
        target_prob: float,
        tag: Optional[Tag],
    ) -> StructuralFailure:
        ratio = target_prob / max(draft_prob, 1e-12)
        kind: Tag = tag or "overconfidence"
        labels = {
            "indent": "Indentation divergence",
            "syntax": "Syntax token mismatch",
            "phrasing": "Phrasing divergence",
            "boundary": "Domain boundary failure",
            "overconfidence": "Draft overconfidence",
        }
        return StructuralFailure(
            kind=kind,
            label=labels[kind],
            detail=(
                f"Draft p={draft_prob:.4f} vs target q={target_prob:.4f} "
                f"on {display!r} (q/p={ratio:.3f})."
            ),
            token_index=index,
        )

    def to_dict(self) -> dict[str, Any]:
        accepted = sum(1 for b in self.blocks for t in b.tokens if t.outcome == "accepted")
        drafted = sum(1 for b in self.blocks for t in b.tokens if t.outcome != "bonus")
        verified = sum(
            1
            for b in self.blocks
            for t in b.tokens
            if t.outcome in {"accepted", "rejected"}
        )
        wall_ms = (time.perf_counter() - self._t0) * 1000.0
        return {
            "schema_version": SCHEMA_VERSION,
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "engine": {
                "name": self.engine_name,
                "version": self.engine_version,
                "draft_strategy": self.draft_strategy,
                "target_model": self.target_model,
                "draft_model": self.draft_model,
                "temperature": self.temperature,
                "extra": self.extra,
            },
            "request": {
                "prompt": self.prompt,
                "request_id": self.request_id,
                "domain_hint": self.domain_hint,
            },
            "blocks": [b.to_dict() for b in self.blocks],
            "metrics": {
                "wall_time_ms": round(wall_ms, 3),
                "acceptance_rate": (accepted / verified) if verified else None,
                "tokens_accepted": accepted,
                "tokens_drafted": drafted,
            },
        }

    def dump(self, path: Optional[str | Path] = None) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = Path(path) if path else self.output_dir / f"{self.trace_id}.json"
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out


class SpeculativeTraceHook:
    def __init__(self, **kwargs: Any) -> None:
        self.writer = SpeculativeTraceWriter(**kwargs)

    def begin_block(self) -> None:
        self.writer.begin_block()

    def on_verify(
        self,
        *,
        token: str,
        draft_prob: float,
        target_prob: float,
        roll: Optional[float] = None,
        token_id: Optional[int] = None,
        display: Optional[str] = None,
        tag: Optional[Tag] = None,
    ) -> TraceToken:
        return self.writer.record_token(
            token=token,
            draft_prob=draft_prob,
            target_prob=target_prob,
            roll=roll,
            token_id=token_id,
            display=display,
            tag=tag,
        )

    def on_bonus(self, *, token: str, target_prob: float, token_id: Optional[int] = None) -> None:
        self.writer.record_bonus(token, target_prob, token_id=token_id)

    def end_block(self) -> None:
        self.writer.end_block()

    def flush(self, path: Optional[str | Path] = None) -> Path:
        return self.writer.dump(path)
