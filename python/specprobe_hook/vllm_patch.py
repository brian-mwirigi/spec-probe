"""
Monkey-patch vLLM v1 RejectionSampler.forward.

Canonical target:
  vllm/v1/sample/rejection_sampler.py  →  class RejectionSampler(nn.Module)

vLLM tracks three token classes inside this module:
  accepted  — passed raw p_target/p_draft check
  recovered — residual sample after rejection
  bonus     — target sample when the full draft is accepted
  output tokens = accepted + recovered + bonus

We patch `RejectionSampler.forward`, capture draft_probs + target logits at the
rejection_sample boundary, then schedule non-blocking GPU→CPU extract → JSONL.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from typing import Any, Optional

logger = logging.getLogger("specprobe.vllm_patch")

_installed = False
_install_lock = threading.Lock()
_step_counter = 0
_step_lock = threading.Lock()
_run_id = os.environ.get("SPECPROBE_RUN_ID") or f"run_{uuid.uuid4().hex[:10]}"


def _next_step() -> int:
    global _step_counter
    with _step_lock:
        step = _step_counter
        _step_counter += 1
        return step


def _strategy() -> str:
    return os.environ.get("SPECPROBE_STRATEGY", "unknown")


def _disabled() -> bool:
    return os.environ.get("SPECPROBE_DISABLE", "").lower() in {"1", "true", "yes"}


def _temperature(sampling_metadata: Any) -> Optional[float]:
    try:
        if getattr(sampling_metadata, "all_greedy", False):
            return 0.0
        t = getattr(sampling_metadata, "temperature", None)
        if t is None:
            return None
        if hasattr(t, "detach"):
            return float(t.detach().float().mean().item())
        return float(t)
    except Exception:
        return None


def _schedule_from_rejection_args(
    *,
    draft_token_ids,
    num_draft_tokens,
    cu_num_draft_tokens,
    draft_probs,
    target_logits,
    bonus_token_ids,
    output_token_ids,
    sampling_metadata,
) -> None:
    """Fire async extract — must not block the CUDA hot path."""
    from .extract import schedule_extract

    schedule_extract(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=list(num_draft_tokens),
        cu_num_draft_tokens=cu_num_draft_tokens,
        draft_probs=draft_probs,
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        output_token_ids=output_token_ids,
        uniform_probs=None,
        run_id=_run_id,
        draft_strategy=_strategy(),
        prompt=os.environ.get("SPECPROBE_PROMPT"),
        temperature=_temperature(sampling_metadata),
        step=_next_step(),
    )


def _wrap_rejection_sample(original):
    """Capture tensors at the accepted+recovered+bonus composition boundary."""

    def hooked(
        draft_token_ids,
        num_draft_tokens,
        max_spec_len,
        cu_num_draft_tokens,
        draft_probs,
        target_logits,
        bonus_token_ids,
        sampling_metadata,
        synthetic_mode: bool = False,
        synthetic_conditional_rates=None,
        use_fp64_gumbel: bool = False,
    ):
        output_token_ids = original(
            draft_token_ids,
            num_draft_tokens,
            max_spec_len,
            cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            synthetic_mode=synthetic_mode,
            synthetic_conditional_rates=synthetic_conditional_rates,
            use_fp64_gumbel=use_fp64_gumbel,
        )

        if not _disabled():
            try:
                # Right after composition: output = accepted + recovered + bonus.
                # Detach draft_probs + target_logits via async D2H (not sync .cpu()).
                _schedule_from_rejection_args(
                    draft_token_ids=draft_token_ids,
                    num_draft_tokens=num_draft_tokens,
                    cu_num_draft_tokens=cu_num_draft_tokens,
                    draft_probs=draft_probs,
                    target_logits=target_logits,
                    bonus_token_ids=bonus_token_ids,
                    output_token_ids=output_token_ids,
                    sampling_metadata=sampling_metadata,
                )
            except Exception:
                logger.exception("SpecProbe schedule_extract failed (generation continues)")

        return output_token_ids

    hooked._specprobe_patched = True  # type: ignore[attr-defined]
    hooked._specprobe_original = original  # type: ignore[attr-defined]
    return hooked


def _wrap_forward(original_forward, rs_module):
    """
    Primary patch target: RejectionSampler.forward (nn.Module).

    Ensures the module-level rejection_sample hook is installed for this call
    and documents the three token classes in logs once.
    """

    def forward(self, metadata, draft_probs, logits, sampling_metadata):
        """
        SpecProbe-wrapped forward.

        Intercepts draft_probs + logits-derived target distributions at the
        rejection_sample boundary inside this forward. Token classes:
          accepted / recovered / bonus  →  output_token_ids
        """
        if _disabled():
            return original_forward(self, metadata, draft_probs, logits, sampling_metadata)

        # Guarantee module global is our hooked rejection_sample (forward looks
        # it up from vllm.v1.sample.rejection_sampler at call time).
        current = rs_module.rejection_sample
        if not getattr(current, "_specprobe_patched", False):
            rs_module.rejection_sample = _wrap_rejection_sample(current)

        return original_forward(self, metadata, draft_probs, logits, sampling_metadata)

    forward._specprobe_patched = True  # type: ignore[attr-defined]
    forward._specprobe_original = original_forward  # type: ignore[attr-defined]
    return forward


def install(force: bool = False) -> bool:
    """Monkey-patch RejectionSampler.forward (+ rejection_sample capture)."""
    global _installed
    with _install_lock:
        if _installed and not force:
            return True
        try:
            import vllm.v1.sample.rejection_sampler as rs
        except ImportError as exc:
            logger.error(
                "vLLM not importable — install vLLM before patching (%s)",
                exc,
            )
            return False

        if not hasattr(rs, "RejectionSampler"):
            logger.error("vllm.v1.sample.rejection_sampler lacks RejectionSampler")
            return False

        cls = rs.RejectionSampler
        if getattr(cls.forward, "_specprobe_patched", False) and not force:
            _installed = True
            return True

        # 1) Primary: patch the nn.Module forward
        cls.forward = _wrap_forward(cls.forward, rs)

        # 2) Capture boundary: patch rejection_sample (composition site)
        if not getattr(rs.rejection_sample, "_specprobe_patched", False) or force:
            original = getattr(rs.rejection_sample, "_specprobe_original", rs.rejection_sample)
            if getattr(rs.rejection_sample, "_specprobe_patched", False):
                original = rs.rejection_sample._specprobe_original  # type: ignore[attr-defined]
            rs.rejection_sample = _wrap_rejection_sample(original)

        _installed = True
        logger.info(
            "SpecProbe patch installed on RejectionSampler.forward "
            "(accepted/recovered/bonus) run_id=%s jsonl=%s",
            _run_id,
            os.environ.get("SPECPROBE_JSONL", "traces/live.jsonl"),
        )
        return True


def uninstall() -> None:
    global _installed
    with _install_lock:
        try:
            import vllm.v1.sample.rejection_sampler as rs

            cls = rs.RejectionSampler
            fwd = cls.forward
            if getattr(fwd, "_specprobe_original", None) is not None:
                cls.forward = fwd._specprobe_original  # type: ignore[attr-defined]

            original = getattr(rs.rejection_sample, "_specprobe_original", None)
            if original is not None:
                rs.rejection_sample = original
        except Exception:
            logger.exception("SpecProbe uninstall failed")
        _installed = False


def try_install_from_env() -> Optional[bool]:
    if os.environ.get("SPECPROBE_PATCH", "").lower() in {"1", "true", "yes"}:
        return install()
    return None
