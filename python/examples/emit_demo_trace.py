"""Emit a realistic demo trace without running vLLM — same schema the hook dumps."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from specprobe_hook import SpeculativeTraceHook

# Prose → code boundary collapse (matches the React demo scenario)
PROMPT = "Here is a minimal example:\n\n```python\n"
DRAFT = [
    ("The", 0.62, 0.04, 0.91, "boundary"),
    (" following", 0.48, 0.03, 0.55, "phrasing"),
    (" function", 0.41, 0.02, 0.40, "phrasing"),
    (" computes", 0.37, 0.02, 0.22, "phrasing"),
    ("\n", 0.18, 0.33, 0.10, "boundary"),
    ("def", 0.22, 0.71, 0.05, "syntax"),
    (" add", 0.20, 0.55, 0.05, "syntax"),
    ("(", 0.19, 0.48, 0.05, "syntax"),
    ("a", 0.17, 0.44, 0.05, None),
    (",", 0.16, 0.40, 0.05, None),
]


def main() -> None:
    out_dir = Path(__file__).resolve().parents[2] / "public" / "traces"
    hook = SpeculativeTraceHook(
        prompt=PROMPT,
        draft_strategy="ngram",
        engine_name="simulator",
        engine_version="0.1.0",
        target_model="demo-target",
        draft_model="demo-ngram",
        temperature=0.8,
        domain_hint="mixed",
        output_dir=out_dir,
        extra={"note": "Synthetic boundary-failure trace for the React lane."},
    )
    hook.begin_block()
    for token, p, q, roll, tag in DRAFT:
        hook.on_verify(
            token=token,
            draft_prob=p,
            target_prob=q,
            roll=roll,
            tag=tag,  # type: ignore[arg-type]
        )
    hook.end_block()
    path = hook.flush(out_dir / "mixed-boundary.ngram.json")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
