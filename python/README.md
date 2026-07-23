# spec-probe ML boundary

Canonical injection point (vLLM v1):

```text
vllm/v1/sample/rejection_sampler.py
  class RejectionSampler(nn.Module)
      def forward(...)          ← primary monkey-patch
          rejection_sample(...) ← capture boundary
              output = accepted + recovered + bonus
```

Public `/metrics` only exposes `spec_decode_draft_acceptance_rate`. This hook
detaches `draft_probs` + target logits and writes JSONL without stalling CUDA.

## Token classes (vLLM terminology)

| Class | Meaning |
|-------|---------|
| **accepted** | Passed raw `min(1, p_target/p_draft)` check |
| **recovered** | Sampled from residual `max(p_target - p_draft, 0)` after reject |
| **bonus** | Extra target sample when the entire draft is accepted |

## Non-blocking extract

Hot path: `.to("cpu", non_blocking=True)` + `cuda.Event.record()` → enqueue.
Worker thread: `event.synchronize()` → numpy/lists → JSONL append.

Set `SPECPROBE_SYNC_EXTRACT=1` only for debugging.

## Install

```bash
cd python
pip install -e ".[bridge,dev]"
```

## Patch

```bash
export SPECPROBE_PATCH=1
export SPECPROBE_JSONL=traces/live.jsonl
export SPECPROBE_STRATEGY=eagle
python -c "from specprobe_hook import install; assert install()"
```

## Sweep configs (hardcoded constraints)

```bash
# N-gram — uses method + prompt_lookup_max
python -m specprobe_hook.sweep --model ... --ngram --ngram-max 4 --prompt-file ../prompts/boundary.txt

# EAGLE — pass draft model directly; do NOT set "method"
python -m specprobe_hook.sweep --model ... \
  --eagle-model yuhuili/EAGLE-LLaMA3-Instruct-8B \
  --prompt-file ../prompts/boundary.txt
```

Always forced:
- `pipeline_parallel_size=1` (spec decode + PP unsupported)
- `draft_tensor_parallel_size=1`

## Bridge

```bash
SPECPROBE_JSONL=../public/traces/live.jsonl uvicorn specprobe_hook.bridge:app --port 8787
```
