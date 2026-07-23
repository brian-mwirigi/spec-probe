# spec-probe

[![GitHub](https://img.shields.io/badge/github-brian--mwirigi%2Fspec--probe-181717?logo=github)](https://github.com/brian-mwirigi/spec-probe)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brian-mwirigi/spec-probe/blob/main/notebooks/SpecProbe_Live_Fire.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE)
[![npm name](https://img.shields.io/badge/npm-spec--probe-CB3837?logo=npm)](https://www.npmjs.com/package/spec-probe)
[![PyPI name](https://img.shields.io/badge/PyPI-spec--probe-3775A9?logo=pypi&logoColor=white)](https://pypi.org/project/spec-probe/)

**Token-level diagnostics for vLLM speculative decoding.**

vLLM's `/metrics` only gives you a macro `spec_decode_draft_acceptance_rate`.  
**spec-probe** guts `RejectionSampler.forward`, dumps `p_draft` / `p_target`, and renders a verification lane so you can see *why* a draft token died — indent drift, domain boundaries, overconfidence — and compare EAGLE vs n-gram on the same prompt.

```text
vllm/v1/sample/rejection_sampler.py
  RejectionSampler.forward          ← primary monkey-patch
      rejection_sample(...)         ← capture boundary
          output = accepted + recovered + bonus
                │
                ▼  async D2H (non_blocking) → worker → JSONL
traces/*.jsonl
                │
                ▼
FastAPI :8787  (/events · /ws · /bundle)
                │
                ▼
Vite verification lane + side-by-side strategy sweep
```

## Why it exists

| Without spec-probe | With spec-probe |
|--------------------|-----------------|
| One acceptance % | Per-token `p` vs `q` |
| Blind temperature / draft tuning | Structural failure tags (indent, syntax, boundary) |
| Guess EAGLE vs n-gram | Side-by-side lanes on the same prompt |
| Observer bias risk | Async extract — JSONL I/O off the CUDA hot path |

## Quick start

### Local UI (demo JSONL, no GPU)

```bash
git clone https://github.com/brian-mwirigi/spec-probe.git
cd spec-probe
npm install
npm run dev
```

Open the app → **indent drift · ngram vs eagle**.  
Watch n-gram die at token `0` on whitespace while EAGLE clears the block.

### Colab (live GPU, no local VRAM)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brian-mwirigi/spec-probe/blob/main/notebooks/SpecProbe_Live_Fire.ipynb)

The notebook:

1. Applies `nest_asyncio` (Colab event-loop safe)
2. Patches `RejectionSampler.forward`
3. **Greedy A/B losslessness check** (`with patch == without patch`)
4. Emits `live.jsonl` and **downloads it** (best offline path)
5. Optional bridge + `serve_kernel_port_as_window(8787)`

Drop the downloaded JSONL into the local Vite UI.

### Hook a local vLLM process

```bash
cd python
pip install -e ".[bridge,vllm]"

export SPECPROBE_PATCH=1
export SPECPROBE_JSONL=traces/live.jsonl
export SPECPROBE_STRATEGY=eagle
python -c "from specprobe_hook import install; assert install()"
```

Bridge:

```bash
SPECPROBE_JSONL=../public/traces/live.jsonl uvicorn specprobe_hook.bridge:app --port 8787
# or: spec-probe-bridge
```

### Strategy sweep

```bash
# n-gram: {"method":"ngram","prompt_lookup_max":4,...}
# EAGLE:  {"model":"yuhuili/EAGLE-...",...}  — no "method" key
# always: pipeline_parallel_size=1, draft_tensor_parallel_size=1
python -m specprobe_hook.sweep \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --prompt-file prompts/boundary.txt \
  --out traces/sweep \
  --ngram --ngram-max 4 \
  --eagle-model yuhuili/EAGLE-LLaMA3-Instruct-8B
```

## What gets extracted

| Field | Meaning |
|-------|---------|
| `draft_tokens` | Proposed token IDs |
| `p_draft` / `p_target` | Mass on those tokens |
| `acceptance_mask` | Leviathan–Matias accept/reject |
| `accepted_tokens` | Passed the raw probability check |
| `recovered_token` | Residual sample after reject |
| `bonus_token` | Extra target sample on full accept |

Schemas: [`schemas/speculative-trace-v1.json`](schemas/speculative-trace-v1.json), [`schemas/specprobe-event-v1.json`](schemas/specprobe-event-v1.json).

## Repo layout

```text
spec-probe/
├── src/                  # Vite + React verification lane
├── python/specprobe_hook # vLLM patch, JSONL sink, FastAPI bridge, sweep CLI
├── notebooks/            # Colab live-fire notebook
├── schemas/              # trace + event JSON Schema
├── public/traces/        # demo JSON / JSONL
└── prompts/              # sweep prompts
```

## Package names

| Registry | Name | Notes |
|----------|------|--------|
| GitHub | [brian-mwirigi/spec-probe](https://github.com/brian-mwirigi/spec-probe) | source of truth |
| npm | `spec-probe` | frontend app |
| PyPI | `spec-probe` | install name; import `specprobe_hook` |

## License

MIT — see [LICENSE](LICENSE).
