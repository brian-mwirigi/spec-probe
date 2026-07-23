# spec-probe

Gut vLLM's rejection sampler. Render token-level `p_draft` / `p_target` — not the macro `/metrics` acceptance rate.

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

## Colab (no local GPU required)

Open [`notebooks/SpecProbe_Live_Fire.ipynb`](notebooks/SpecProbe_Live_Fire.ipynb) in Google Colab (GPU runtime).

The notebook:

1. Applies **`nest_asyncio`** so FastAPI/vLLM don't clash with Colab's existing event loop  
2. Installs `spec-probe` + vLLM, patches `RejectionSampler.forward`  
3. Runs a **greedy A/B losslessness check** (`with patch == without patch`)  
4. Emits **`live.jsonl`** and **downloads it** (offline path — drop into local Vite)  
5. Optionally starts the FastAPI bridge and exposes it with  
   `google.colab.output.serve_kernel_port_as_window(8787)` (no ngrok required)  
6. Optional n-gram vs EAGLE sweep → downloadable side-by-side JSONL  

**Recommended path:** download `live.jsonl` → local `npm run dev` → drop the file in the UI.

## UI (local)

```bash
npm install
npm run dev
```

Load **indent drift · ngram vs eagle**, or drop a Colab-exported JSONL.

## Hook + bridge (local GPU)

```bash
cd python
pip install -e ".[bridge,dev]"

SPECPROBE_JSONL=../public/traces/live.jsonl uvicorn specprobe_hook.bridge:app --port 8787
# or: spec-probe-bridge
```

```bash
export SPECPROBE_PATCH=1
export SPECPROBE_JSONL=traces/live.jsonl
export SPECPROBE_STRATEGY=eagle
python -c "from specprobe_hook import install; assert install()"
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

## Package names

| Registry | Name | Status |
|----------|------|--------|
| npm | `spec-probe` | available |
| PyPI | `spec-probe` | available |

Python import module remains `specprobe_hook` (underscores). Wire schema ids stay `specprobe.trace.v1` / `specprobe.event.v1`.

Schemas: `schemas/speculative-trace-v1.json`, `schemas/specprobe-event-v1.json`.
