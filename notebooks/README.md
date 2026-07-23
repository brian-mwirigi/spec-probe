# Notebooks

| File | Purpose |
|------|---------|
| [SpecProbe_Live_Fire.ipynb](SpecProbe_Live_Fire.ipynb) | Colab GPU: patch → losslessness check → `live.jsonl` download → optional bridge |

Product / package name: **spec-probe** (npm + PyPI).

## Open in Colab

1. Upload this repo (or set `REPO_URL` to your fork inside the notebook).  
2. Runtime → GPU.  
3. Run all.  
4. Download `live.jsonl` and drop it into the local Vite UI.

Colab-specific: `nest_asyncio.apply()` before uvicorn;  
`output.serve_kernel_port_as_window(8787)` for native port forward.
