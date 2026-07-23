"""
Strategy sweep CLI — restart vLLM once per draft method, same prompt.

Speculative-config schemas (vLLM v1):
  N-gram:  {"method": "ngram", "prompt_lookup_max": N, ...}
  EAGLE:   {"model": "yuhuili/EAGLE-...", ...}   # no "method" key

Hard constraints (vLLM will crash otherwise):
  - speculative decoding + pipeline parallelism = unsupported → PP=1
  - draft models: draft_tensor_parallel_size = 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib import request


# vLLM speculative decoding does not support pipeline parallel.
DEFAULT_PP_SIZE = 1
# Draft heads must stay on a single TP rank.
DEFAULT_DRAFT_TP_SIZE = 1


def _http_json(url: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_healthy(base: str, timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with request.urlopen(base + "/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(1.5)
    raise RuntimeError(f"vLLM did not become healthy at {base}")


def _write_sitecustomize(site_dir: Path) -> Path:
    site_dir.mkdir(parents=True, exist_ok=True)
    path = site_dir / "sitecustomize.py"
    path.write_text(
        "import os\n"
        "if os.environ.get('SPECPROBE_PATCH', '').lower() in {'1', 'true', 'yes'}:\n"
        "    try:\n"
        "        from specprobe_hook.vllm_patch import install\n"
        "        install()\n"
        "    except Exception as exc:\n"
        "        print(f'[specprobe] patch failed: {exc}', file=__import__('sys').stderr)\n",
        encoding="utf-8",
    )
    return path


def ngram_speculative_config(
    *,
    num_speculative_tokens: int,
    prompt_lookup_max: int,
) -> dict[str, Any]:
    return {
        "method": "ngram",
        "num_speculative_tokens": num_speculative_tokens,
        "prompt_lookup_max": prompt_lookup_max,
        "draft_tensor_parallel_size": DEFAULT_DRAFT_TP_SIZE,
    }


def eagle_speculative_config(
    *,
    eagle_model: str,
    num_speculative_tokens: int,
) -> dict[str, Any]:
    """
    EAGLE draft: pass the draft model directly — do NOT set "method".

    Example: {"model": "yuhuili/EAGLE-LLaMA3-Instruct-8B", "num_speculative_tokens": 5, ...}
    """
    return {
        "model": eagle_model,
        "num_speculative_tokens": num_speculative_tokens,
        "draft_tensor_parallel_size": DEFAULT_DRAFT_TP_SIZE,
    }


def sanitize_vllm_args(extra: list[str]) -> list[str]:
    """
    Strip/override flags that break speculative decoding.
    Forces --pipeline-parallel-size 1.
    """
    out: list[str] = []
    skip_next = False
    i = 0
    while i < len(extra):
        if skip_next:
            skip_next = False
            i += 1
            continue
        arg = extra[i]
        if arg in {"--pipeline-parallel-size", "-pp"}:
            # drop user value; we hardcode PP=1 below
            skip_next = True
            i += 1
            continue
        if arg.startswith("--pipeline-parallel-size="):
            i += 1
            continue
        out.append(arg)
        i += 1

    out.extend(["--pipeline-parallel-size", str(DEFAULT_PP_SIZE)])
    return out


def run_one(
    *,
    model: str,
    prompt: str,
    strategy: str,
    speculative_config: dict[str, Any],
    out_dir: Path,
    port: int,
    max_tokens: int,
    temperature: float,
    python_exe: str,
    extra_vllm_args: list[str],
) -> Path:
    run_id = f"{strategy}_{uuid.uuid4().hex[:8]}"
    jsonl = out_dir / f"{run_id}.jsonl"
    site_dir = out_dir / f".site_{run_id}"
    _write_sitecustomize(site_dir)

    env = os.environ.copy()
    env["SPECPROBE_PATCH"] = "1"
    env["SPECPROBE_JSONL"] = str(jsonl)
    env["SPECPROBE_STRATEGY"] = strategy
    env["SPECPROBE_RUN_ID"] = run_id
    env["SPECPROBE_PROMPT"] = prompt[:2000]
    env["PYTHONPATH"] = str(site_dir) + os.pathsep + env.get("PYTHONPATH", "")

    safe_extra = sanitize_vllm_args(extra_vllm_args)

    # Always persist draft_tensor_parallel_size inside speculative_config
    speculative_config = {
        **speculative_config,
        "draft_tensor_parallel_size": DEFAULT_DRAFT_TP_SIZE,
    }

    cmd = [
        python_exe,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--port",
        str(port),
        "--speculative-config",
        json.dumps(speculative_config),
        *safe_extra,
    ]

    print(f"[sweep] starting {strategy}")
    print(f"[sweep] speculative_config={json.dumps(speculative_config)}")
    print(f"[sweep] constraints: pipeline_parallel_size={DEFAULT_PP_SIZE}, draft_tp={DEFAULT_DRAFT_TP_SIZE}")
    print(f"[sweep] jsonl -> {jsonl}")
    proc = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)

    base = f"http://127.0.0.1:{port}"
    try:
        _wait_healthy(base)
        _http_json(
            base + "/v1/completions",
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        # Let async extract worker flush D2H + JSONL
        time.sleep(1.0)
    finally:
        if proc.poll() is None:
            if os.name == "nt":
                proc.terminate()
            else:
                proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(site_dir, ignore_errors=True)

    if not jsonl.exists() or jsonl.stat().st_size == 0:
        print(f"[sweep] WARNING: no events written for {strategy}", file=sys.stderr)
    return jsonl


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="spec-probe strategy sweep (ngram vs EAGLE)")
    p.add_argument("--model", required=True, help="Target HF model id/path")
    p.add_argument("--prompt", default=None, help="Prompt string")
    p.add_argument("--prompt-file", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("traces/sweep"))
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--ngram", action="store_true", help="Run n-gram lookahead")
    p.add_argument("--ngram-max", type=int, default=4, help="prompt_lookup_max")
    p.add_argument(
        "--eagle-model",
        default=None,
        help='EAGLE draft HF id (e.g. yuhuili/EAGLE-LLaMA3-Instruct-8B). Config uses "model", not "method".',
    )
    p.add_argument("--num-speculative-tokens", type=int, default=5)
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args after -- passed to vLLM (PP>1 will be overridden to 1)",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    prompt = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    if not prompt:
        print("Provide --prompt or --prompt-file", file=sys.stderr)
        return 2

    if not args.ngram and not args.eagle_model:
        print("Enable at least one of --ngram or --eagle-model", file=sys.stderr)
        return 2

    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "prompt": prompt,
        "constraints": {
            "pipeline_parallel_size": DEFAULT_PP_SIZE,
            "draft_tensor_parallel_size": DEFAULT_DRAFT_TP_SIZE,
        },
        "runs": [],
    }

    if args.ngram:
        cfg = ngram_speculative_config(
            num_speculative_tokens=args.num_speculative_tokens,
            prompt_lookup_max=args.ngram_max,
        )
        path = run_one(
            model=args.model,
            prompt=prompt,
            strategy="ngram",
            speculative_config=cfg,
            out_dir=args.out,
            port=args.port,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            python_exe=args.python,
            extra_vllm_args=extra,
        )
        manifest["runs"].append({"strategy": "ngram", "jsonl": str(path), "config": cfg})

    if args.eagle_model:
        cfg = eagle_speculative_config(
            eagle_model=args.eagle_model,
            num_speculative_tokens=args.num_speculative_tokens,
        )
        path = run_one(
            model=args.model,
            prompt=prompt,
            strategy="eagle",
            speculative_config=cfg,
            out_dir=args.out,
            port=args.port,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            python_exe=args.python,
            extra_vllm_args=extra,
        )
        manifest["runs"].append({"strategy": "eagle", "jsonl": str(path), "config": cfg})

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[sweep] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
