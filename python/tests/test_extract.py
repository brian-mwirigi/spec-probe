from __future__ import annotations

from specprobe_hook.extract import (
    PLACEHOLDER_TOKEN_ID,
    acceptance_from_output,
    classify_output_tokens,
)
from specprobe_hook.sweep import (
    eagle_speculative_config,
    ngram_speculative_config,
    sanitize_vllm_args,
)


def test_full_accept_with_bonus():
    c = classify_output_tokens([10, 11, 12], [10, 11, 12, 99])
    assert c["accepted_tokens"] == [10, 11, 12]
    assert c["recovered_token"] is None
    assert c["bonus_token"] == 99
    assert c["rejected_at"] is None


def test_reject_with_recovered_and_placeholders():
    c = classify_output_tokens(
        [10, 11, 12, 13],
        [10, 11, 77, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
    )
    assert c["accepted_tokens"] == [10, 11]
    assert c["recovered_token"] == 77
    assert c["bonus_token"] is None
    assert c["rejected_at"] == 2
    assert c["acceptance_mask"] == [True, True, False, False]


def test_reject_at_zero():
    mask, rejected_at, recovered, bonus = acceptance_from_output(
        [5, 6],
        [42, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
    )
    assert mask == [False, False]
    assert rejected_at == 0
    assert recovered == 42
    assert bonus is None


def test_ngram_config_schema():
    cfg = ngram_speculative_config(num_speculative_tokens=5, prompt_lookup_max=4)
    assert cfg["method"] == "ngram"
    assert cfg["prompt_lookup_max"] == 4
    assert cfg["draft_tensor_parallel_size"] == 1


def test_eagle_config_has_no_method_key():
    cfg = eagle_speculative_config(
        eagle_model="yuhuili/EAGLE-LLaMA3-Instruct-8B",
        num_speculative_tokens=5,
    )
    assert "method" not in cfg
    assert cfg["model"] == "yuhuili/EAGLE-LLaMA3-Instruct-8B"
    assert cfg["draft_tensor_parallel_size"] == 1


def test_sanitize_forces_pp1():
    out = sanitize_vllm_args(["--tensor-parallel-size", "2", "--pipeline-parallel-size", "4"])
    assert "--pipeline-parallel-size" in out
    assert out[out.index("--pipeline-parallel-size") + 1] == "1"
    assert "4" not in out
