import os
from decimal import Decimal
from typing import Literal

ModelTask = Literal["analysis", "chat"]

MODEL_ENV_VARS = {
    "analysis": "OPENAI_ANALYSIS_MODEL",
    "chat": "OPENAI_CHAT_MODEL",
}

# Provisional server-side safety limits. Review before production rollout.
# The existing largest service request needs 8,000 output tokens for a full world profile.
MAX_LLM_OUTPUT_TOKENS = 8000
MAX_LLM_INPUT_BYTES = 60000
# JSON-serialized training input can expand control characters to six bytes each.
# Chat keeps its separate 60 KB guard.
MAX_TRAINING_INPUT_BYTES = 2_000_000
# Current ANALYSIS model has a 1,050,000-token context. Leave room for output
# and request overhead, using the provider's exact preflight token count.
MAX_TRAINING_INPUT_TOKENS = 1_000_000

# Source-text token budgets for synchronous training. Every generated request is
# still checked with the provider's exact full-payload count in the LLM gateway.
TRAINING_CHUNK_TARGET_TOKENS = 12_000
TRAINING_CHUNK_HARD_MAX_TOKENS = 18_000
TRAINING_CHUNK_OVERLAP_TOKENS = 750
DIRECT_TRAINING_SOURCE_TOKENS = TRAINING_CHUNK_HARD_MAX_TOKENS
# Count exceptionally large UTF-8 source text in pieces so even token-dense
# 300k-character input need not fit a single model context just for planning.
TRAINING_TOKEN_COUNT_SEGMENT_BYTES = 750_000
MAX_TIMELINE_SYNTHESIS_SUMMARY_CHARS = 500
MAX_EXISTING_WORLD_SYNTHESIS_FACTS = 100

# USD per 1M text tokens. Estimates only: no regional uplift or non-text/tool fees.
# Standard, short-context text prices: https://developers.openai.com/api/docs/models
# Keep historical prices for previously recorded usage; update when API pricing changes.
MODEL_PRICES_USD_PER_MILLION = {
    "gpt-6.1-sol": {
        "input": Decimal("2.00"),
        "cached_input": Decimal("0.10"),
        "output": Decimal("10.00"),
        # Provider bills the entire request at these rates above 272K input tokens.
        "long_context_threshold": 272_000,
        "long_context_input_multiplier": Decimal("2"),
        "long_context_output_multiplier": Decimal("1.5"),
    },
    "gpt-6-luna": {
        "input": Decimal("0.10"),
        "cached_input": Decimal("0.01"),
        "output": Decimal("0.50"),
    },
    "gpt-5.5": {
        "input": Decimal("5.00"),
        "cached_input": Decimal("0.50"),
        "output": Decimal("30.00"),
    },
    "gpt-5.5-2026-04-23": {
        "input": Decimal("5.00"),
        "cached_input": Decimal("0.50"),
        "output": Decimal("30.00"),
    },
}


def model_for_task(task: ModelTask) -> str:
    """Resolve a workload role without coupling services to provider model IDs."""
    try:
        env_var = MODEL_ENV_VARS[task]
    except KeyError as exc:
        raise ValueError(f"Unknown LLM task: {task}") from exc

    model = os.getenv(env_var, "").strip()
    if not model:
        raise RuntimeError(f"{env_var} is not set. Check your .env file.")
    if model not in MODEL_PRICES_USD_PER_MILLION:
        raise RuntimeError(f"No pricing configured for {env_var} model: {model}")
    return model
