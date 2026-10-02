from decimal import Decimal

# Provisional server-side safety limits. Review before production rollout.
# The existing largest service request needs 8,000 output tokens for a full world profile.
MAX_LLM_OUTPUT_TOKENS = 8000
MAX_LLM_INPUT_BYTES = 60000

# USD per 1M text tokens. Estimates only: no regional uplift or non-text/tool fees.
# Update this mapping when model pricing changes or OPENAI_MODEL is changed.
MODEL_PRICES_USD_PER_MILLION = {
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
