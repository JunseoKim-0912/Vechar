import os
import json

from fastapi import HTTPException
from openai import APIStatusError, OpenAI
from sqlalchemy.orm import Session

from .llm_config import MAX_LLM_INPUT_BYTES, MAX_LLM_OUTPUT_TOKENS
from .llm_usage import check_capacity, record_failure, record_preflight_failure, record_response, reserve_usage


def _get_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file.")
    # Automatic retries could duplicate a billed generation behind one reservation.
    return OpenAI(max_retries=0)


def generate_text(
    db: Session,
    user_id: str,
    request_type: str,
    instructions: str,
    input_messages: list[dict],
    max_output_tokens: int,
) -> str:
    """Send the existing system prompt and conversation to the Responses API."""
    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError("OPENAI_MODEL is not set. Check your .env file.")
    if not 0 < max_output_tokens <= MAX_LLM_OUTPUT_TOKENS:
        raise HTTPException(status_code=413, detail={"code": "llm_output_too_large"})
    input_bytes = len(instructions.encode("utf-8")) + len(
        json.dumps(input_messages, ensure_ascii=False).encode("utf-8")
    )
    if input_bytes > MAX_LLM_INPUT_BYTES:
        raise HTTPException(status_code=413, detail={"code": "llm_input_too_large"})

    client = _get_client()
    bind = db.get_bind()
    check_capacity(bind, user_id, max_output_tokens)
    # Exact preflight count lets the reservation cover input plus the full output cap.
    try:
        input_tokens = client.responses.input_tokens.count(
            model=model, instructions=instructions, input=input_messages
        ).input_tokens
    except Exception as exc:
        record_preflight_failure(bind, user_id, request_type, model, type(exc).__name__)
        raise
    usage_id = reserve_usage(bind, user_id, request_type, model, input_tokens, max_output_tokens)

    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=input_messages,
            max_output_tokens=max_output_tokens,
        )
    except APIStatusError as exc:
        record_failure(bind, usage_id, type(exc).__name__, definitely_unbilled=400 <= exc.status_code < 500)
        raise
    except Exception as exc:
        record_failure(bind, usage_id, type(exc).__name__)
        raise

    record_response(bind, usage_id, response)
    if response.usage is None:
        raise RuntimeError("Model response contained no usage data.")
    if response.status != "completed" or not response.output_text:
        raise RuntimeError("Model response contained no completed text output.")
    return response.output_text
