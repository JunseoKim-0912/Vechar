import os
import json
from collections.abc import Callable
from typing import Literal, TypeVar, cast

from fastapi import HTTPException
from openai import APIStatusError, OpenAI
from openai.lib._parsing._responses import type_to_text_format_param
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from .llm_config import (
    MAX_LLM_INPUT_BYTES, MAX_LLM_OUTPUT_TOKENS, MAX_TRAINING_INPUT_BYTES,
    MAX_TRAINING_INPUT_TOKENS, ModelTask, model_for_task,
)
from .llm_failures import FailureKind, LLMResponseError, LLMStructuredOutputError, classify_llm_failure
from .llm_operation import current_operation
from .llm_usage import check_capacity, record_failure, record_preflight_failure, record_response, reserve_usage

StructuredResult = TypeVar("StructuredResult", bound=BaseModel)


class LLMRefusalError(LLMResponseError):
    def __init__(self):
        super().__init__(FailureKind.REFUSAL, "Model refused structured output.")


def _get_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file.")
    # Automatic retries could duplicate a billed generation behind one reservation.
    return OpenAI(max_retries=0)


def input_size_bytes(instructions: str, input_messages: list[dict]) -> int:
    """Use the same serialized-input byte guard for chat budgeting and generation."""
    return len(instructions.encode("utf-8")) + len(
        json.dumps(input_messages, ensure_ascii=False).encode("utf-8")
    )


def _count_input_tokens(client: OpenAI, model: str, instructions: str,
                        input_messages: list[dict], text_config: dict | None = None) -> int:
    count_args = {"model": model, "instructions": instructions, "input": input_messages}
    if text_config:
        count_args["text"] = text_config
    return client.responses.input_tokens.count(**count_args).input_tokens


def make_chat_input_counter(db: Session, user_id: str, output_cap: int) -> Callable[[str, list[dict]], int]:
    """Return the gateway's exact CHAT token counter for layer selection.

    Capacity is checked before any provider count call. Generation still performs
    its own authoritative count and atomic reservation after context selection.
    """
    if not 0 < output_cap <= MAX_LLM_OUTPUT_TOKENS:
        raise HTTPException(status_code=413, detail={"code": "llm_output_too_large"})
    model = model_for_task("chat")
    bind = db.get_bind()
    check_capacity(bind, user_id, model, output_cap)
    client = _get_client()

    def count(instructions: str, input_messages: list[dict]) -> int:
        try:
            return _count_input_tokens(client, model, instructions, input_messages)
        except Exception as exc:
            record_preflight_failure(bind, user_id, "chat", model, type(exc).__name__)
            raise

    return count


def make_training_text_counter(
    db: Session, user_id: str, request_type: str, output_cap: int,
) -> Callable[[str], int]:
    """Exact source-text counts for direct/chunk decisions through the shared gateway.

    Generation still repeats the authoritative full-payload count and reservation.
    Reusing one client avoids creating a connection for every candidate boundary.
    """
    model = model_for_task("analysis")
    bind = db.get_bind()
    check_capacity(bind, user_id, model, output_cap)
    client = _get_client()

    def count(text: str) -> int:
        try:
            return _count_input_tokens(client, model, "", [{"role": "user", "content": text}])
        except Exception as exc:
            record_preflight_failure(bind, user_id, request_type, model, type(exc).__name__)
            raise

    return count


def generate_text(
    db: Session,
    user_id: str,
    request_type: str,
    instructions: str,
    input_messages: list[dict],
    max_output_tokens: int,
    *,
    task: ModelTask,
) -> str:
    """Send the existing system prompt and conversation to the Responses API."""
    return cast(str, _generate_response(
        db, user_id, request_type, instructions, input_messages, max_output_tokens, task=task
    ))


def generate_chat_turn(
    db: Session, user_id: str, request_type: str, instructions: str,
    input_messages: list[dict], max_output_tokens: int, *, task: ModelTask,
):
    """Meter visible reply and compact progression hints in one Responses call."""
    from .services.conversation_runtime import ChatTurnResult

    return generate_structured(
        db, user_id, request_type, instructions, input_messages, max_output_tokens,
        task=task, response_model=ChatTurnResult,
    )


def generate_structured(
    db: Session,
    user_id: str,
    request_type: str,
    instructions: str,
    input_messages: list[dict],
    max_output_tokens: int,
    *,
    task: ModelTask,
    response_model: type[StructuredResult],
    input_policy: Literal["standard", "training"] = "standard",
) -> StructuredResult:
    """Return a validated Pydantic result through the same metered gateway."""
    return cast(StructuredResult, _generate_response(
        db, user_id, request_type, instructions, input_messages, max_output_tokens,
        task=task, response_model=response_model, input_policy=input_policy,
    ))


def _require_output_fields(value: BaseModel) -> None:
    """Pydantic defaults are useful for stored data, but must not mask missing LLM fields."""
    missing = set(type(value).model_fields) - value.model_fields_set
    if missing:
        raise LLMStructuredOutputError(f"Structured response missing required fields: {', '.join(sorted(missing))}")
    for field_name in type(value).model_fields:
        field_value = getattr(value, field_name)
        if isinstance(field_value, BaseModel):
            _require_output_fields(field_value)
        elif isinstance(field_value, list):
            for item in field_value:
                if isinstance(item, BaseModel):
                    _require_output_fields(item)


def _parsed_output(response, response_model: type[StructuredResult]) -> StructuredResult:
    if response.status != "completed":
        reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
        kind = (FailureKind.OUTPUT_LIMIT if response.status == "incomplete" and reason == "max_output_tokens"
                else FailureKind.PROVIDER_TERMINAL)
        raise LLMResponseError(kind, f"Model response was not completed: {response.status}")
    for output in response.output or []:
        if getattr(output, "type", None) == "message":
            for item in output.content or []:
                if getattr(item, "type", None) == "refusal":
                    raise LLMRefusalError()
    if not response.output_text:
        raise LLMStructuredOutputError("Model response contained no structured output.")
    try:
        parsed = response_model.model_validate_json(response.output_text, extra="forbid")
    except ValidationError as exc:
        raise LLMStructuredOutputError("Model response failed structured schema validation.") from exc
    _require_output_fields(parsed)
    return parsed


def _generate_response(
    db: Session,
    user_id: str,
    request_type: str,
    instructions: str,
    input_messages: list[dict],
    max_output_tokens: int,
    *,
    task: ModelTask,
    response_model: type[BaseModel] | None = None,
    input_policy: Literal["standard", "training"] = "standard",
) -> str | BaseModel:
    model = model_for_task(task)
    if not 0 < max_output_tokens <= MAX_LLM_OUTPUT_TOKENS:
        raise HTTPException(status_code=413, detail={"code": "llm_output_too_large"})
    input_bytes = input_size_bytes(instructions, input_messages)
    input_byte_limit = MAX_TRAINING_INPUT_BYTES if input_policy == "training" else MAX_LLM_INPUT_BYTES
    if input_bytes > input_byte_limit:
        raise HTTPException(status_code=413, detail={"code": "llm_input_too_large"})

    # The pinned SDK converts the Pydantic model to its strict Responses text format.
    # Use create so provider usage remains available even if local validation fails.
    text_config = {"format": type_to_text_format_param(response_model)} if response_model else None
    client = _get_client()
    bind = db.get_bind()
    check_capacity(bind, user_id, model, max_output_tokens)
    # Exact preflight count lets the reservation cover input plus the full output cap.
    try:
        input_tokens = _count_input_tokens(client, model, instructions, input_messages, text_config)
    except Exception as exc:
        record_preflight_failure(bind, user_id, request_type, model, type(exc).__name__)
        raise
    if input_policy == "training" and input_tokens + max_output_tokens > MAX_TRAINING_INPUT_TOKENS:
        record_preflight_failure(bind, user_id, request_type, model, "TrainingContextTooLarge")
        raise HTTPException(status_code=413, detail={"code": "training_context_too_large"})
    operation = current_operation()
    reservation_kwargs = {"operation_key": operation.key(request_type)} if operation else {}
    usage_id = reserve_usage(bind, user_id, request_type, model, input_tokens,
                             max_output_tokens, **reservation_kwargs)

    try:
        create_args = {
            "model": model, "instructions": instructions, "input": input_messages,
            "max_output_tokens": max_output_tokens,
        }
        if text_config:
            create_args["text"] = text_config
        from .services.chat_latency import provider_started, provider_completed, stage
        provider_started()
        with stage("provider_request"):
            response = client.responses.create(**create_args)
        provider_completed()
    except APIStatusError as exc:
        record_failure(bind, usage_id, type(exc).__name__,
                       definitely_unbilled=400 <= exc.status_code < 500,
                       failure_class=classify_llm_failure(exc).kind.value)
        raise
    except Exception as exc:
        record_failure(bind, usage_id, type(exc).__name__,
                       failure_class=classify_llm_failure(exc).kind.value)
        raise

    try:
        if response.usage is None:
            raise RuntimeError("Model response contained no usage data.")
        if response_model:
            result = _parsed_output(response, response_model)
        else:
            if response.status != "completed":
                reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
                kind = (FailureKind.OUTPUT_LIMIT if response.status == "incomplete" and reason == "max_output_tokens"
                        else FailureKind.PROVIDER_TERMINAL)
                raise LLMResponseError(kind, "Model response contained no completed text output.")
            if not response.output_text:
                raise LLMResponseError(FailureKind.STRUCTURED_OUTPUT,
                                       "Model response contained no completed text output.")
            result = response.output_text
    except Exception as exc:
        record_response(bind, usage_id, response, output_error=type(exc).__name__,
                        failure_class=classify_llm_failure(exc).kind.value)
        raise
    record_response(bind, usage_id, response)
    return result
