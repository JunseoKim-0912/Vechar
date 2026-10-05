"""One retry policy for metered LLM failures and training workers."""

from dataclasses import dataclass
from enum import StrEnum

from fastapi import HTTPException
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError


class FailureKind(StrEnum):
    TRANSIENT = "transient"
    BUDGET = "budget"
    VALIDATION = "validation"
    OUTPUT_LIMIT = "output_limit"
    STRUCTURED_OUTPUT = "structured_output"
    REFUSAL = "refusal"
    PROVIDER_TERMINAL = "provider_terminal"
    DUPLICATE_OPERATION = "duplicate_operation"


@dataclass(frozen=True)
class FailureClassification:
    kind: FailureKind
    retryable: bool
    code: str


class LLMResponseError(RuntimeError):
    def __init__(self, kind: FailureKind, message: str):
        super().__init__(message)
        self.kind = kind


class LLMStructuredOutputError(ValueError):
    """A billed response could not satisfy the expected structured contract."""


class DuplicateLLMOperation(RuntimeError):
    """A logical attempt has already reserved usage; never start it twice."""


_BUDGET_CODES = {
    "daily_limit_reached", "monthly_limit_reached",
    "request_exceeds_remaining_daily_budget", "request_exceeds_remaining_monthly_budget",
}


def classify_llm_failure(exc: Exception) -> FailureClassification:
    """Unknown failures fail closed instead of repeating a potentially billed call."""
    if isinstance(exc, DuplicateLLMOperation):
        return FailureClassification(FailureKind.DUPLICATE_OPERATION, False, "duplicate_llm_operation")
    if isinstance(exc, LLMResponseError):
        code = {
            FailureKind.OUTPUT_LIMIT: "llm_output_limit",
            FailureKind.REFUSAL: "llm_refusal",
            FailureKind.STRUCTURED_OUTPUT: "llm_structured_output_invalid",
            FailureKind.PROVIDER_TERMINAL: "llm_response_failed",
        }.get(exc.kind, "llm_response_failed")
        return FailureClassification(exc.kind, False, code)
    if isinstance(exc, LLMStructuredOutputError):
        return FailureClassification(FailureKind.STRUCTURED_OUTPUT, False, "llm_structured_output_invalid")
    if isinstance(exc, HTTPException):
        code = exc.detail.get("code") if isinstance(exc.detail, dict) else None
        if code in _BUDGET_CODES:
            return FailureClassification(FailureKind.BUDGET, False, code)
        if exc.status_code == 429 or exc.status_code >= 500:
            return FailureClassification(FailureKind.TRANSIENT, True, "llm_provider_transient")
        return FailureClassification(FailureKind.VALIDATION, False, code or "training_input_rejected")
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError, TimeoutError, ConnectionError)):
        return FailureClassification(FailureKind.TRANSIENT, True, "llm_provider_transient")
    if isinstance(exc, APIStatusError):
        if exc.status_code == 429 or exc.status_code >= 500:
            return FailureClassification(FailureKind.TRANSIENT, True, "llm_provider_transient")
        return FailureClassification(FailureKind.VALIDATION, False, "llm_request_rejected")
    if isinstance(exc, (ValueError, TypeError)):
        return FailureClassification(FailureKind.VALIDATION, False, "training_input_rejected")
    return FailureClassification(FailureKind.PROVIDER_TERMINAL, False, "training_failed")
