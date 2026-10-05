"""Configuration for the optional external memory adapter.

An external provider is opt-in. Configuration does not create a client or
contact a server, and must not silently fall back to no-op on invalid input.
"""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit


class MemoryConfigurationError(ValueError):
    """Memory provider selection or required settings are invalid."""


@dataclass(frozen=True)
class MemoryConfig:
    provider: str = "noop"
    base_url: str | None = None
    org_id: str | None = None
    project_id: str | None = None
    api_key: str | None = None
    timeout_seconds: int = 3
    candidate_limit: int = 10
    max_ingestion_attempts: int = 5
    max_deletion_attempts: int = 10


def _positive_int(env: Mapping[str, str], name: str, default: int, maximum: int) -> int:
    raw = env.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise MemoryConfigurationError(f"{name} must be a positive integer") from exc
    if value <= 0 or value > maximum:
        raise MemoryConfigurationError(f"{name} must be between 1 and {maximum}")
    return value


def load_memory_config(env: Mapping[str, str] | None = None) -> MemoryConfig:
    env = os.environ if env is None else env
    provider = (env.get("MEMORY_PROVIDER") or "noop").strip().lower()
    if provider not in {"noop", "memmachine"}:
        raise MemoryConfigurationError(f"Unsupported MEMORY_PROVIDER: {provider}")
    if provider == "noop":
        return MemoryConfig()

    base_url = (env.get("MEMMACHINE_BASE_URL") or "").strip()
    org_id = (env.get("MEMMACHINE_ORG_ID") or "").strip()
    project_id = (env.get("MEMMACHINE_PROJECT_ID") or "").strip()
    missing = [name for name, value in (
        ("MEMMACHINE_BASE_URL", base_url),
        ("MEMMACHINE_ORG_ID", org_id),
        ("MEMMACHINE_PROJECT_ID", project_id),
    ) if not value]
    if missing:
        raise MemoryConfigurationError("Missing MemMachine configuration: " + ", ".join(missing))
    for name, value in (("MEMMACHINE_ORG_ID", org_id), ("MEMMACHINE_PROJECT_ID", project_id)):
        if not re.fullmatch(r"[\w:-]+", value):
            raise MemoryConfigurationError(f"{name} contains unsupported ID characters")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise MemoryConfigurationError("MEMMACHINE_BASE_URL must be an http(s) origin without credentials or query")
    return MemoryConfig(
        provider=provider,
        base_url=base_url,
        org_id=org_id,
        project_id=project_id,
        api_key=(env.get("MEMMACHINE_API_KEY") or "").strip() or None,
        timeout_seconds=_positive_int(env, "MEMMACHINE_TIMEOUT_SECONDS", 3, 30),
        candidate_limit=_positive_int(env, "MEMMACHINE_CANDIDATE_LIMIT", 10, 50),
        max_ingestion_attempts=_positive_int(env, "MAX_MEMORY_INGESTION_ATTEMPTS", 5, 50),
        max_deletion_attempts=_positive_int(env, "MAX_MEMORY_DELETION_ATTEMPTS", 10, 100),
    )
