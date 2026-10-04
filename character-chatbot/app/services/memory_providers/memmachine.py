"""MemMachine 0.3.9 adapter behind Vechar's provider-neutral memory contract.

This is a mock-tested integration seam, not permission to enable an external
provider in production. Durable deletion intent, retrieval tombstones, and
retryable cleanup are still required before real user data is stored.
"""

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version

import requests
from memmachine_client import MemMachineClient
from memmachine_common.api import MemoryType

from ...memory_config import MemoryConfig, MemoryConfigurationError
from ..memory_service import (
    MemoryAuthenticationError,
    MemoryCandidate,
    MemoryDeletionError,
    MemoryInvalidResult,
    MemoryPartialIngestionError,
    MemoryProviderError,
    MemoryRateLimitError,
    MemoryScopeError,
)


def _external_id(kind: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{kind} ID must be a nonempty string")
    return f"{kind}:{value}"


def _internal_id(kind: str, value: object) -> str | None:
    if value is None:
        return None
    prefix = f"{kind}:"
    if not isinstance(value, str) or not value.startswith(prefix) or not value[len(prefix):]:
        raise MemoryInvalidResult(f"MemMachine {kind} source ID is malformed")
    return value[len(prefix):]


def _field(item: object, name: str) -> object:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


COMPATIBLE_VERSION = "0.3.9"


def validate_sdk_versions() -> None:
    """Fail before project operations if the installed SDK differs from the PoC version."""
    for package in ("memmachine-client", "memmachine-common"):
        try:
            installed = version(package)
        except PackageNotFoundError as exc:
            raise MemoryConfigurationError(f"{package} is not installed") from exc
        if installed != COMPATIBLE_VERSION:
            raise MemoryConfigurationError(
                f"{package} {installed} is incompatible; expected {COMPATIBLE_VERSION}"
            )


class MemMachineAdapter:
    """Maps Vechar scopes to public MemMachine project/memory SDK methods."""

    def __init__(self, config: MemoryConfig, *, client: object | None = None):
        if config.provider != "memmachine":
            raise ValueError("MemMachineAdapter requires the memmachine provider")
        self.config = config
        self.client = client if client is not None else MemMachineClient(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=0,
        )
        # The provider is lazy at the Memory Service boundary. Once selected,
        # validate before the first chat retrieval or write, not after one.
        self._project_cache = self._load_and_validate_project()

    def _call(self, operation, *, missing_is_configuration: bool = False):
        try:
            return operation()
        except MemoryScopeError:
            raise
        except MemoryProviderError:
            raise
        except (TimeoutError, ConnectionError):
            raise
        except requests.Timeout as exc:
            raise TimeoutError("MemMachine request timed out") from exc
        except requests.ConnectionError as exc:
            raise ConnectionError("MemMachine connection failed") from exc
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if missing_is_configuration and status == 404:
                raise MemoryConfigurationError("MemMachine project or configuration API was not found") from exc
            if status in {401, 403}:
                raise MemoryAuthenticationError("MemMachine authentication failed") from exc
            if status == 429:
                raise MemoryRateLimitError("MemMachine rate limit reached") from exc
            raise MemoryProviderError(f"MemMachine HTTP error: {status or 'unknown'}") from exc
        except Exception as exc:
            raise MemoryProviderError("MemMachine request or response failed") from exc

    def _load_and_validate_project(self):
        validate_sdk_versions()

        health = self._call(lambda: self.client.request(
            "GET", f"{self.config.base_url.rstrip('/')}/api/v2/health",
            timeout=self.config.timeout_seconds,
        ))
        self._call(health.raise_for_status, missing_is_configuration=True)
        try:
            health_data = health.json()
        except ValueError as exc:
            raise MemoryConfigurationError("MemMachine health response is not JSON") from exc
        if not isinstance(health_data, Mapping):
            raise MemoryConfigurationError("MemMachine health response is malformed")
        if health_data.get("status") != "healthy":
            raise MemoryProviderError("MemMachine server is not healthy")
        if health_data.get("version") != COMPATIBLE_VERSION:
            raise MemoryConfigurationError(
                f"MemMachine server version must be {COMPATIBLE_VERSION}"
            )

        # Never create a project from a chat request. Provision it separately.
        project = self._call(lambda: self.client.get_project(
            org_id=self.config.org_id,
            project_id=self.config.project_id,
            timeout=self.config.timeout_seconds,
        ), missing_is_configuration=True)
        if _field(project, "org_id") != self.config.org_id or _field(project, "project_id") != self.config.project_id:
            raise MemoryConfigurationError("MemMachine project identity does not match configuration")
        memory = self._call(lambda: project.memory(metadata={}))
        episodic = self._call(lambda: memory.get_episodic_memory_config(
            timeout=self.config.timeout_seconds,
        ), missing_is_configuration=True)
        if _field(episodic, "enabled") is not True or _field(episodic, "long_term_memory_enabled") is not True:
            raise MemoryConfigurationError("MemMachine long-term episodic memory must be enabled")
        if _field(episodic, "short_term_memory_enabled") is not False:
            raise MemoryConfigurationError("MemMachine short-term episodic memory must be disabled")
        return project

    def _project(self):
        return self._project_cache

    def _memory(self, *, user_id: str, character_id: str, conversation_id: str | None = None):
        metadata = {
            "user_id": _external_id("user", user_id),
            "agent_id": _external_id("character", character_id),
        }
        if conversation_id is not None:
            metadata["session_id"] = _external_id("conversation", conversation_id)
        project = self._project()
        return self._call(lambda: project.memory(metadata=metadata))

    def retrieve(self, *, user_id: str, character_id: str, conversation_id: str,
                 current_message: str) -> tuple[MemoryCandidate, ...]:
        # Deliberately omit session_id from the search context: previous sessions
        # under this user/agent scope must remain searchable. SDK 0.3.9 searches
        # episodic + semantic internally; only long-term episodic hits are used.
        memory = self._memory(user_id=user_id, character_id=character_id)
        result = self._call(lambda: memory.search(
            current_message, limit=self.config.candidate_limit,
            expand_context=0, timeout=self.config.timeout_seconds,
        ))
        if _field(result, "status") != 0:
            raise MemoryInvalidResult("MemMachine search returned an unsuccessful status")
        content = _field(result, "content")
        if content is None:
            raise MemoryInvalidResult("MemMachine search has no content")
        episodic = _field(content, "episodic_memory")
        if episodic is None:
            if _field(content, "semantic_memory"):
                raise MemoryInvalidResult("MemMachine returned semantic memories without episodic results")
            return ()
        short_term = _field(episodic, "short_term_memory")
        summaries = _field(short_term, "episode_summary")
        if _field(short_term, "episodes") or (
            isinstance(summaries, (list, tuple)) and any(
                isinstance(summary, str) and summary.strip() for summary in summaries
            )
        ):
            # v0.3.9 prioritizes short-term duplicates over long-term hits.
            # A stale server-side project cache can still return these just
            # after reconfiguration; never mistake that for an empty memory.
            raise MemoryConfigurationError("MemMachine returned short-term context; restart the server after project configuration")
        long_term = _field(episodic, "long_term_memory")
        episodes = _field(long_term, "episodes")
        if not isinstance(episodes, (list, tuple)):
            raise MemoryInvalidResult("MemMachine long-term episodes are malformed")

        external_user = _external_id("user", user_id)
        external_agent = _external_id("character", character_id)
        candidates = []
        for episode in episodes[:self.config.candidate_limit]:
            metadata = _field(episode, "metadata")
            if not isinstance(metadata, Mapping):
                raise MemoryInvalidResult("MemMachine episode has no scope metadata")
            if metadata.get("user_id") != external_user or metadata.get("agent_id") != external_agent:
                raise MemoryScopeError("MemMachine episode crosses the requested user/character scope")
            memory_id = _field(episode, "uid")
            episode_content = _field(episode, "content")
            if not isinstance(memory_id, str) or not memory_id or not isinstance(episode_content, str) or not episode_content.strip():
                raise MemoryInvalidResult("MemMachine episode has no usable ID or content")
            candidates.append(MemoryCandidate(
                memory_id=memory_id,
                content=episode_content,
                user_id=user_id,
                character_id=character_id,
                source_conversation_id=_internal_id("conversation", metadata.get("session_id")),
                source_user_message_id=metadata.get("user_message_id"),
                source_assistant_message_id=metadata.get("assistant_message_id"),
                created_at=_field(episode, "created_at"),
                relevance_score=_field(episode, "score"),
                metadata={"provider_role": _field(episode, "producer_role")},
            ))
        return tuple(candidates)

    def record_completed_turn(self, *, user_id: str, character_id: str,
                              conversation_id: str, user_message_id: str,
                              assistant_message_id: str, user_message: str,
                              assistant_message: str) -> None:
        memory = self._memory(
            user_id=user_id, character_id=character_id, conversation_id=conversation_id
        )
        source = {
            "user_message_id": user_message_id,
            "assistant_message_id": assistant_message_id,
        }
        stored_count = 0
        for role, text, message_id in (
            ("user", user_message, user_message_id),
            ("assistant", assistant_message, assistant_message_id),
        ):
            try:
                result = self._call(lambda role=role, text=text, message_id=message_id: memory.add(
                    content=text,
                    role=role,
                    metadata={**source, "source_message_id": message_id},
                    memory_types=[MemoryType.Episodic],
                    timeout=self.config.timeout_seconds,
                ))
                if not isinstance(result, list) or len(result) != 1 or not isinstance(_field(result[0], "uid"), str) or not _field(result[0], "uid"):
                    raise MemoryInvalidResult("MemMachine add returned no confirmed episode ID")
                stored_count += 1
            except (MemoryProviderError, TimeoutError, ConnectionError) as exc:
                if stored_count:
                    raise MemoryPartialIngestionError("MemMachine completed turn was only partly stored") from exc
                raise

    def delete_conversation(self, *, user_id: str, character_id: str, conversation_id: str) -> None:
        raise MemoryDeletionError(
            "MemMachine 0.3.9 has no atomic source-conversation deletion API",
            retryable=False,
        )

    def delete_character(self, *, user_id: str, character_id: str) -> None:
        raise MemoryDeletionError(
            "MemMachine 0.3.9 has no atomic user/agent-scope deletion API",
            retryable=False,
        )

    def delete_user(self, *, user_id: str) -> None:
        raise MemoryDeletionError(
            "MemMachine 0.3.9 has no atomic user-scope deletion API",
            retryable=False,
        )
