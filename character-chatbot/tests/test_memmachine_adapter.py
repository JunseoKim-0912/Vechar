"""No server, network, or OpenAI calls: fake the public MemMachine SDK objects."""

import os
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from memmachine_client import MemMachineClient
from memmachine_client.project import Project
from memmachine_common.api import MemoryType
from memmachine_common.api.spec import SearchResult

from app.memory_config import MemoryConfigurationError, load_memory_config
from app.services import memory_service
from app.services.chat_context_budget import select_chat_context
from app.services.memory_providers.memmachine import MemMachineAdapter


def config(**overrides):
    env = {
        "MEMORY_PROVIDER": "memmachine",
        "MEMMACHINE_BASE_URL": "http://127.0.0.1:8080",
        "MEMMACHINE_ORG_ID": "vechar-org",
        "MEMMACHINE_PROJECT_ID": "vechar-test",
    }
    env.update(overrides)
    return load_memory_config(env)


def episode(index, **metadata_overrides):
    metadata = {
        "user_id": "user:u1", "agent_id": "character:c1",
        "session_id": "conversation:old-session",
        "user_message_id": "um1", "assistant_message_id": "am1",
    }
    metadata.update(metadata_overrides)
    return {
        "uid": f"episode-{index}", "content": f"remembered fact {index}",
        "producer_id": "user:u1", "producer_role": "user",
        "metadata": metadata,
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "score": 0.8,
    }


def search_result(episodes=(), *, semantic=None):
    return SearchResult.model_validate({
        "status": 0,
        "content": {
            "episodic_memory": {
                "long_term_memory": {"episodes": list(episodes)},
                "short_term_memory": {"episodes": [], "episode_summary": []},
            },
            "semantic_memory": semantic,
        },
    })


class FakeMemory:
    def __init__(self, result=None):
        self.result = result if result is not None else search_result()
        self.episodic_config = SimpleNamespace(
            enabled=True, long_term_memory_enabled=True, short_term_memory_enabled=False,
        )
        self.search_calls = []
        self.add_calls = []
        self.error = None
        self.fail_second_add = False
        self.stored = []
        self.scope_metadata = {}

    def search(self, *args, **kwargs):
        self.search_calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.result

    def add(self, **kwargs):
        self.add_calls.append(kwargs)
        if self.fail_second_add and len(self.add_calls) == 2:
            raise requests.Timeout("fake timeout")
        self.stored.append(SimpleNamespace(uid=f"new-{len(self.add_calls)}",
                                           metadata={**self.scope_metadata, **kwargs["metadata"]}))
        return [SimpleNamespace(uid=f"new-{len(self.add_calls)}")]

    def get_context(self):
        return {"metadata": self.scope_metadata}

    def list(self, *, filter_dict=None, **kwargs):
        items = [item for item in self.stored if all(
            item.metadata.get(key) == value for key, value in self.scope_metadata.items()
        )]
        if filter_dict:
            items = [item for item in items if all(
                item.metadata.get(key.removeprefix("metadata.")) == value
                for key, value in filter_dict.items()
            )]
        return SimpleNamespace(status=0, content=SimpleNamespace(episodic_memory=items))

    def delete_episodic(self, *, episodic_id="", episodic_ids=None, **kwargs):
        ids = {episodic_id} | set(episodic_ids or [])
        self.stored = [item for item in self.stored if item.uid not in ids]
        return True

    def get_episodic_memory_config(self, *, timeout):
        return self.episodic_config


class FakeProject:
    def __init__(self, memory):
        self.org_id = "vechar-org"
        self.project_id = "vechar-test"
        self.memory_object = memory
        self.contexts = []

    def memory(self, *, metadata):
        if metadata:
            self.contexts.append(metadata)
            self.memory_object.scope_metadata = metadata
        return self.memory_object


class FakeClient:
    def __init__(self, project):
        self.project = project
        self.project_calls = []
        self.server_version = "0.3.9"
        self.health_error = None

    def request(self, method, url, *, timeout):
        if self.health_error:
            raise self.health_error
        if method != "GET" or not url.endswith("/api/v2/health"):
            raise AssertionError("Unexpected SDK request")
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"status": "healthy", "version": self.server_version},
        )

    def get_project(self, **kwargs):
        self.project_calls.append(kwargs)
        return self.project


class MemMachineAdapterTests(unittest.TestCase):
    def adapter(self, result=None):
        memory = FakeMemory(result)
        project = FakeProject(memory)
        client = FakeClient(project)
        return MemMachineAdapter(config(), client=client), memory, project, client

    def gateway(self, adapter):
        return patch.multiple(
            memory_service,
            PROVIDER_NAME="memmachine",
            _memmachine_adapter=lambda: adapter,
        )

    def test_configuration_is_opt_in_and_invalid_settings_fail(self):
        self.assertEqual(load_memory_config({}).provider, "noop")
        self.assertEqual(config().provider, "memmachine")
        with self.assertRaises(MemoryConfigurationError):
            load_memory_config({"MEMORY_PROVIDER": "typo"})
        with self.assertRaisesRegex(MemoryConfigurationError, "MEMMACHINE_PROJECT_ID"):
            load_memory_config({
                "MEMORY_PROVIDER": "memmachine",
                "MEMMACHINE_BASE_URL": "http://127.0.0.1:8080",
                "MEMMACHINE_ORG_ID": "vechar-org",
            })
        with self.assertRaises(MemoryConfigurationError):
            config(MEMMACHINE_TIMEOUT_SECONDS="0")
        with self.assertRaises(MemoryConfigurationError):
            config(MEMMACHINE_BASE_URL="not-a-url")
        with self.assertRaises(MemoryConfigurationError):
            config(MEMMACHINE_PROJECT_ID="bad project")

    def test_environment_selects_memmachine_without_contacting_server(self):
        env = os.environ.copy()
        env.update({
            "MEMORY_PROVIDER": "memmachine",
            "MEMMACHINE_BASE_URL": "http://127.0.0.1:8080",
            "MEMMACHINE_ORG_ID": "vechar-org",
            "MEMMACHINE_PROJECT_ID": "vechar-test",
        })
        result = subprocess.run(
            [sys.executable, "-c", "from app.services import memory_service; print(memory_service.PROVIDER_NAME)"],
            env=env, capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout.strip(), "memmachine")

    def test_noop_does_not_construct_or_import_adapter(self):
        with patch.object(memory_service, "_memmachine_adapter", side_effect=AssertionError("SDK used")):
            result = memory_service.retrieve_for_turn(
                user_id="u1", character_id="c1", conversation_id="s1", current_message="hello",
            )
        self.assertEqual((result.provider, result.candidate_count, result.success), ("noop", 0, True))

    def test_invalid_project_settings_fail_before_search_or_ingestion(self):
        for settings in (
            {"enabled": False},
            {"long_term_memory_enabled": False},
            {"short_term_memory_enabled": True},
        ):
            with self.subTest(settings=settings):
                memory = FakeMemory()
                for name, value in settings.items():
                    setattr(memory.episodic_config, name, value)
                client = FakeClient(FakeProject(memory))
                with self.assertRaises(MemoryConfigurationError):
                    MemMachineAdapter(config(), client=client)
                self.assertEqual((memory.search_calls, memory.add_calls), ([], []))

    def test_project_identity_and_server_version_must_match(self):
        memory = FakeMemory()
        project = FakeProject(memory)
        project.org_id = "another-org"
        with self.assertRaisesRegex(MemoryConfigurationError, "identity"):
            MemMachineAdapter(config(), client=FakeClient(project))

        client = FakeClient(FakeProject(memory))
        client.server_version = "0.3.8"
        with self.assertRaisesRegex(MemoryConfigurationError, "server version"):
            MemMachineAdapter(config(), client=client)
        self.assertEqual(client.project_calls, [])

        client = FakeClient(FakeProject(memory))
        with patch("app.services.memory_providers.memmachine.version", return_value="0.3.8"):
            with self.assertRaisesRegex(MemoryConfigurationError, "memmachine-client"):
                MemMachineAdapter(config(), client=client)
        self.assertEqual(client.project_calls, [])

    def test_missing_project_is_configuration_error_not_empty_memory(self):
        client = FakeClient(FakeProject(FakeMemory()))
        response = requests.Response()
        response.status_code = 404
        client.get_project = Mock(side_effect=requests.HTTPError("not found", response=response))
        with self.assertRaises(MemoryConfigurationError):
            MemMachineAdapter(config(), client=client)

    def test_provider_outage_during_validation_still_gracefully_degrades(self):
        client = FakeClient(FakeProject(FakeMemory()))
        client.health_error = requests.ConnectionError("offline")
        with patch.object(memory_service, "PROVIDER_NAME", "memmachine"), patch.object(
            memory_service, "_memmachine_adapter",
            side_effect=lambda: MemMachineAdapter(config(), client=client),
        ), self.assertLogs(memory_service.logger, level="WARNING"):
            result = memory_service.retrieve_for_turn(
                user_id="u1", character_id="c1", conversation_id="s1", current_message="hello",
            )
        self.assertEqual((result.success, result.candidate_count, result.error_type),
                         (False, 0, "ConnectionError"))

    def test_project_configuration_mismatch_degrades_retrieval(self):
        memory = FakeMemory()
        memory.episodic_config.short_term_memory_enabled = True
        client = FakeClient(FakeProject(memory))
        with patch.object(memory_service, "PROVIDER_NAME", "memmachine"), patch.object(
            memory_service, "_memmachine_adapter",
            side_effect=lambda: MemMachineAdapter(config(), client=client),
        ):
            with self.assertLogs(memory_service.logger, level="WARNING"):
                result = memory_service.retrieve_for_turn(
                    user_id="u1", character_id="c1", conversation_id="s1", current_message="hello",
                )
        self.assertEqual((result.success, result.error_type), (False, "MemoryConfigurationError"))

    def test_cross_session_scope_and_long_term_episode_mapping(self):
        result = search_result([episode(1)], semantic=[])
        adapter, memory, project, client = self.adapter(result)
        with self.gateway(adapter):
            response = memory_service.retrieve_for_turn(
                user_id="u1", character_id="c1", conversation_id="new-session",
                current_message="What did we discuss?",
            )
        self.assertTrue(response.success)
        self.assertEqual(response.provider, "memmachine")
        self.assertEqual(client.project_calls, [{
            "org_id": "vechar-org", "project_id": "vechar-test", "timeout": 3,
        }])
        self.assertEqual(project.contexts, [{"user_id": "user:u1", "agent_id": "character:c1"}])
        self.assertEqual(memory.search_calls, [
            (("What did we discuss?",), {"limit": 10, "expand_context": 0, "timeout": 3})
        ])
        candidate = response.candidates[0]
        self.assertEqual((candidate.memory_id, candidate.source_conversation_id),
                         ("episode-1", "old-session"))
        self.assertEqual((candidate.source_user_message_id, candidate.source_assistant_message_id),
                         ("um1", "am1"))
        self.assertEqual((candidate.user_id, candidate.character_id, candidate.relevance_score),
                         ("u1", "c1", 0.8))
        self.assertIsNotNone(candidate.created_at)

    def test_completed_turn_cross_session_contract_and_scope_isolation(self):
        memory = FakeMemory()
        empty_memory = FakeMemory()

        class ScopedProject(FakeProject):
            def memory(self, *, metadata):
                self.contexts.append(metadata)
                if not metadata or (metadata.get("user_id") == "user:u1" and
                                    metadata.get("agent_id") == "character:c1"):
                    memory.scope_metadata = metadata
                    return memory
                empty_memory.scope_metadata = metadata
                return empty_memory

        project = ScopedProject(memory)
        adapter = MemMachineAdapter(config(), client=FakeClient(project))
        adapter.record_completed_turn(
            user_id="u1", character_id="c1", conversation_id="poc-a",
            user_message_id="um1", assistant_message_id="am1",
            user_message="blue key at clock tower", assistant_message="we kept the key",
        )
        self.assertEqual(project.contexts[1]["session_id"], "conversation:poc-a")
        memory.result = search_result([episode(1, session_id="conversation:poc-a")])
        from_b = adapter.retrieve(
            user_id="u1", character_id="c1", conversation_id="poc-b",
            current_message="What did we find?",
        )
        other_character = adapter.retrieve(
            user_id="u1", character_id="c2", conversation_id="poc-c",
            current_message="What did we find?",
        )
        other_user = adapter.retrieve(
            user_id="u2", character_id="c1", conversation_id="poc-c",
            current_message="What did we find?",
        )
        self.assertEqual((len(from_b), from_b[0].source_conversation_id), (1, "poc-a"))
        self.assertEqual((other_character, other_user), ((), ()))
        self.assertNotIn("session_id", project.contexts[2])

    def test_semantic_results_are_ignored_and_stale_short_term_is_rejected(self):
        memory = FakeMemory(SimpleNamespace(
            status=0,
            content=SimpleNamespace(
                episodic_memory=SimpleNamespace(
                    long_term_memory=SimpleNamespace(episodes=[episode(1)]),
                    short_term_memory=SimpleNamespace(episodes=[]),
                ),
                semantic_memory=[{"content": "must not modify canonical profile"}],
            ),
        ))
        adapter = MemMachineAdapter(config(), client=FakeClient(FakeProject(memory)))
        candidates = adapter.retrieve(
            user_id="u1", character_id="c1", conversation_id="poc-b", current_message="hello",
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].content, "remembered fact 1")

        memory.result.content.episodic_memory.short_term_memory.episodes = [episode(1)]
        with self.assertRaisesRegex(MemoryConfigurationError, "short-term context"):
            adapter.retrieve(
                user_id="u1", character_id="c1", conversation_id="poc-b", current_message="hello",
            )
        memory.result.content.episodic_memory.short_term_memory.episodes = []
        memory.result.content.episodic_memory.short_term_memory.episode_summary = ["stale summary"]
        with self.assertRaisesRegex(MemoryConfigurationError, "short-term context"):
            adapter.retrieve(
                user_id="u1", character_id="c1", conversation_id="poc-b", current_message="hello",
            )

    def test_installed_sdk_uses_public_filter_and_bounded_search_without_network(self):
        client = MemMachineClient(base_url="http://127.0.0.1:8080", timeout=3, max_retries=0)
        project = Project(client=client, org_id="vechar-org", project_id="vechar-test")
        client.get_project = Mock(return_value=project)
        def sdk_response(method, url, **kwargs):
            response = Mock()
            if url.endswith("/api/v2/health"):
                response.json.return_value = {"status": "healthy", "version": "0.3.9"}
            elif url.endswith("/api/v2/memory/episodic/config/get"):
                response.json.return_value = {
                    "enabled": True, "long_term_memory_enabled": True,
                    "short_term_memory_enabled": False,
                }
            elif url.endswith("/api/v2/memories/search"):
                response.json.return_value = search_result([episode(1)]).model_dump(mode="json")
            else:
                raise AssertionError(f"Unexpected SDK request: {url}")
            return response
        client.request = Mock(side_effect=sdk_response)
        try:
            adapter = MemMachineAdapter(config(), client=client)
            candidates = adapter.retrieve(
                user_id="u1", character_id="c1", conversation_id="new-session",
                current_message="What did we discuss?",
            )
            self.assertEqual(len(candidates), 1)
            search_call = next(call for call in client.request.call_args_list
                               if call.args[1].endswith("/api/v2/memories/search"))
            request_payload = search_call.kwargs["json"]
            self.assertEqual(request_payload["top_k"], 10)
            self.assertEqual(request_payload["types"], ["episodic", "semantic"])
            self.assertIn("metadata.user_id='user:u1'", request_payload["filter"])
            self.assertIn("metadata.agent_id='character:c1'", request_payload["filter"])
            self.assertNotIn("session_id", request_payload["filter"])
            self.assertEqual(client.request.call_args.kwargs["timeout"], 3)
        finally:
            client.close()

    def test_empty_search_and_hundred_results_stay_bounded(self):
        adapter, memory, _, _ = self.adapter()
        with self.gateway(adapter):
            empty = memory_service.retrieve_for_turn(
                user_id="u1", character_id="c1", conversation_id="s1", current_message="hello",
            )
        self.assertEqual((empty.success, empty.candidate_count), (True, 0))

        memory.result = search_result([episode(i) for i in range(100)])
        with self.gateway(adapter):
            response = memory_service.retrieve_for_turn(
                user_id="u1", character_id="c1", conversation_id="s2", current_message="hello",
            )
        self.assertEqual(response.candidate_count, 10)
        budget = select_chat_context(
            "instructions", [], "hello",
            memories=[item.content for item in response.candidates],
            count_input_tokens=lambda instructions, messages: len(instructions) + sum(
                len(item["content"]) + 2 for item in messages
            ),
        )
        self.assertLessEqual(budget.metadata.memory_count, 5)
        self.assertLessEqual(budget.metadata.memory_tokens, 1500)

    def test_scope_mismatch_is_not_a_graceful_provider_fallback(self):
        for mismatch in ({"agent_id": "character:c2"}, {"user_id": "user:u2"}):
            with self.subTest(mismatch=mismatch):
                adapter, _, _, _ = self.adapter(search_result([episode(1, **mismatch)]))
                with self.gateway(adapter):
                    with self.assertRaises(memory_service.MemoryScopeError):
                        memory_service.retrieve_for_turn(
                            user_id="u1", character_id="c1", conversation_id="s1",
                            current_message="hello",
                        )

    def test_malformed_result_is_graceful_fallback(self):
        adapter, memory, _, _ = self.adapter()
        memory.result = SimpleNamespace(status=0, content=SimpleNamespace(
            episodic_memory=SimpleNamespace(long_term_memory=SimpleNamespace(episodes=None))
        ))
        with self.gateway(adapter), self.assertLogs(memory_service.logger, level="WARNING"):
            response = memory_service.retrieve_for_turn(
                user_id="u1", character_id="c1", conversation_id="s1", current_message="hello",
            )
        self.assertEqual((response.success, response.error_type), (False, "MemoryInvalidResult"))

    def test_completed_turn_preserves_roles_source_ids_and_episodic_type(self):
        adapter, memory, project, _ = self.adapter()
        with self.gateway(adapter):
            success = memory_service.record_completed_turn(
                user_id="u1", character_id="c1", conversation_id="s1",
                user_message_id="um1", assistant_message_id="am1",
                user_message="hello", assistant_message="hi",
            )
        self.assertTrue(success)
        self.assertEqual(project.contexts[-1], {
            "user_id": "user:u1", "agent_id": "character:c1", "session_id": "conversation:s1",
        })
        self.assertEqual([call["role"] for call in memory.add_calls], ["user", "assistant"])
        self.assertEqual([call["content"] for call in memory.add_calls], ["hello", "hi"])
        self.assertEqual([call["metadata"]["source_message_id"] for call in memory.add_calls],
                         ["um1", "am1"])
        for call in memory.add_calls:
            self.assertEqual(call["memory_types"], [MemoryType.Episodic])
            self.assertEqual(call["metadata"]["user_message_id"], "um1")
            self.assertEqual(call["metadata"]["assistant_message_id"], "am1")

    def test_source_metadata_avoids_duplicate_provider_add_on_retry(self):
        adapter, memory, _, _ = self.adapter()
        values = dict(user_id="u1", character_id="c1", conversation_id="s1",
                      user_message_id="um1", assistant_message_id="am1",
                      user_message="hello", assistant_message="hi")
        adapter.record_completed_turn(**values)
        adapter.record_completed_turn(**values)
        self.assertEqual(len(memory.add_calls), 2)
        self.assertEqual(len(memory.stored), 2)

    def test_partial_first_write_is_resumed_without_readding_it(self):
        adapter, memory, _, _ = self.adapter()
        memory.fail_second_add = True
        values = dict(user_id="u1", character_id="c1", conversation_id="s1",
                      user_message_id="um1", assistant_message_id="am1",
                      user_message="hello", assistant_message="hi")
        with self.assertRaises(memory_service.MemoryPartialIngestionError):
            adapter.record_completed_turn(**values)
        memory.fail_second_add = False
        adapter.record_completed_turn(**values)
        self.assertEqual(len(memory.stored), 2)
        self.assertEqual([call["role"] for call in memory.add_calls], ["user", "assistant", "assistant"])

    def test_character_delete_only_removes_target_scope_and_repeat_is_safe(self):
        adapter, memory, _, _ = self.adapter()
        adapter.record_completed_turn(user_id="u1", character_id="c1", conversation_id="s1",
                                      user_message_id="um1", assistant_message_id="am1",
                                      user_message="hello", assistant_message="hi")
        adapter.record_completed_turn(user_id="u1", character_id="c2", conversation_id="s2",
                                      user_message_id="um2", assistant_message_id="am2",
                                      user_message="other", assistant_message="other reply")
        adapter.delete_character(user_id="u1", character_id="c1")
        adapter.delete_character(user_id="u1", character_id="c1")
        self.assertEqual(len(memory.stored), 2)
        self.assertTrue(all(item.metadata["agent_id"] == "character:c2" for item in memory.stored))

    def test_installed_sdk_serializes_completed_turn_provenance_without_network(self):
        client = MemMachineClient(base_url="http://127.0.0.1:8080", timeout=3, max_retries=0)
        project = Project(client=client, org_id="vechar-org", project_id="vechar-test")
        client.get_project = Mock(return_value=project)
        def sdk_response(method, url, **kwargs):
            response = Mock()
            if url.endswith("/api/v2/health"):
                response.json.return_value = {"status": "healthy", "version": "0.3.9"}
            elif url.endswith("/api/v2/memory/episodic/config/get"):
                response.json.return_value = {
                    "enabled": True, "long_term_memory_enabled": True,
                    "short_term_memory_enabled": False,
                }
            elif url.endswith("/api/v2/memories"):
                response.json.return_value = {"results": [{"uid": "new-episode"}]}
            elif url.endswith("/api/v2/memories/list"):
                response.json.return_value = {"status": 0, "content": {"episodic_memory": []}}
            else:
                raise AssertionError(f"Unexpected SDK request: {url}")
            return response
        client.request = Mock(side_effect=sdk_response)
        try:
            adapter = MemMachineAdapter(config(), client=client)
            adapter.record_completed_turn(
                user_id="u1", character_id="c1", conversation_id="s1",
                user_message_id="um1", assistant_message_id="am1",
                user_message="hello", assistant_message="hi",
            )
            payloads = [call.kwargs["json"] for call in client.request.call_args_list
                        if call.args[1].endswith("/api/v2/memories")]
            self.assertEqual(len(payloads), 2)
            self.assertEqual([payload["messages"][0]["role"] for payload in payloads],
                             ["user", "assistant"])
            for payload in payloads:
                self.assertEqual(payload["types"], ["episodic"])
                metadata = payload["messages"][0]["metadata"]
                self.assertEqual(metadata["user_id"], "user:u1")
                self.assertEqual(metadata["agent_id"], "character:c1")
                self.assertEqual(metadata["session_id"], "conversation:s1")
                self.assertEqual(metadata["user_message_id"], "um1")
                self.assertEqual(metadata["assistant_message_id"], "am1")
        finally:
            client.close()

    def test_installed_sdk_serializes_scoped_list_and_episode_delete_without_network(self):
        client = MemMachineClient(base_url="http://127.0.0.1:8080", timeout=3, max_retries=0)
        project = Project(client=client, org_id="vechar-org", project_id="vechar-test")
        client.get_project = Mock(return_value=project)
        seen = []

        def sdk_response(method, url, **kwargs):
            response = Mock()
            if url.endswith("/api/v2/health"):
                response.json.return_value = {"status": "healthy", "version": "0.3.9"}
            elif url.endswith("/api/v2/memory/episodic/config/get"):
                response.json.return_value = {
                    "enabled": True, "long_term_memory_enabled": True,
                    "short_term_memory_enabled": False,
                }
            elif url.endswith("/api/v2/memories/list"):
                seen.append(kwargs["json"])
                episodes = [] if len(seen) > 1 else [{
                    "uid": "target-episode", "content": "hello", "session_key": "s1",
                    "created_at": "2026-01-01T00:00:00Z", "producer_id": "user:u1",
                    "producer_role": "user", "metadata": {
                        "user_id": "user:u1", "agent_id": "character:c1",
                    },
                }]
                response.json.return_value = {"status": 0, "content": {"episodic_memory": episodes}}
            elif url.endswith("/api/v2/memories/episodic/delete"):
                response.json.return_value = {}
            else:
                raise AssertionError(f"Unexpected SDK request: {url}")
            return response

        client.request = Mock(side_effect=sdk_response)
        try:
            adapter = MemMachineAdapter(config(), client=client)
            adapter.delete_character(user_id="u1", character_id="c1")
            self.assertEqual(len(seen), 2)
            self.assertIn("metadata.user_id='user:u1'", seen[0]["filter"])
            self.assertIn("metadata.agent_id='character:c1'", seen[0]["filter"])
            deletes = [call.kwargs["json"] for call in client.request.call_args_list
                       if call.args[1].endswith("/api/v2/memories/episodic/delete")]
            self.assertEqual(len(deletes), 1)
            self.assertEqual(deletes[0]["episodic_id"], "target-episode")
        finally:
            client.close()

    def test_timeout_connection_and_rate_limit_fallback(self):
        adapter, memory, _, _ = self.adapter()
        response = requests.Response()
        response.status_code = 429
        cases = [
            (requests.Timeout("fake"), "TimeoutError", True),
            (requests.ConnectionError("fake"), "ConnectionError", False),
            (requests.HTTPError("fake", response=response), "MemoryRateLimitError", False),
        ]
        for error, expected_type, timed_out in cases:
            with self.subTest(expected_type=expected_type):
                memory.error = error
                with self.gateway(adapter), self.assertLogs(memory_service.logger, level="WARNING"):
                    result = memory_service.retrieve_for_turn(
                        user_id="u1", character_id="c1", conversation_id="s1", current_message="hello",
                    )
                self.assertEqual((result.success, result.error_type, result.timed_out),
                                 (False, expected_type, timed_out))

    def test_partial_turn_ingestion_is_reported_as_failure(self):
        adapter, memory, _, _ = self.adapter()
        memory.fail_second_add = True
        with self.gateway(adapter), self.assertLogs(memory_service.logger, level="WARNING"):
            success = memory_service.record_completed_turn(
                user_id="u1", character_id="c1", conversation_id="s1",
                user_message_id="um1", assistant_message_id="am1",
                user_message="hello", assistant_message="hi",
            )
        self.assertFalse(success)
        self.assertEqual(len(memory.add_calls), 2)

    def test_scoped_deletion_uses_list_and_id_delete(self):
        adapter, memory, _, _ = self.adapter()
        with self.gateway(adapter):
            character = memory_service.delete_character_memories(user_id="u1", character_id="c1")
            conversation = memory_service.delete_conversation_memories(
                user_id="u1", character_id="c1", conversation_id="s1"
            )
            user = memory_service.delete_user_memories(user_id="u1")
        for result in (character, conversation, user):
            self.assertEqual((result.success, result.retryable, result.provider),
                             (True, False, "memmachine"))
        self.assertEqual(memory.add_calls, [])


if __name__ == "__main__":
    unittest.main()
