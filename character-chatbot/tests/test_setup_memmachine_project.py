"""The admin helper is explicit and uses no server, user data, or OpenAI calls."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

import setup_memmachine_project as setup
from app.memory_config import MemoryConfigurationError, load_memory_config


def config(provider="memmachine"):
    if provider == "noop":
        return load_memory_config({"MEMORY_PROVIDER": "noop"})
    return load_memory_config({
        "MEMORY_PROVIDER": "memmachine",
        "MEMMACHINE_BASE_URL": "http://127.0.0.1:8080",
        "MEMMACHINE_ORG_ID": "vechar-org",
        "MEMMACHINE_PROJECT_ID": "vechar-project",
    })


def fake_client(*, enabled=True, long_term=True, short_term=False, version="0.3.9"):
    memory = Mock()
    memory.get_episodic_memory_config.return_value = SimpleNamespace(
        enabled=enabled, long_term_memory_enabled=long_term,
        short_term_memory_enabled=short_term,
    )
    project = Mock()
    project.org_id = "vechar-org"
    project.project_id = "vechar-project"
    project.memory.return_value = memory
    client = Mock()
    client.get_project.return_value = project
    client.create_project.return_value = project
    health = Mock()
    health.json.return_value = {"status": "healthy", "version": version}
    client.request.return_value = health
    return client, memory


class SetupMemMachineProjectTests(unittest.TestCase):
    def test_check_mode_is_read_only(self):
        client, memory = fake_client()
        with patch.object(setup, "load_memory_config", return_value=config()), patch.object(
            setup, "MemMachineAdapter",
        ) as adapter:
            result = setup.prepare_project(client=client)
        self.assertEqual(result, (False, False))
        adapter.assert_called_once()
        client.create_project.assert_not_called()
        client.get_project.assert_not_called()
        memory.configure_episodic_memory.assert_not_called()

    def test_apply_existing_project_sets_long_term_only(self):
        client, memory = fake_client(short_term=True)
        with patch.object(setup, "load_memory_config", return_value=config()), patch.object(
            setup, "MemMachineAdapter",
        ):
            result = setup.prepare_project(apply=True, client=client)
        self.assertEqual(result, (False, True))
        client.create_project.assert_not_called()
        memory.configure_episodic_memory.assert_called_once_with(
            enabled=True, long_term_memory_enabled=True,
            short_term_memory_enabled=False, timeout=3,
        )

    def test_apply_creates_only_when_project_is_missing(self):
        client, memory = fake_client(enabled=False, long_term=False, short_term=True)
        response = requests.Response()
        response.status_code = 404
        client.get_project.side_effect = requests.HTTPError("missing", response=response)
        with patch.object(setup, "load_memory_config", return_value=config()), patch.object(
            setup, "MemMachineAdapter",
        ):
            result = setup.prepare_project(apply=True, client=client)
        self.assertEqual(result, (True, True))
        client.create_project.assert_called_once()
        self.assertEqual(client.create_project.call_args.kwargs["org_id"], "vechar-org")
        self.assertEqual(client.create_project.call_args.kwargs["project_id"], "vechar-project")
        memory.configure_episodic_memory.assert_called_once()

    def test_incompatible_server_cannot_be_modified(self):
        client, memory = fake_client(version="0.3.8")
        with patch.object(setup, "load_memory_config", return_value=config()), patch.object(
            setup, "MemMachineAdapter",
        ):
            with self.assertRaises(MemoryConfigurationError):
                setup.prepare_project(apply=True, client=client)
        client.get_project.assert_not_called()
        client.create_project.assert_not_called()
        memory.configure_episodic_memory.assert_not_called()

    def test_incompatible_sdk_cannot_be_modified(self):
        client, memory = fake_client()
        with patch.object(setup, "load_memory_config", return_value=config()), patch.object(
            setup, "validate_sdk_versions",
            side_effect=MemoryConfigurationError("incompatible SDK"),
        ):
            with self.assertRaises(MemoryConfigurationError):
                setup.prepare_project(apply=True, client=client)
        client.request.assert_not_called()
        client.create_project.assert_not_called()
        memory.configure_episodic_memory.assert_not_called()

    def test_wrong_project_identity_cannot_be_modified(self):
        client, memory = fake_client(short_term=True)
        client.get_project.return_value.org_id = "other-org"
        with patch.object(setup, "load_memory_config", return_value=config()), patch.object(
            setup, "MemMachineAdapter",
        ):
            with self.assertRaisesRegex(MemoryConfigurationError, "identity"):
                setup.prepare_project(apply=True, client=client)
        memory.configure_episodic_memory.assert_not_called()

    def test_noop_environment_cannot_run_apply(self):
        client, _ = fake_client()
        with patch.object(setup, "load_memory_config", return_value=config("noop")):
            with self.assertRaises(MemoryConfigurationError):
                setup.prepare_project(apply=True, client=client)
        client.request.assert_not_called()
        client.create_project.assert_not_called()


if __name__ == "__main__":
    unittest.main()
