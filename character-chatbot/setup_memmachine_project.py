"""Explicit MemMachine 0.3.9 project setup; never called by Vechar chat.

With no arguments this is a read-only compatibility check. --apply is an
intentional admin operation that may create or reconfigure the named project.
It does not read Vechar user data or load the application's .env file.
"""

import argparse

import requests
from memmachine_client import MemMachineClient

from app.memory_config import MemoryConfigurationError, load_memory_config
from app.services.memory_providers.memmachine import (
    COMPATIBLE_VERSION, MemMachineAdapter, validate_sdk_versions,
)


def prepare_project(*, apply: bool = False, client=None) -> tuple[bool, bool]:
    """Return (created, changed) using only public MemMachine SDK methods."""
    config = load_memory_config()
    if config.provider != "memmachine":
        raise MemoryConfigurationError("Set MEMORY_PROVIDER=memmachine for this explicit admin operation")

    owned_client = client is None
    if client is None:
        client = MemMachineClient(
            base_url=config.base_url, api_key=config.api_key,
            timeout=config.timeout_seconds, max_retries=0,
        )
    try:
        if not apply:
            MemMachineAdapter(config, client=client)
            return False, False

        # Never mutate a project on an incompatible server.
        validate_sdk_versions()
        health = client.request(
            "GET", f"{config.base_url.rstrip('/')}/api/v2/health", timeout=config.timeout_seconds,
        )
        health.raise_for_status()
        health_data = health.json()
        if not isinstance(health_data, dict):
            raise MemoryConfigurationError("MemMachine health response is malformed")
        if health_data.get("status") != "healthy":
            raise RuntimeError("MemMachine server is not healthy")
        if health_data.get("version") != COMPATIBLE_VERSION:
            raise MemoryConfigurationError(f"MemMachine server version must be {COMPATIBLE_VERSION}")

        created = False
        try:
            project = client.get_project(
                org_id=config.org_id, project_id=config.project_id,
                timeout=config.timeout_seconds,
            )
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 404:
                raise
            project = client.create_project(
                org_id=config.org_id, project_id=config.project_id,
                description="Vechar long-term episodic memory",
                timeout=config.timeout_seconds,
            )
            created = True

        if project.org_id != config.org_id or project.project_id != config.project_id:
            raise MemoryConfigurationError("MemMachine project identity does not match configuration")

        memory = project.memory(metadata={})
        current = memory.get_episodic_memory_config(timeout=config.timeout_seconds)
        changed = (current.enabled is not True or
                   current.long_term_memory_enabled is not True or
                   current.short_term_memory_enabled is not False)
        if changed:
            memory.configure_episodic_memory(
                enabled=True, long_term_memory_enabled=True,
                short_term_memory_enabled=False, timeout=config.timeout_seconds,
            )

        MemMachineAdapter(config, client=client)
        return created, changed
    finally:
        if owned_client:
            client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Explicitly create/reconfigure the selected MemMachine project",
    )
    args = parser.parse_args()
    created, changed = prepare_project(apply=args.apply)
    print(f"Vechar MemMachine project verified (created={created}, changed={changed}).")
    if changed:
        print("Restart MemMachine, then rerun without --apply before using this project.")


if __name__ == "__main__":
    main()
