"""
Kernel host: the native kernel runs a program and answers its own I/O.

Needs the native extension (ORBITAL_KERNEL_DIR) and the language a compile ships (ORBITAL_LANGUAGE_JSON, the
generated/language.json of any compiled project); skipped without them.
"""
import json
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from orbital_app.core.kernel_host import KernelHost, register_kernel_routes

KERNEL_DIR = os.environ.get("ORBITAL_KERNEL_DIR")
LANGUAGE = os.environ.get("ORBITAL_LANGUAGE_JSON")

pytestmark = pytest.mark.skipif(
    KERNEL_DIR is None or LANGUAGE is None,
    reason="needs ORBITAL_KERNEL_DIR (the native kernel extension) and ORBITAL_LANGUAGE_JSON (a compiled project's language.json)",
)

MEMORY = {"NODE_ENV": "development", "DATA_BACKEND": "mock"}
EMIT = {"emit": {"success": "SAVED", "failure": "SAVE_FAILED"}}
NOTES = {
    "name": "notes",
    "orbitals": [
        {
            "name": "Main",
            "entity": {
                "name": "Note",
                "persistence": "persistent",
                "collection": "notes",
                "fields": [{"name": "id", "type": "string"}, {"name": "title", "type": "string"}],
            },
            "traits": [
                {
                    "name": "Draft",
                    "linkedEntity": "Note",
                    "stateMachine": {
                        "states": [{"name": "editing", "isInitial": True}, {"name": "saving"}, {"name": "saved"}, {"name": "failed"}],
                        "transitions": [
                            {"from": "editing", "to": "saving", "event": "SAVE", "effects": [["persist", "create", "Note", {"id": "d1", "title": "draft"}, EMIT]]},
                            {"from": "saving", "to": "saved", "event": "SAVED"},
                            {"from": "saving", "to": "failed", "event": "SAVE_FAILED"},
                        ],
                    },
                }
            ],
        }
    ],
}


@pytest.fixture
def host():
    kernel = KernelHost(json.dumps(NOTES), Path(LANGUAGE).read_text(encoding="utf-8"), MEMORY, directory=Path(KERNEL_DIR))
    yield kernel
    kernel.close()


def test_an_event_runs_to_its_response_with_the_row_in_the_kernels_own_store(host):
    response = host.dispatch("Main", {"event": "SAVE", "targetTrait": "Draft"})
    assert response["success"] is True
    assert response["states"]["Draft"] == "saved"
    assert host.state("Draft") == "saved"


def test_control_an_unknown_orbital_is_an_error_and_a_reset_starts_a_fresh_kernel(host):
    with pytest.raises(RuntimeError, match="Orbital not found"):
        host.dispatch("Nowhere", {"event": "SAVE"})
    host.dispatch("Main", {"event": "SAVE", "targetTrait": "Draft"})
    host.reset()
    assert host.state("Draft") == "editing", "a fresh kernel is at the initial state"
    assert host.dispatch("Main", {"event": "SAVE", "targetTrait": "Draft"})["states"]["Draft"] == "saved"


def test_control_production_refuses_the_memory_store():
    with pytest.raises(RuntimeError, match="not permitted"):
        KernelHost(json.dumps(NOTES), Path(LANGUAGE).read_text(encoding="utf-8"), {"DATA_BACKEND": "mock"}, directory=Path(KERNEL_DIR))


def test_the_routes_serve_events_state_and_a_guarded_reset(host, monkeypatch):
    app = FastAPI()
    register_kernel_routes(app, {"draft": ("Main", "Draft")}, host)
    client = TestClient(app)
    body = client.post("/api/draft/events", json={"event": "SAVE"}).json()
    assert body["success"] is True and body["newState"] == "saved"
    assert client.get("/api/draft/state").json() == {"state": "saved"}
    monkeypatch.delenv("TESTING", raising=False)
    assert client.post("/api/draft/reset").status_code == 403
    monkeypatch.setenv("TESTING", "1")
    assert client.post("/api/draft/reset").status_code == 200
    assert client.get("/api/draft/state").json() == {"state": "editing"}
    assert client.post("/api/draft/events", json={}).json()["success"] is False
