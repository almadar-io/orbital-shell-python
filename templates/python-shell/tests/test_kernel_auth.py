"""
Who a request is in kernel mode: the viewer comes from the verified bearer, never from the request body.

A forged `user` in the body is ignored; a valid bearer is that viewer's @user; an invalid bearer is a 401; no bearer is
the anonymous viewer. A live websocket carries only the rows its viewer may read: an anonymous socket what @read admits
for no one, a signed-in one its own rows as well.

Needs the native extension (ORBITAL_KERNEL_DIR), the language a compile ships (ORBITAL_LANGUAGE_JSON) and the sign-in
fixtures of @almadar/auth (ORBITAL_AUTH_FIXTURES, its test/fixtures/tokens.json); skipped without them.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from orbital_app.core.kernel_host import KernelHost, LiveHub, register_kernel_routes

KERNEL_DIR = os.environ.get("ORBITAL_KERNEL_DIR")
LANGUAGE = os.environ.get("ORBITAL_LANGUAGE_JSON")
FIXTURES = os.environ.get("ORBITAL_AUTH_FIXTURES")

pytestmark = pytest.mark.skipif(
    KERNEL_DIR is None or LANGUAGE is None or FIXTURES is None,
    reason="needs ORBITAL_KERNEL_DIR, ORBITAL_LANGUAGE_JSON and ORBITAL_AUTH_FIXTURES",
)

READ = ["or", ["=", ["object/get", "@entity", "visibility"], "public"], ["=", ["object/get", "@entity", "ownerId"], "@user.id"]]


def program() -> dict:
    return {
        "name": "board",
        "orbitals": [
            {
                "name": "Main",
                "entity": {
                    "name": "Note",
                    "persistence": "persistent",
                    "collection": "notes",
                    "read_policy": READ,
                    "fields": [{"name": "id", "type": "string"}, {"name": "visibility", "type": "string"}, {"name": "ownerId", "type": "string"}],
                },
                "traits": [
                    {
                        "name": "Board",
                        "linkedEntity": "Note",
                        "stateMachine": {
                            "states": [{"name": "idle", "isInitial": True}],
                            "transitions": [
                                {"from": "idle", "to": "idle", "event": "WHOAMI", "effects": [["emit", "WHO", {"id": "@user.id"}]]},
                                {
                                    "from": "idle",
                                    "to": "idle",
                                    "event": "CREATE",
                                    "effects": [
                                        ["persist", "create", "Note", {"id": "@payload.id", "visibility": "@payload.visibility", "ownerId": "@payload.owner"}, {"emit": {"success": "SAVED"}}]
                                    ],
                                },
                            ],
                        },
                    }
                ],
            }
        ],
    }


@pytest.fixture
def tokens():
    return json.loads(Path(FIXTURES).read_text())["firebase"]


@pytest.fixture
def app(tokens):
    certs = json.dumps(tokens["certs"]).encode()

    class Certs(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("cache-control", "public, max-age=3600")
            self.send_header("content-length", str(len(certs)))
            self.end_headers()
            self.wfile.write(certs)

        def log_message(self, *args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Certs)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    env = {
        "NODE_ENV": "development",
        "DATA_BACKEND": "mock",
        "FIREBASE_PROJECT_ID": tokens["projectId"],
        "FIREBASE_CERTS_URL": f"{origin}/certs",
    }
    hub = LiveHub()
    host = KernelHost(json.dumps(program()), Path(LANGUAGE).read_text(encoding="utf-8"), env, hub.publish, Path(KERNEL_DIR))
    application = FastAPI()
    register_kernel_routes(application, {"board": ("Main", "Board")}, host, hub)
    yield TestClient(application)
    host.close()
    server.shutdown()


@pytest.fixture
def token(tokens):
    return next(case["token"] for case in tokens["cases"] if case["name"] == "project token")


def who(client, headers=None, **body):
    response = client.post("/api/board/events", json={"event": "WHOAMI", **body}, headers=headers or {})
    return response


def emitted_id(response):
    return next(e["payload"]["id"] for e in response.json()["emittedEvents"] if e["event"] == "WHO")


def test_a_valid_bearer_is_that_viewers_user(app, token):
    response = who(app, {"authorization": f"Bearer {token}"})
    assert response.status_code == 200 and emitted_id(response) == "uid-1"


def test_control_under_sign_in_no_bearer_is_a_401_and_a_forged_body_user_is_ignored(app, token):
    assert who(app).status_code == 401, "signed-out requests under sign-in are refused (as orbital-server does)"
    assert who(app, user={"id": "mallory", "role": "admin"}).status_code == 401, "the body user is never read"
    signed_in = who(app, {"authorization": f"Bearer {token}"}, user={"id": "mallory"})
    assert emitted_id(signed_in) == "uid-1", "control: nor beside a valid bearer"


def test_an_invalid_bearer_is_a_401(app, token):
    tampered = token[:-4] + ("AAAA" if not token.endswith("AAAA") else "BBBB")
    assert who(app, {"authorization": f"Bearer {tampered}"}).status_code == 401
    assert who(app, {"authorization": "Bearer nonsense"}).status_code == 401
    assert who(app, {"authorization": token}).status_code == 401, "a header that is not a Bearer is refused"


def create(client, token, row_id, visibility, owner):
    response = client.post(
        "/api/board/events",
        json={"event": "CREATE", "clientId": "writer", "payload": {"id": row_id, "visibility": visibility, "owner": owner}},
        headers={"authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200 and response.json()["success"] is True, response.text


def drain(socket):
    socket.send_text("ping")
    ids = []
    while True:
        text = socket.receive_text()
        if text == "pong":
            return ids
        frame = json.loads(text)
        if frame["event"] == "SAVED":
            ids.append(frame["payload"]["id"])


def test_an_anonymous_socket_gets_only_the_public_row_and_a_signed_in_socket_its_own_as_well(app, token):
    with app.websocket_connect("/ws/live?clientId=reader") as anonymous, app.websocket_connect(f"/ws/live?clientId=member&token={token}") as member:
        create(app, token, "n1", "public", "someone")
        create(app, token, "n2", "private", "uid-1")
        create(app, token, "n3", "private", "someone")
        assert drain(anonymous) == ["n1"]
        assert drain(member) == ["n1", "n2"]


def test_an_invalid_websocket_token_is_refused(app):
    with pytest.raises(WebSocketDisconnect) as refused:
        with app.websocket_connect("/ws/live?clientId=x&token=nonsense"):
            pass
    assert refused.value.code == 4401
