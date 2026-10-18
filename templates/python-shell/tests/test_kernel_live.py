"""
Live items through the kernel host: a streaming std-llm generate (a recorded SSE reply) delivers its deltas to the
websocket of the request that made the call and to no other connection.

Needs the native extension (ORBITAL_KERNEL_DIR), the language a compile ships (ORBITAL_LANGUAGE_JSON) and `orb`
(ORBITAL_ORB) run from a directory where @almadar/std resolves (ORBITAL_PROJECT_DIR) to resolve the fixture program;
skipped without them.
"""
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from orbital_app.core.kernel_host import KernelHost, LiveHub, register_kernel_routes

KERNEL_DIR = os.environ.get("ORBITAL_KERNEL_DIR")
LANGUAGE = os.environ.get("ORBITAL_LANGUAGE_JSON")
ORB = os.environ.get("ORBITAL_ORB")
PROJECT_DIR = os.environ.get("ORBITAL_PROJECT_DIR")
FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skipif(
    KERNEL_DIR is None or LANGUAGE is None or ORB is None or PROJECT_DIR is None,
    reason="needs ORBITAL_KERNEL_DIR, ORBITAL_LANGUAGE_JSON, ORBITAL_ORB and ORBITAL_PROJECT_DIR",
)

SSE = (FIXTURES / "chat-stream.sse").read_bytes()


class RecordedChat(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length", 0)))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(SSE)))
        self.end_headers()
        self.wfile.write(SSE)

    def log_message(self, *args) -> None:
        return


@pytest.fixture
def program(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), RecordedChat)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    source = tmp_path / "stream-chat.lolo"
    source.write_text((FIXTURES / "stream-chat.lolo.tmpl").read_text().replace("__PORT__", str(server.server_address[1])))
    resolved = subprocess.run([ORB, "resolve", str(source)], capture_output=True, text=True, check=True, cwd=PROJECT_DIR)
    yield resolved.stdout
    server.shutdown()


def messages(socket):
    """Everything queued on a socket, read up to the pong that follows a ping."""
    socket.send_text("ping")
    frames = []
    while True:
        text = socket.receive_text()
        if text == "pong":
            return frames
        frames.append(json.loads(text))


def serve(program):
    hub = LiveHub()
    host = KernelHost(program, Path(LANGUAGE).read_text(encoding="utf-8"), {"NODE_ENV": "development", "DATA_BACKEND": "mock"}, hub.publish, Path(KERNEL_DIR))
    app = FastAPI()
    register_kernel_routes(app, {"chat-llm": ("ChatOrbital", "ChatLlm")}, host, hub)
    return host, TestClient(app)


def generate(client, client_id):
    body = {"event": "LLM_GENERATE", "payload": {"userPrompt": "hi", "stream": True, "key": "k1"}}
    if client_id is not None:
        body["clientId"] = client_id
    response = client.post("/api/chat-llm/events", json=body).json()
    assert response["success"] is True, response


def stream_text(frames):
    streamed = [f for f in frames if f["event"] == "LLM_CHAT_STREAM_MSG"]
    data = [f["payload"]["data"] for f in streamed if f["payload"]["data"] != "[DONE]"]
    return len(streamed), "".join(json.loads(d).get("choices", [{}])[0].get("delta", {}).get("content", "") for d in data)


def test_the_requesting_connection_gets_the_deltas_and_another_connection_gets_none(program):
    host, client = serve(program)
    with client.websocket_connect("/ws/live?clientId=alice") as alice, client.websocket_connect("/ws/live?clientId=bob") as bob:
        generate(client, "alice")
        count, text = stream_text(messages(alice))
        assert count >= 4 and text == "Hello, world"
        assert messages(bob) == [], "control: a connection that did not make the request gets none of the stream"
    host.close()


def test_control_the_stream_follows_the_request_not_the_first_connection(program):
    host, client = serve(program)
    with client.websocket_connect("/ws/live?clientId=alice") as alice, client.websocket_connect("/ws/live?clientId=bob") as bob:
        generate(client, "bob")
        assert stream_text(messages(bob))[1] == "Hello, world"
        assert messages(alice) == []
    host.close()


def test_control_a_request_with_no_client_id_streams_to_nobody(program):
    host, client = serve(program)
    with client.websocket_connect("/ws/live?clientId=alice") as alice, client.websocket_connect("/ws/live") as anonymous:
        generate(client, None)
        assert messages(alice) == [] and messages(anonymous) == []
    host.close()
