"""
Kernel host: the native kernel runs the program and answers every ask itself.

The `orbital_kernel` extension runs the program on the native server (`NativeServer`, the routes `orb serve` serves) and holds
the whole execution (guards, transitions, effects) and its I/O: rows from the
store the environment selects (DATA_BACKEND, DATABASE_URL, COUCHDB_URL, the Firebase variables), `http` and `os/`
under the program's own egress and grants, secrets and credentials from the environment, the clock and random bytes.
This module loads it, serves its events and routes the live items a dispatch produces (a streaming call's messages to
the connection that made the request, `persist` successes to the others) with the kernel's own routing rule.
"""
import asyncio
import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import APIRouter, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from .websocket import connection_manager

KERNEL_DIR_ENV = "ORBITAL_KERNEL_DIR"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
GENERATED = Path(__file__).resolve().parents[1] / "generated"


def catalogs_of(program: str, locales_dir: Path) -> Dict[str, Any]:
    """The message catalogs of every locale the program declares, from the compile's `locales/`."""
    catalogs: Dict[str, Any] = {}
    for locale in json.loads(program).get("locales") or []:
        file = locales_dir / f"{locale}.json"
        if not file.is_file():
            raise RuntimeError(f'the program declares locale "{locale}" but {file} is missing')
        catalogs[locale] = json.loads(file.read_text(encoding="utf-8"))
    return catalogs


def load_extension(directory: Optional[Path] = None) -> Any:
    """The native kernel extension from `directory`, else $ORBITAL_KERNEL_DIR, else the project's kernel/."""
    found = directory or Path(os.environ.get(KERNEL_DIR_ENV) or PROJECT_ROOT / "kernel")
    if str(found) not in sys.path:
        sys.path.insert(0, str(found))
    try:
        import orbital_kernel
    except ImportError as error:
        raise RuntimeError(
            f"the native kernel extension is not in {found} (put orbital_kernel.abi3.so there or set {KERNEL_DIR_ENV}): {error}"
        ) from error
    return orbital_kernel


class KernelHost:
    """One program running in the native kernel."""

    def __init__(
        self,
        program: str,
        language: str,
        env: Dict[str, str],
        on_live: Optional[Callable[[str], None]] = None,
        directory: Optional[Path] = None,
        catalogs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._kernel = load_extension(directory)
        self._kernel.install_language(language)
        self._program = program
        self._env = json.dumps(env)
        self._catalogs = json.dumps(catalogs or {})
        self._on_live = on_live
        self._server = self._open()

    def _open(self) -> Any:
        return self._kernel.NativeServer.open(self._program, self._env, self._catalogs, self._on_live)

    @classmethod
    def from_project(cls, on_live: Optional[Callable[[str], None]] = None) -> "KernelHost":
        """The program and language the compile wrote under generated/, with the process environment."""
        program = (GENERATED / "program.json").read_text(encoding="utf-8")
        return cls(
            program,
            (GENERATED / "language.json").read_text(encoding="utf-8"),
            dict(os.environ),
            on_live,
            catalogs=catalogs_of(program, PROJECT_ROOT / "locales"),
        )

    def dispatch(self, orbital: str, request: Dict[str, Any], authorization: Optional[str] = None) -> Dict[str, Any]:
        """
        Run one event through the server's `POST /api/orbitals/<orbital>/events` as the viewer `authorization` (an
        Authorization header value) names; a `user` in the request is dropped. A dispatch, or a refusal of it (403), answers
        with its response; a bearer that does not verify raises PermissionError, any other failure RuntimeError.
        """
        request = {k: v for k, v in request.items() if k != "user"}
        headers = [("content-type", "application/json")] + ([("authorization", authorization)] if authorization else [])
        status, _, body = self._server.handle("POST", f"/api/orbitals/{orbital}/events", headers, json.dumps(request).encode())
        answer = json.loads(bytes(body))
        if status == 401:
            raise PermissionError(answer.get("error", "Unauthorized"))
        if status != 403 and not 200 <= status < 300:
            raise RuntimeError(answer.get("error", f"the server answered {status}"))
        return answer

    def viewer(self, authorization: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """The user `authorization` names (the anonymous viewer without one), None when the program declares no sign-in."""
        return json.loads(self._server.viewer(authorization))

    def live_recipients(self, item_json: str, connections: List[Dict[str, Any]]) -> List[int]:
        """The indexes of `connections` ({clientId, viewer}) that receive the live item: the kernel's routing rule."""
        return json.loads(self._server.live_recipients(item_json, json.dumps(connections)))

    def state(self, trait: str) -> str:
        return self._server.state(trait)

    def reset(self) -> None:
        """A fresh kernel on the same program: every trait back at its initial state, the store as the environment selects."""
        self._server = self._open()

    def close(self) -> None:
        self._server = None


class LiveHub:
    """The websocket connections a kernel host fans live items out to; the kernel decides who receives each one."""

    def __init__(self) -> None:
        self._connections: List[Tuple[WebSocket, Optional[str], Optional[Dict[str, Any]]]] = []
        self._host: Optional[KernelHost] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pending: List[concurrent.futures.Future] = []
        self._lock = threading.Lock()

    def attach(self, host: KernelHost, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self._host = host
        if loop is not None:
            self._loop = loop

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, item_json: str) -> None:
        """Called by the kernel, from the thread running a dispatch, for each live item."""
        if self._loop is None:
            raise RuntimeError("the live hub has no event loop to deliver on")
        future = asyncio.run_coroutine_threadsafe(self._deliver(item_json), self._loop)
        with self._lock:
            self._pending.append(future)

    async def settle(self) -> None:
        """Wait until every item published so far has been sent."""
        with self._lock:
            pending, self._pending = self._pending, []
        await asyncio.gather(*(asyncio.wrap_future(future) for future in pending))

    async def _deliver(self, item_json: str) -> None:
        if self._host is None:
            raise RuntimeError("the live hub has no kernel host")
        connections = list(self._connections)
        recipients = self._host.live_recipients(
            item_json, [{"clientId": client_id, "viewer": viewer} for _, client_id, viewer in connections]
        )
        item = json.loads(item_json)
        frame = json.dumps(
            {"type": "bus", "event": item["event"], "payload": item.get("payload"), "source": item["source"], "timestamp": int(time.time() * 1000)}
        )
        for at in recipients:
            try:
                await connections[at][0].send_text(frame)
            except Exception:
                self._drop(connections[at][0])

    def _drop(self, websocket: WebSocket) -> None:
        self._connections = [c for c in self._connections if c[0] is not websocket]

    async def serve(self, websocket: WebSocket, client_id: Optional[str], viewer: Optional[Dict[str, Any]]) -> None:
        await websocket.accept()
        self._connections.append((websocket, client_id, viewer))
        try:
            while True:
                if await websocket.receive_text() == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            self._drop(websocket)


class EventRequest(BaseModel):
    """The body of POST /<trait>/events."""

    event: Optional[str] = None
    entity_id: Optional[str] = Field(None, alias="entityId")
    payload: Dict[str, Any] = {}
    client_id: Optional[str] = Field(None, alias="clientId")
    model_config = ConfigDict(populate_by_name=True)


def _handler(host: KernelHost, hub: LiveHub, orbital: str, trait: str, path: str) -> Callable[..., Any]:
    async def handle_event(request: EventRequest, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        if not request.event:
            return {"success": False, "error": "Missing 'event' field"}
        body: Dict[str, Any] = {"event": request.event, "targetTrait": trait, "payload": request.payload}
        if request.entity_id is not None:
            body["entityId"] = request.entity_id
        if request.client_id is not None:
            body["clientId"] = request.client_id
        hub.bind_loop(asyncio.get_running_loop())
        try:
            response = await asyncio.to_thread(host.dispatch, orbital, body, authorization)
        except PermissionError as error:
            raise HTTPException(status_code=401, detail=str(error)) from error
        except Exception as error:
            return {"success": False, "error": str(error)}
        await hub.settle()
        response["newState"] = response.get("states", {}).get(trait)
        effects = response.get("clientEffects") or []
        if effects:
            await connection_manager.broadcast_client_effects(path, request.entity_id or "unknown", request.event, list(effects), {})
        return response

    return handle_event


def _state_route(host: KernelHost, trait: str) -> Callable[[], Any]:
    async def get_state() -> Dict[str, str]:
        return {"state": host.state(trait)}

    return get_state


def _reset_route(host: KernelHost) -> Callable[[], Any]:
    async def reset() -> Dict[str, str]:
        if not os.environ.get("TESTING"):
            raise HTTPException(status_code=403, detail="Reset only available in test mode")
        host.reset()
        return {"status": "ok"}

    return reset


def register_kernel_routes(
    app: FastAPI,
    traits: Dict[str, Tuple[str, str]],
    host: Optional[KernelHost] = None,
    hub: Optional[LiveHub] = None,
) -> KernelHost:
    """
    POST /api/<trait>/events, GET /api/<trait>/state and (TESTING only) POST /api/<trait>/reset for each trait, and the
    live channel WS /ws/live?clientId=&token= every dispatch's live items are routed to.
    """
    hub = hub or LiveHub()
    host = host or KernelHost.from_project(hub.publish)
    hub.attach(host)

    @app.websocket("/ws/live")
    async def live(websocket: WebSocket, clientId: Optional[str] = None, token: Optional[str] = None) -> None:
        try:
            viewer = host.viewer(f"Bearer {token}" if token else None)
        except PermissionError:
            await websocket.close(code=4401)
            return
        await hub.serve(websocket, clientId, viewer)

    router = APIRouter(prefix="/api")
    for path, (orbital, trait) in traits.items():
        router.add_api_route(f"/{path}/events", _handler(host, hub, orbital, trait, path), methods=["POST"], tags=[trait])
        router.add_api_route(f"/{path}/state", _state_route(host, trait), methods=["GET"], tags=[trait])
        router.add_api_route(f"/{path}/reset", _reset_route(host), methods=["POST"], tags=[trait])
    app.include_router(router)
    app.state.kernel_host = host
    return host
