"""Runtime sidecar client — delegates deferred effects to the JS runtime.

The compiled Python app is the PRIMARY executor of the orbital circuit. Some
effects (notably ``call-service "llm" …`` → real LLM providers) are deferred
to the JS runtime sidecar (``@almadar-io/playground-runtime``), which owns the
configured integrations. This module is the HTTP bridge: each deferred
call-service is POSTed to the sidecar's ``POST /api/eval`` endpoint and the
returned value is bound back to the handler's ``?<field>`` captures.

No third-party dependency: stdlib ``urllib`` only, so the shell boots without
``requests``/``httpx``. The blocking urllib call is offloaded to a thread
executor so the async event loop is not stalled.

Configure via ``SIDECAR_URL`` (default ``http://localhost:4001``).
"""

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

DEFAULT_SIDECAR_URL = "http://localhost:4001"


class RuntimeSidecar:
    """HTTP client for the JS runtime sidecar's ``/api/eval`` endpoint."""

    def __init__(self, url: Optional[str] = None, timeout: float = 120.0) -> None:
        self.url = (
            url or os.environ.get("SIDECAR_URL") or DEFAULT_SIDECAR_URL
        ).rstrip("/")
        self.timeout = timeout

    def _post_sync(self, body: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/api/eval",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"sidecar {self.url}/api/eval returned HTTP {e.code}: {detail}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"sidecar {self.url}/api/eval unreachable: {e.reason}"
            ) from e
        return json.loads(raw.decode("utf-8"))

    async def call_service(
        self, service: str, action: str, params: Any
    ) -> Any:
        """Delegate one ``call-service`` to the sidecar.

        Returns the integration's result value on success. Raises
        ``RuntimeError`` on failure (sidecar reported ``{ok:false}`` or was
        unreachable) so the caller's ``except`` branch routes to the trait's
        failure event (e.g. ``EMBED_FAILED``).
        """
        body = {
            "primitive": "call-service",
            "args": {
                "service": service,
                "action": action,
                "params": _as_params(params),
            },
        }
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(None, self._post_sync, body)
        if not payload.get("ok"):
            raise RuntimeError(
                payload.get("error") or f"{service}.{action} failed"
            )
        return payload.get("value")


def _as_params(params: Any) -> Dict[str, Any]:
    """Normalize the call-service args into the dict the sidecar expects.

    The compiled ``call-service`` form passes a dict (``{texts: [...]}``);
    the legacy substrate operator form (``llm/embed``) passed a positional
    list — wrapped here under a ``args`` key so it still reaches the sidecar
    as a dict. ``None`` becomes ``{}``.
    """
    if params is None:
        return {}
    if isinstance(params, dict):
        return params
    return {"args": params}
