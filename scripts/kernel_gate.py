"""
The kernel-host gate: compile each program with the Python shell, boot the app and drive it over HTTP.

For each program: `orb compile --shell python` writes a kernel-hosted project; the app boots under uvicorn; /health
answers; every trait's state route answers; every guard-free event of a trait's initial state is POSTed with an empty
payload and must answer HTTP 200 with a `success` verdict and no error naming a host ask. The trace (event, newState,
emitted event names) goes to --trace as JSON, the file `kernel_twin.ts` replays on the Node add-on.

  kernel_gate.py --orb ORB --kernel-dir DIR [--trace FILE] PROGRAM.lolo...
"""
import argparse
import ast
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


def kebab(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def call(url: str, body: Optional[Dict[str, Any]] = None) -> Any:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def initial_events(trait: Dict[str, Any]) -> List[str]:
    machine = trait.get("stateMachine") or {}
    initial = next((s["name"] for s in machine.get("states", []) if s.get("isInitial")), None)
    events: List[str] = []
    for transition in machine.get("transitions", []):
        sources = transition["from"] if isinstance(transition["from"], list) else [transition["from"]]
        if initial in sources and not transition.get("guard") and transition["event"] not in events:
            events.append(transition["event"])
    return events


def trait_events(project: Path) -> Dict[str, List[str]]:
    program = json.loads((project / "src/orbital_app/generated/program.json").read_text())
    by_name = {trait["name"]: initial_events(trait) for orbital in program["orbitals"] for trait in orbital["traits"] if isinstance(trait, dict)}
    routes = (project / "src/orbital_app/generated/routes/__init__.py").read_text()
    traits = ast.literal_eval(re.search(r"TRAITS = (\{.*?\n\})", routes, re.S).group(1))
    return {path: by_name.get(trait, []) for path, (_orbital, trait) in traits.items()}


def gate(program: Path, orb: str, python: str, kernel_dir: str, work: Path) -> Dict[str, Any]:
    project = work / program.stem
    compiled = subprocess.run([orb, "compile", str(program), "--shell", "python", "-o", str(project)], capture_output=True, text=True)
    if compiled.returncode != 0:
        raise SystemExit(f"{program.stem}: compile failed\n{compiled.stdout}{compiled.stderr}")
    port = free_port()
    env = {**os.environ, "PYTHONPATH": str(project / "src"), "ORBITAL_KERNEL_DIR": kernel_dir, "NODE_ENV": "development", "DATA_BACKEND": "mock", "TESTING": "1"}
    log = open(work / f"{program.stem}.log", "w")
    server = subprocess.Popen([python, "-m", "uvicorn", "orbital_app.main:app", "--port", str(port)], cwd=project, env=env, stdout=log, stderr=log)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if call(f"{base}/health")["status"] == "healthy":
                    break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.2)
        else:
            raise SystemExit(f"{program.stem}: the app did not boot\n{(work / (program.stem + '.log')).read_text()}")
        trace: Dict[str, Any] = {}
        for path, events in trait_events(project).items():
            call(f"{base}/api/{path}/state")
            for event in events:
                result = call(f"{base}/api/{path}/events", {"event": event, "payload": {}})
                if not isinstance(result.get("success"), bool):
                    raise SystemExit(f"{program.stem} {path} {event}: no success verdict: {result}")
                if "asked the host" in str(result.get("error", "")):
                    raise SystemExit(f"{program.stem} {path} {event}: {result['error']}")
                trace[f"{path}/{event}"] = {
                    "success": result["success"],
                    "newState": result.get("newState"),
                    "emitted": [e["event"] for e in result.get("emittedEvents", [])],
                }
        return {"traits": len(trace), "events": trace}
    finally:
        server.terminate()
        server.wait(timeout=15)
        log.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orb", required=True)
    parser.add_argument("--kernel-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--trace")
    parser.add_argument("programs", nargs="+", type=Path)
    args = parser.parse_args()
    report: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for program in args.programs:
            report[program.stem] = gate(program, args.orb, args.python, args.kernel_dir, Path(tmp))
            print(f"PASS {program.stem}: booted, {len(report[program.stem]['events'])} initial event(s) driven")
    if args.trace:
        Path(args.trace).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
