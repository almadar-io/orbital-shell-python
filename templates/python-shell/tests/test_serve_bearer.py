"""A served program binds @user from an Authorization: Bearer token, on the built native kernel.

The program (tests/fixtures/who-served.lolo, std-http-server with std-bearer-verify) answers the id of the viewer a
dispatch ran as. With AUTH_PROVIDER=firebase and the Auth emulator host set, std-bearer-verify verifies an unsigned
(alg none) emulator token with no network and no signature, so no emulator is started.

Run: ORBITAL_KERNEL_DIR=<dir holding orbital_kernel.abi3.so> python3 -m unittest discover -s tests
Skips when orbital_kernel is not importable from there, or when no `orb` binary is found (ORB_BIN, else orb-dev on PATH).
"""
import base64
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1]
FIXTURE = TEMPLATE / "tests" / "fixtures" / "who-served.lolo"
MONOREPO = TEMPLATE.parents[4]
KERNEL_DIR = Path(os.environ.get("ORBITAL_KERNEL_DIR") or TEMPLATE / "kernel")
PROJECT = "demo-project"
SUBJECT = "emu-user-1"
SIGN_IN = ("AUTH_PROVIDER", "FIREBASE_PROJECT_ID", "OIDC_ISSUER_URL", "OIDC_AUDIENCE", "FIREBASE_AUTH_EMULATOR_HOST")
SERVE = "import sys; sys.path.insert(0, sys.argv[1]); import orbital_kernel; orbital_kernel.serve(open(sys.argv[2], encoding='utf-8').read())"


def kernel_importable() -> bool:
    probe = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import orbital_kernel", str(KERNEL_DIR)], capture_output=True)
    return probe.returncode == 0


def orb_binary():
    return os.environ.get("ORB_BIN") or shutil.which("orb-dev")


def b64url(value: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")


def emulator_token(subject: str) -> str:
    now = int(time.time())
    claims = {"iss": f"https://securetoken.google.com/{PROJECT}", "aud": PROJECT, "sub": subject, "iat": now - 60, "auth_time": now - 60, "exp": now + 3600}
    return f"{b64url({'alg': 'none', 'typ': 'JWT'})}.{b64url(claims)}."


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@unittest.skipUnless(kernel_importable(), f"orbital_kernel is not importable from {KERNEL_DIR} (set ORBITAL_KERNEL_DIR to the directory holding orbital_kernel.abi3.so)")
@unittest.skipUnless(orb_binary(), "no orb binary to compile the fixture program (set ORB_BIN, or put orb-dev on PATH)")
class ServedBearer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="orbital_python_bearer_"))
        out = cls.root / "compiled"
        compiled = subprocess.run([orb_binary(), "compile", str(FIXTURE), "-s", "typescript", "-m", "server", "-o", str(out)], cwd=MONOREPO, capture_output=True, text=True)
        assert compiled.returncode == 0, f"compiling the fixture failed:\n{compiled.stdout}\n{compiled.stderr}"
        cls.program = out / "packages" / "server" / "generated" / "program.orb"
        assert cls.program.exists(), f"the compile wrote no served program at {cls.program}"
        cls.servers = []

    @classmethod
    def tearDownClass(cls):
        for process, _ in cls.servers:
            stop(process)
        shutil.rmtree(cls.root, ignore_errors=True)

    @classmethod
    def serve(cls, name: str, sign_in: dict) -> str:
        cwd = cls.root / name
        (cwd / "data" / "rows").mkdir(parents=True)
        (cwd / "data" / "rows" / "members.json").write_text(json.dumps({"changes": [], "rows": [{"id": SUBJECT, "name": "Emulator User", "email": "emu@example.com"}]}), encoding="utf-8")
        port = free_port()
        env = {k: v for k, v in os.environ.items() if k not in SIGN_IN}
        env.update(sign_in)
        env["PORT"] = str(port)
        log = (cwd / "serve.log").open("wb")
        process = subprocess.Popen([sys.executable, "-c", SERVE, str(KERNEL_DIR), str(cls.program)], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        cls.servers.append((process, log))
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 90
        while time.time() < deadline:
            assert process.poll() is None, f"the server exited ({process.returncode}) before it listened:\n{(cwd / 'serve.log').read_text()}"
            try:
                urllib.request.urlopen(f"{base}/api/health", timeout=2).read()
                return base
            except OSError:
                time.sleep(0.5)
        raise AssertionError("the server did not listen within 90 seconds")

    @staticmethod
    def whoami(base: str, token=None):
        headers = {"content-type": "application/json"}
        if token is not None:
            headers["authorization"] = f"Bearer {token}"
        request = urllib.request.Request(f"{base}/api/orbitals/Who/events", data=json.dumps({"event": "ASK", "clientId": "proof-client", "payload": {}}).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=30) as reply:
                return reply.status, json.loads(reply.read())
        except urllib.error.HTTPError as refused:
            return refused.code, json.loads(refused.read())

    @staticmethod
    def viewer(body: dict) -> str:
        return next(e["payload"]["viewer"] for e in body["emittedEvents"] if e["event"] == "ANSWER")

    def test_an_emulator_token_binds_the_viewer_and_no_header_is_anonymous(self):
        base = self.serve("with-emulator", {"AUTH_PROVIDER": "firebase", "FIREBASE_PROJECT_ID": PROJECT, "FIREBASE_AUTH_EMULATOR_HOST": "127.0.0.1:9099"})
        status, body = self.whoami(base)
        self.assertEqual(status, 200, body)
        self.assertEqual(self.viewer(body), "anonymous", "control: no header is the anonymous viewer")
        status, body = self.whoami(base, emulator_token(SUBJECT))
        self.assertEqual(status, 200, body)
        self.assertEqual(self.viewer(body), SUBJECT, "the verified subject is @user")

    def test_control_an_unsigned_token_is_refused_when_no_emulator_host_is_set(self):
        base = self.serve("without-emulator", {"AUTH_PROVIDER": "firebase", "FIREBASE_PROJECT_ID": PROJECT})
        status, body = self.whoami(base, emulator_token(SUBJECT))
        self.assertEqual((status, body.get("code")), (401, "UNAUTHORIZED"))


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    unittest.main()
