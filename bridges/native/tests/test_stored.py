"""The Python extension's plain server leaves the row ask to the host.

Run: ORBITAL_KERNEL_DIR=<dir holding orbital_kernel.abi3.so> python3 -m unittest crates/orbital-shell-python/bridges/native/tests/test_stored.py
"""
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.environ["ORBITAL_KERNEL_DIR"])
import orbital_kernel  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "stored"
SAVE = json.dumps({"event": "SAVE", "targetTrait": "Draft"})


class PlainServer(unittest.TestCase):
    def schema(self) -> str:
        return (FIXTURES / "notes.schema.json").read_text()

    def test_the_plain_server_leaves_the_row_ask_to_the_host(self):
        server = orbital_kernel.Server(self.schema())
        waiting = json.loads(server.start("Main", SAVE))
        self.assertEqual(waiting["status"], "await")
        self.assertEqual([r["request"]["kind"] for r in waiting["requests"]], ["readRow"])

    def test_a_stateful_server_refuses_carried_circuit_state_outside_a_delegated_leg(self):
        carried = [{"trait": "Draft", "from": "idle"}]
        server = orbital_kernel.Server(self.schema())
        server.set_stateful(True)
        refused = json.loads(server.start("Main", json.dumps({"event": "SAVE", "traits": carried})))
        self.assertEqual(refused["status"], "done")
        self.assertEqual(refused["response"]["rejections"][0]["code"], "client-circuit-state-refused")
        leg = json.loads(server.start("Main", json.dumps({"event": "SAVE", "targetTrait": "Draft", "traits": carried})))
        self.assertEqual(leg["status"], "await", "control: a targeted leg is the delegated leg")
        stateless = orbital_kernel.Server(self.schema())
        taken = json.loads(stateless.start("Main", json.dumps({"event": "SAVE", "traits": carried})))
        self.assertNotEqual(taken.get("response", {}).get("rejections", [{}])[0].get("code"), "client-circuit-state-refused", "control: a stateless host takes the client's copy")

    def test_serve_refuses_catalogs_that_are_not_json(self):
        with self.assertRaisesRegex(RuntimeError, "parse catalogs_json"):
            orbital_kernel.serve(self.schema(), "not json")

    def test_a_freed_server_is_an_error(self):
        server = orbital_kernel.Server(self.schema())
        server.free()
        with self.assertRaisesRegex(RuntimeError, "freed"):
            server.start("Main", SAVE)


if __name__ == "__main__":
    unittest.main()
