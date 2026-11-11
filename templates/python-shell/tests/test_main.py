"""The entry point hands the kernel the project's .env, so the served program reads its sign-in settings.

Run: python3 -m unittest discover -s tests   (no network, no native kernel: a recording stand-in is loaded)
"""
import importlib
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1]
SIGN_IN = ("AUTH_PROVIDER", "FIREBASE_PROJECT_ID", "OIDC_ISSUER_URL", "OIDC_AUDIENCE", "FIREBASE_AUTH_EMULATOR_HOST")

RECORDING_KERNEL = '''
import json, os
def install_language(language):
    os.environ["RECORDED_LANGUAGE"] = language
def serve(program):
    with open(os.environ["RECORD_TO"], "w", encoding="utf-8") as out:
        json.dump({"program": program, "language": os.environ["RECORDED_LANGUAGE"], "env": dict(os.environ)}, out)
'''


def dotenv_or_reader() -> None:
    """python-dotenv when installed; else a KEY=VALUE reader with its documented default (set only what is unset)."""
    try:
        import dotenv  # noqa: F401
        return
    except ImportError:
        pass

    def load_dotenv(path):
        if not Path(path).exists():
            return False
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip())
        return True

    reader = types.ModuleType("dotenv")
    reader.load_dotenv = load_dotenv
    sys.modules["dotenv"] = reader


class EntryPoint(unittest.TestCase):
    def setUp(self):
        self.saved = dict(os.environ)
        self.modules = set(sys.modules)
        self.path = list(sys.path)
        for name in SIGN_IN:
            os.environ.pop(name, None)
        self.root = Path(tempfile.mkdtemp(prefix="orbital_python_entry_"))
        shutil.copytree(TEMPLATE / "src", self.root / "src")
        generated = self.root / "src" / "orbital_app" / "generated"
        generated.mkdir(exist_ok=True)
        (generated / "program.orb").write_text('{"name":"app"}', encoding="utf-8")
        (generated / "language.json").write_text('{"language":1}', encoding="utf-8")
        kernel = self.root / "kernel"
        kernel.mkdir()
        (kernel / "orbital_kernel.py").write_text(RECORDING_KERNEL, encoding="utf-8")
        os.environ["ORBITAL_KERNEL_DIR"] = str(kernel)
        os.environ["RECORD_TO"] = str(self.root / "served.json")
        dotenv_or_reader()
        sys.path.insert(0, str(self.root / "src"))

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.saved)
        sys.path[:] = self.path
        for name in set(sys.modules) - self.modules:
            del sys.modules[name]
        shutil.rmtree(self.root)

    def served(self) -> dict:
        import json

        importlib.import_module("orbital_app.main").main()
        return json.loads((self.root / "served.json").read_text(encoding="utf-8"))

    def test_the_kernel_serves_the_generated_program_with_the_sign_in_settings_of_the_env_file(self):
        (self.root / ".env").write_text("AUTH_PROVIDER=firebase\nFIREBASE_PROJECT_ID=demo-project\n", encoding="utf-8")
        served = self.served()
        self.assertEqual(served["program"], '{"name":"app"}')
        self.assertEqual(served["language"], '{"language":1}')
        self.assertEqual(served["env"]["AUTH_PROVIDER"], "firebase")
        self.assertEqual(served["env"]["FIREBASE_PROJECT_ID"], "demo-project")

    def test_control_no_env_file_names_no_provider(self):
        served = self.served()
        for name in SIGN_IN:
            self.assertNotIn(name, served["env"], "the host picks no sign-in value the project did not declare")

    def test_the_process_environment_wins_over_the_env_file(self):
        (self.root / ".env").write_text("AUTH_PROVIDER=firebase\n", encoding="utf-8")
        os.environ["AUTH_PROVIDER"] = "oidc"
        self.assertEqual(self.served()["env"]["AUTH_PROVIDER"], "oidc")

    def test_the_shipped_development_env_names_the_provider_and_the_auth_emulator(self):
        shutil.copy(TEMPLATE / ".env.development", self.root / ".env")
        env = self.served()["env"]
        self.assertEqual(env["AUTH_PROVIDER"], "firebase")
        self.assertEqual(env["FIREBASE_AUTH_EMULATOR_HOST"], "localhost:9099")
        self.assertEqual(env["FIREBASE_PROJECT_ID"], "demo-project")

    def test_a_missing_kernel_is_an_error_naming_where_it_looked(self):
        os.environ["ORBITAL_KERNEL_DIR"] = str(self.root / "absent")
        with self.assertRaisesRegex(RuntimeError, "absent"):
            self.served()


if __name__ == "__main__":
    unittest.main()
