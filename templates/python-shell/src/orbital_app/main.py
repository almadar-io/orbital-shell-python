"""
Orbital Python host: the native kernel (the `orbital_kernel` extension) serves the program the compile wrote
under generated/, with the message catalogs of the locales it declares.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

KERNEL_DIR_ENV = "ORBITAL_KERNEL_DIR"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATED = Path(__file__).resolve().parent / "generated"


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


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    program = (GENERATED / "program.orb").read_text(encoding="utf-8")
    language = (GENERATED / "language.json").read_text(encoding="utf-8")
    catalogs = catalogs_of(program, PROJECT_ROOT / "locales")
    kernel = load_extension()
    kernel.install_language(language)
    kernel.serve(program, json.dumps(catalogs))


if __name__ == "__main__":
    main()
