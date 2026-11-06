"""
Orbital Python host: the native kernel (the `orbital_kernel` extension) serves the program the compile wrote
under generated/; the program carries its own messages.
"""
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

KERNEL_DIR_ENV = "ORBITAL_KERNEL_DIR"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATED = Path(__file__).resolve().parent / "generated"


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
    kernel = load_extension()
    kernel.install_language(language)
    kernel.serve(program)


if __name__ == "__main__":
    main()
