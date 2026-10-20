"""The message catalogs the kernel host opens the native server with: one per locale the program declares."""
import json
from pathlib import Path

import pytest

from orbital_app.core.kernel_host import catalogs_of


def write(directory: Path, locale: str, messages: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{locale}.json").write_text(json.dumps(messages), encoding="utf-8")


def test_each_declared_locale_is_read_from_the_compile_locales(tmp_path: Path) -> None:
    write(tmp_path, "en", {"hello": "Hello"})
    write(tmp_path, "ar", {"hello": "مرحبا"})
    write(tmp_path, "sl", {"hello": "Zdravo"})
    catalogs = catalogs_of(json.dumps({"locales": ["en", "ar"]}), tmp_path)
    assert catalogs == {"en": {"hello": "Hello"}, "ar": {"hello": "مرحبا"}}


def test_a_program_without_locales_opens_with_no_catalogs(tmp_path: Path) -> None:
    assert catalogs_of(json.dumps({"name": "notes"}), tmp_path) == {}


def test_a_declared_locale_with_no_catalog_is_refused(tmp_path: Path) -> None:
    write(tmp_path, "en", {})
    with pytest.raises(RuntimeError, match='declares locale "ar"'):
        catalogs_of(json.dumps({"locales": ["en", "ar"]}), tmp_path)
