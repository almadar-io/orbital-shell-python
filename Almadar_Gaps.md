<!-- Gap ledger for this repo: the source of truth for its open gaps. -->
# orbital-shell-python — open gaps

Every open gap this repo owns lives here. This file is the source of truth; the monorepo's `docs/Almadar_Gaps.md` only rolls it up.

- **One entry per gap:** `- **<code>** — <what is wrong and where>. <owning package> [mechanical|architectural] — <evidence, prevention rung>`.
- **Codes:** new gaps use this repo's prefix `G-ORBITAL-SHELL-PYTHON-`. Take the "Next code" below, then bump it in the same edit. Codes are never reused or renamed.
- **Close by deleting.** Remove the entry in the same commit as the fix.

Next code: `G-ORBITAL-SHELL-PYTHON-011`

- **G-ORBITAL-SHELL-PYTHON-010** [mechanical] — The Python shell carries dead surfaces (R2 shell survey 2026-10-29): core/almadar-core-py (models.py 279 + _generated/asset.schema.json 602) has no importer and no live generator; bridges/native lib.rs exports a 17-method Server the template never uses (only install_language and serve; Server only in tests/test_stored.py); os_glob_matches has no caller. Retire. Update 2026-11-07 (re-verified against source): models.py gone; retire schema json, Server pymethods, os_glob_matches. Root cause: T2.


