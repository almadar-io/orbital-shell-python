<!-- Gap ledger for this repo: the source of truth for its open gaps. -->
# orbital-shell-python — open gaps

Every open gap this repo owns lives here. This file is the source of truth; the monorepo's `docs/Almadar_Gaps.md` only rolls it up.

- **One entry per gap:** `- **<code>** — <what is wrong and where>. <owning package> [mechanical|architectural] — <evidence, prevention rung>`.
- **Codes:** new gaps use this repo's prefix `G-ORBITAL-SHELL-PYTHON-`. Take the "Next code" below, then bump it in the same edit. Codes are never reused or renamed.
- **Close by deleting.** Remove the entry in the same commit as the fix.

Next code: `G-ORBITAL-SHELL-PYTHON-011`

- **G-ORBITAL-SHELL-PYTHON-010** [mechanical] — The Python shell carries dead surfaces (R2 shell survey 2026-10-29): core/almadar-core-py (models.py 279 + _generated/asset.schema.json 602) has no importer and no live generator; bridges/native lib.rs exports a 17-method Server the template never uses (only install_language and serve; Server only in tests/test_stored.py); os_glob_matches has no caller. Retire. Root cause: T2.

- **G-ORBITAL-SHELL-PYTHON-007** — Python-only ML programs (`std-ml-infer` and the `nn`/`tensor`/`train` operators) run on no Python host. The generated host that carried them retired on 2026-10-23 (owner: retire everything at once; it called the retired `NativeServer` and translated program logic into Python, a parallel implementation), so the Python app now runs only the served program on the native kernel. They return when the kernel carries ML asks the Python host answers as raw ports (owner ruling A6, P-008). `orbital-shell-python` [architectural] — prevention rung: `orb validate` should refuse a Python-only operator when the target host carries no port for it (decidable from the operator table). Root cause: periphery.

