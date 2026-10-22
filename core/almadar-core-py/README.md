# almadar-core-py

GENERATED from `@almadar/core`'s asset-domain zod schemas
(`packages/almadar-core/src/types/asset.ts`) via `almadar-sync core-py`
(`tools/almadar-pattern-sync/src/commands/core-py.ts`). Pipeline: zod schemas
→ JSON Schema (`zod-to-json-schema`) → pydantic v2 (`datamodel-codegen`).

Never hand-edit `almadar_core_py/models.py` — fix the zod schema in
`@almadar/core` and re-run the generator. This is a placeholder local package
tree: a future standalone submodule extraction (a `git init` + remote add
here) is meant to be a no-op.

Regenerating requires `datamodel-codegen` on PATH:
```
pip install datamodel-code-generator
```
