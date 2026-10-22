#!/usr/bin/env bash
# Drift gate for almadar-core-py: regenerate from @almadar/core's zod schemas and
# fail if the generated tree differs from what's committed. Idempotent generation
# (verified in Stage 1b) means a clean diff here is the actual correctness signal.
#
# NOT wired into any automated CI yet. tools/almadar-pattern-sync/.github/workflows/
# ci.yml does a STANDALONE checkout of only that submodule's own repo, so it cannot
# see orbital-rust/crates/orbital-shell-python/core/almadar-core-py (a directory in the parent monorepo, not inside
# almadar-pattern-sync's own git tree) to diff it — a step referencing
# ../../almadar-core-py there would pass locally and silently no-op in real CI.
# orbital-rust/crates/orbital-shell-python/core/almadar-core-py is not yet its own submodule with its own CI either (that
# split is a separate, later stage per docs/Almadar_Asset_Workflow_Restructure_PLAN.md
# Stage 7). Run this manually after any @almadar/core asset-schema change, or plug
# it into whichever CI ends up checking out the full monorepo (a root-level workflow,
# or almadar-core-py's own CI once Stage 7 extracts it) with one line:
#   orbital-rust/crates/orbital-shell-python/tooling/core-py-drift.sh
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../core/almadar-core-py" && pwd)"
REPO_ROOT="$(cd "$PACKAGE_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

if ! command -v datamodel-codegen >/dev/null 2>&1; then
  echo "ERROR: datamodel-codegen not on PATH (pip install datamodel-code-generator)" >&2
  exit 1
fi

node tools/almadar-pattern-sync/dist/index.js core-py

# The package lives in the orbital-shell-python repo (a submodule), so its diff is read there.
if git -C "$PACKAGE_DIR" diff --exit-code -- almadar_core_py; then
  echo "OK: orbital-rust/crates/orbital-shell-python/core/almadar-core-py is in sync with @almadar/core's asset zod schemas"
else
  echo "ERROR: orbital-rust/crates/orbital-shell-python/core/almadar-core-py is stale — re-run 'node tools/almadar-pattern-sync/dist/index.js core-py' and commit the diff" >&2
  exit 1
fi
