#!/usr/bin/env bash
# Reproduce every paper suite using the currently activated Python environment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python -m pip install -e . --no-deps
physlite-prepare
pytest -q
physlite-run
physlite-report --verify
physlite-probe
