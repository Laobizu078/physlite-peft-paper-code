#!/usr/bin/env bash
# Reproduce the core tables using the currently activated Python environment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python -m pip install -e . --no-deps
physlite-prepare
pytest -q
physlite-run --suite main --suite repeated --suite deit_b --suite staged
physlite-report --suite main --suite repeated --suite deit_b --suite staged --verify
