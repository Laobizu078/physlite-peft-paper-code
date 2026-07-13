#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${CONDA_ENV_NAME:-cvpr_paper}"
RUN=(conda run --no-capture-output -n "${ENV_NAME}")

"${RUN[@]}" pip install -e . --no-deps
"${RUN[@]}" physlite-prepare
"${RUN[@]}" pytest -q
"${RUN[@]}" physlite-run --suite main --suite repeated --suite deit_b --suite staged
"${RUN[@]}" physlite-report --suite main --suite repeated --suite deit_b --suite staged --verify
