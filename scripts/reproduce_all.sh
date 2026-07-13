#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${CONDA_ENV_NAME:-cvpr_paper}"
RUN=(conda run --no-capture-output -n "${ENV_NAME}")

"${RUN[@]}" pip install -e . --no-deps
"${RUN[@]}" physlite-prepare
"${RUN[@]}" pytest -q
"${RUN[@]}" physlite-run
"${RUN[@]}" physlite-report --verify
"${RUN[@]}" physlite-probe
