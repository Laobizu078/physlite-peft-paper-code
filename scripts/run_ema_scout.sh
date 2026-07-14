#!/usr/bin/env bash
# Scout trainable-parameter EMA settings on the fixed D-SSF-LoRA baseline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

COMMON=(
  physlite-train --data-root data/Physion --manifest data/manifests/main.csv
  --pretrained --backbone deit_small_patch16_224
  --epochs 3 --batch-size 8 --frames 8 --image-size 224
  --temporal motion_bins --sampling uniform
  --lr 0.001 --weight-decay 0.05
  --method ssf_lora_qv --ssf-layers all --ssf-lr 0.001
  --lora-rank 4 --lora-alpha 8 --lora-layers last8 --lora-targets q
  --peft-lr 0.001 --device cuda --amp --skip-test
)

run() {
  local name="$1" seed="$2" decay="$3" start="$4"
  local output="outputs/ema_scout/${name}/seed_${seed}.json"
  if [[ -f "${output}" ]]; then
    echo "SKIP ${name} seed=${seed}"
    return
  fi
  echo "RUN  ${name} seed=${seed}"
  "${COMMON[@]}" --seed "${seed}" --peft-ema-decay "${decay}" \
    --peft-ema-start-epoch "${start}" --output-json "${output}"
}

for seed in 0 1 2; do
  run ema90_start1 "${seed}" 0.90 1
  run ema95_start2 "${seed}" 0.95 2
  run ema99_start2 "${seed}" 0.99 2
done

python scripts/summarize_ema_scout.py
