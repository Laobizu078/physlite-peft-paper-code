#!/usr/bin/env bash
# Compare baseline and query-dominant LoRA with paired data-loader randomness.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

COMMON=(
  physlite-train --data-root data/Physion --manifest data/manifests/main.csv
  --pretrained --backbone deit_small_patch16_224
  --epochs 3 --batch-size 8 --frames 8 --image-size 224
  --temporal motion_bins --sampling uniform
  --lr 0.001 --weight-decay 0.05
  --ssf-layers all --ssf-lr 0.001
  --lora-rank 4 --lora-alpha 8 --lora-layers last8 --lora-targets q
  --peft-lr 0.001 --device cuda --amp
)

run() {
  local name="$1" seed="$2"
  shift 2
  local output="outputs/paired_rng/${name}/seed_${seed}.json"
  if [[ -f "${output}" ]]; then
    echo "SKIP ${name} seed=${seed}"
    return
  fi
  echo "RUN  ${name} seed=${seed}"
  "${COMMON[@]}" --seed "${seed}" --loader-seed "${seed}" "$@" --output-json "${output}"
}

for seed in 0 1 2; do
  run baseline_d "${seed}" --method ssf_lora_qv
  run query_dominant_lora "${seed}" --method ssf_query_dominant_lora \
    --lora-value-rank 1 --lora-value-alpha 2 --lora-value-layers last4
done

python scripts/summarize_trick_multiseed.py \
  --root outputs/paired_rng \
  --candidate query_dominant_lora
