#!/usr/bin/env bash
# Validate shortlisted PEFT optimization tricks across three random seeds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

COMMON=(
  physlite-train
  --data-root data/Physion
  --manifest data/manifests/main.csv
  --pretrained
  --backbone deit_small_patch16_224
  --epochs 3
  --batch-size 8
  --frames 8
  --image-size 224
  --temporal motion_bins
  --sampling uniform
  --lr 0.001
  --weight-decay 0.05
  --method ssf_lora_qv
  --ssf-layers all
  --ssf-lr 0.001
  --lora-rank 4
  --lora-alpha 8
  --lora-layers last8
  --lora-targets q
  --peft-lr 0.001
  --device cuda
  --amp
)

run() {
  local method="$1"
  local seed="$2"
  shift 2
  local output="outputs/trick_multiseed/${method}/seed_${seed}.json"
  if [[ -f "${output}" ]]; then
    echo "SKIP ${method} seed=${seed}"
    return
  fi
  echo "RUN  ${method} seed=${seed}"
  "${COMMON[@]}" --seed "${seed}" "$@" --output-json "${output}"
}

for seed in 0 1 2; do
  run baseline_d "${seed}"
  run depth_balanced_loraplus "${seed}" \
    --lora-plus-ratio 4 \
    --lora-depth-profile middle_focus \
    --lora-depth-strength 0.5
  run progressive_depth_balanced_loraplus "${seed}" \
    --lora-plus-ratio 4 \
    --lora-depth-profile middle_focus \
    --lora-depth-strength 0.5 \
    --lora-allocation-warmup-epochs 2
  run validation_gated_ema "${seed}" \
    --peft-ema-decay 0.95 \
    --peft-ema-start-epoch 2
  run symmetry_hflip "${seed}" \
    --train-hflip-prob 0.5
  run query_dominant_lora "${seed}" \
    --method ssf_query_dominant_lora \
    --lora-value-rank 1 \
    --lora-value-alpha 2 \
    --lora-value-layers last4
done

python scripts/summarize_trick_multiseed.py
