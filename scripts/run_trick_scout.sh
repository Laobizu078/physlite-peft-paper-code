#!/usr/bin/env bash
# Run a bounded single-seed validation scout before multi-seed confirmation.
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
  --lora-rank 4
  --lora-alpha 8
  --lora-layers last8
  --lora-targets q
  --peft-lr 0.001
  --device cuda
  --amp
  --seed 0
  --skip-test
)

run() {
  local name="$1"
  shift
  local output="outputs/trick_scout/${name}/seed_0.json"
  if [[ -f "${output}" ]]; then
    echo "SKIP ${name}"
    return
  fi
  echo "RUN  ${name}"
  "${COMMON[@]}" "$@" --output-json "${output}"
}

run baseline_d --method ssf_lora_qv --ssf-layers all --ssf-lr 0.001
run plus4_uniform --method ssf_lora_qv --ssf-layers all --ssf-lr 0.001 --lora-plus-ratio 4
run plus8_uniform --method ssf_lora_qv --ssf-layers all --ssf-lr 0.001 --lora-plus-ratio 8
run plus16_uniform --method ssf_lora_qv --ssf-layers all --ssf-lr 0.001 --lora-plus-ratio 16
run depth_only --method ssf_lora_qv --ssf-layers all --ssf-lr 0.001 --lora-depth-profile middle_focus
run plus4_depth --method ssf_lora_qv --ssf-layers all --ssf-lr 0.001 --lora-plus-ratio 4 --lora-depth-profile middle_focus
run plus8_depth --method ssf_lora_qv --ssf-layers all --ssf-lr 0.001 --lora-plus-ratio 8 --lora-depth-profile middle_focus
run plus16_depth --method ssf_lora_qv --ssf-layers all --ssf-lr 0.001 --lora-plus-ratio 16 --lora-depth-profile middle_focus
run plus8_depth_ssf3e4 --method ssf_lora_qv --ssf-layers all --ssf-lr 0.0003 --lora-plus-ratio 8 --lora-depth-profile middle_focus
run plus16_depth_ssf3e4 --method ssf_lora_qv --ssf-layers all --ssf-lr 0.0003 --lora-plus-ratio 16 --lora-depth-profile middle_focus
run plus8_depth_ssf1e4 --method ssf_lora_qv --ssf-layers all --ssf-lr 0.0001 --lora-plus-ratio 8 --lora-depth-profile middle_focus
run pure_plus8_depth --method lora_qv --lora-plus-ratio 8 --lora-depth-profile middle_focus

# One bounded local refinement around the best first-round validation point.
run plus2_depth --method ssf_lora_qv --ssf-layers all --ssf-lr 0.001 --lora-plus-ratio 2 --lora-depth-profile middle_focus
run plus3_depth --method ssf_lora_qv --ssf-layers all --ssf-lr 0.001 --lora-plus-ratio 3 --lora-depth-profile middle_focus
run plus6_depth --method ssf_lora_qv --ssf-layers all --ssf-lr 0.001 --lora-plus-ratio 6 --lora-depth-profile middle_focus
run plus4_depth_s025 --method ssf_lora_qv --ssf-layers all --ssf-lr 0.001 --lora-plus-ratio 4 --lora-depth-profile middle_focus --lora-depth-strength 0.25
run plus4_depth_s075 --method ssf_lora_qv --ssf-layers all --ssf-lr 0.001 --lora-plus-ratio 4 --lora-depth-profile middle_focus --lora-depth-strength 0.75
run plus4_depth_lr7e4 --method ssf_lora_qv --ssf-layers all --ssf-lr 0.0007 --peft-lr 0.0007 --lora-plus-ratio 4 --lora-depth-profile middle_focus
run plus4_depth_lr13e4 --method ssf_lora_qv --ssf-layers all --ssf-lr 0.0013 --peft-lr 0.0013 --lora-plus-ratio 4 --lora-depth-profile middle_focus

python scripts/summarize_trick_scout.py
