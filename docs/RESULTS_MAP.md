# Paper-to-code result map

All tables originate from per-run JSON files. The paths below are relative to
`outputs/`; `physlite-report` creates each `summary.json` and `summary.md`.

| Paper evidence | Suite / artifact | Reproduction command |
| --- | --- | --- |
| Main PEFT operator comparison | `main/` rows with axis `operator` | `physlite-run --suite main` |
| Tuning support, layer location, target, rank, and LR | `main/` axis subsets | same main command |
| D-SSF-LoRA ablations and parameter efficiency | `main/` allocation and method-ablation rows | same main command |
| Matched-seed effects and confidence intervals | `main/summary.json` → `analysis.paired_contrasts` | `physlite-report --suite main` |
| Family-level uncertainty | `main/summary.json` → `analysis.family_bootstrap` | same report command |
| Scenario gains | `main/summary.json` → `analysis.scenario_effect_d_vs_head` | same report command |
| Five repeated grouped splits | `repeated/` | `physlite-run --suite repeated` |
| DeiT-B transfer | `deit_b/` | `physlite-run --suite deit_b` |
| SSF/LoRA staged optimization | `staged/` | `physlite-run --suite staged` |
| Eight-frame observed prefix and method scout | `prefix8/` | `physlite-run --suite prefix8` |
| Sixteen-frame observed prefix | `prefix16/` | `physlite-run --suite prefix16` |
| High-resolution DeiT and DINOv2 scout | `backbone_scout/` | `physlite-run --suite backbone_scout` |
| Same-model temporal interventions | `counterfactual/probes.json` | `physlite-run --suite counterfactual && physlite-probe` |
| Linear-readout control | `readout/` | `physlite-run --suite readout` |

## Statistical units

- Main, DeiT-B, staged, prefix, backbone, and counterfactual results use the
  optimization seed as the paired unit.
- Repeated-split comparisons use the family-grouped split as the paired unit.
- Family bootstrap resamples test-set families, preserves all videos within each
  sampled family, and averages matched differences over three optimization seeds.
- Reported PEFT parameter counts subtract the shared trainable temporal head from
  the total trainable count. Peak memory is allocated CUDA memory reported by
  PyTorch, not total board occupancy.

## Name of the proposed configuration

The paper's D-SSF-LoRA row is `main/allocation_q_last8_r4`: SSF is applied to all
LayerNorms and rank-4 query-only LoRA is distributed over the final eight
transformer blocks. `main/pure_allocation_q_last8_r4` removes SSF at exactly the
same LoRA support and rank; `main/support_last8_r2` keeps the distributed support
but allocates the matched low-rank budget equally to query and value.
