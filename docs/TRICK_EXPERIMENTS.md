# D-SSF-LoRA Optimization Follow-up

This note records the bounded optimization search conducted after the paper
matrix. It is intentionally separate from the paper-facing reference results:
none of the tested tricks supports a new positive claim on the DeiT-S main
split.

## Fixed protocol

- DeiT-S/224, 8 uniform uncompressed frames, `motion_bins`.
- Family-grouped main split, 3 epochs, seeds 0/1/2, CUDA AMP.
- D-SSF-LoRA: all-layer SSF plus last-8 rank-4 query LoRA.
- Primary metric: test balanced accuracy (BAcc).

## Results

| Candidate | Change from D-SSF-LoRA | Test BAcc | Paired mean change |
| --- | --- | ---: | ---: |
| D-SSF-LoRA | none | `0.7557 +/- 0.0197` | - |
| Static depth-balanced LoRA+ | A/B LR ratio 4; middle/late depth multipliers | `0.7472 +/- 0.0337` | `-0.0085` |
| Progressive depth-balanced LoRA+ | introduce the same allocation over 2 epochs | `0.7264 +/- 0.0108` | `-0.0293` |
| Validation-gated trainable EMA | decay 0.95 from epoch 2 | `0.7326 +/- 0.0347` | `-0.0232` |
| Video-consistent horizontal flip | mirror whole clips with probability 0.5 | `0.7137 +/- 0.0317` | `-0.0420` |
| Query-dominant LoRA | add rank-1 value LoRA in the final 4 blocks | `0.7476 +/- 0.0171` | `-0.0081` |

The validation-only scout initially favored static depth-balanced LoRA+ by
1.91 BAcc points at seed 0. Multi-seed testing reversed that result. EMA and
horizontal flipping showed the same failure pattern: a better validation score
could coincide with a substantially worse test score. These experiments rule
out using the single main validation split for fine-grained trick selection.

## Strict RNG audit

Adapter classes initialize different numbers of tensors, so a global RNG also
changes the later DataLoader shuffle. `--loader-seed` now permits a strictly
paired comparison with identical query initialization and batch order. Under
that audit, the rank-1 late-value candidate reached `0.7336 +/- 0.0285`, versus
`0.7380 +/- 0.0136` for D-SSF-LoRA. The additional branch therefore remains a
negative result rather than an accepted method revision.

## Reproducible accuracy recommendation

The strongest completed configuration remains the backbone-conditioned
DeiT-B allocation already included in the paper suite: all-layer SSF plus
last-4 rank-4 q/v LoRA reaches `0.7916 +/- 0.0043` BAcc. The matched DeiT-B
D-SSF-LoRA allocation reaches `0.7756 +/- 0.0197`. This result supports a
clearer rule than a universal D configuration: re-run the equal-budget support
and target allocation diagnostic when the backbone changes.

Reproduce the DeiT-B comparison with:

```bash
physlite-run --suite deit_b --only last4_r4 d_ssf_lora
physlite-report --suite deit_b --allow-incomplete
```

Reproduce the optimization follow-ups with:

```bash
bash scripts/run_trick_multiseed.sh
bash scripts/run_paired_rng_query_dominant.sh
```
