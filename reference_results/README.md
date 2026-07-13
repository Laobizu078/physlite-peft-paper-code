# Released result snapshot

These compact JSON files are the metrics reported by the paper. They were derived
from the original raw run JSONs with `scripts/import_legacy_results.py`; no values
were transcribed by hand. `physlite-report --verify` compares a reproduction with
the corresponding balanced-accuracy means (default tolerance: 0.02).

The snapshot is a regression target, not a replacement for raw outputs. A fresh
run writes complete per-seed metrics, predictions, environment information and
manifest checksums below `outputs/`.
