# Logged experiment outputs

## Canonical paper numbers

| File | Role |
|------|------|
| **`key_results.json`** | **Primary.** Five-seed (42–46) test means, paired deltas, hardware line (`RTX PRO 6000`, torch 2.8.0+cu128). Matches the manuscript tables. |
| `rtx6000_rerun/KEY_RESULTS.md` | Human-readable copy of the same rerun. |
| `experiment_summary.json` | Per-run `results.json` aggregate; used for **development** λ₁ grid and `update_paper_results.py`. |
| `paper_table_fragments.json` | Auto-generated manifest from `key_results.json` (run `scripts/generate_paper_tables.py`). **Not** single-seed table rows. |

Do **not** treat stale single-run rows in older snapshots as paper-aligned. When in doubt, use `key_results.json`.

`predictions/clfer_lt_l002_no_proto*_test_metrics.json` hold the five-seed LPC ablation scores exported from `checkpoints/*/results.json` (`.npz` dumps are not shipped for that run).

## Mixed-hardware archive

`experiment_summary_mixed_hardware.json` is kept for audit only; the published analysis uses the single-environment RTX PRO 6000 rerun.
