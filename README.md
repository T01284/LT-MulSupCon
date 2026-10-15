# LT-MulSupCon

Component breakdown of CLFER-LT, a contrastive stack for long-tailed multi-label emotion recognition. Experiments use the official splits of GoEmotions and SemEval-2018 Task 1 E-c, with RoBERTa-base and BERT-base as encoders. This repository provides the training and evaluation code, the configurations, a `results.json` for each run, and the aggregated results used in the paper.

Test-set predictions (`y_true`, `y_pred`, `y_prob`, and `labels`) are in `outputs/predictions/` and `outputs/predictions_semeval/`. The weight file `best_model.pt` is not included.

## Layout

```
├── configs/                  # Training configs (main runs, the λ₁ sweep, the second dataset, and the second encoder)
├── data/                     # Official GoEmotions and SemEval-2018 splits
├── src/                      # Data, model, loss, and training
├── scripts/                  # Training, evaluation, plotting, overlap statistics, and packaging
└── outputs/
    ├── key_results.json      # Five-seed aggregates used in the paper tables
    ├── experiment_summary.json
    ├── paper_table_fragments.json
    ├── predictions_manifest.json
    ├── analysis/             # Queue-compensation, dose, and significance summaries
    ├── figures/              # Figures in the manuscript and the supplement
    ├── checkpoints/          # GoEmotions: one results.json per run
    ├── checkpoints_semeval/  # SemEval: one results.json per run
    ├── predictions/          # GoEmotions test-set predictions
    └── predictions_semeval/  # SemEval test-set predictions
```

## Setup

```bash
pip install -r requirements.txt
```

The GoEmotions and SemEval-2018 splits are already in `data/`. To download them again:

```bash
python scripts/download_data.py
python scripts/download_semeval2018.py
```

## Training and evaluation

```bash
python scripts/train.py --config configs/default.yaml --model_type baseline --epochs 5
python scripts/train.py --config configs/default.yaml --model_type clfer_lt --epochs 5
python scripts/eval_checkpoint.py --help
python scripts/export_predictions.py --help
```

Jaccard positive-pair overlap:

```bash
python scripts/positive_pair_overlap_stats.py
```

## Result files

Numbers in the paper are taken from `outputs/key_results.json`. Per-run test scores are in `outputs/checkpoints/<run>/results.json` and `outputs/checkpoints_semeval/<run>/results.json`. Summaries of the queue-compensation and dose analyses are in `outputs/analysis/`.

## Test-set predictions

`outputs/predictions/` and `outputs/predictions_semeval/` each contain one npz file per test run. File sizes and SHA-256 hashes are listed in `outputs/predictions_manifest.json`. To repack only the prediction files:

```bash
python scripts/pack_predictions.py
```
