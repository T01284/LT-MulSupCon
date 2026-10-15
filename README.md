# LT-MulSupCon

多标签长尾对比学习栈 CLFER-LT 的组件分解。实验在 GoEmotions 与 SemEval-2018 Task 1 E-c 的官方划分上完成，编码器为 RoBERTa-base 与 BERT-base。本仓库提供训练与评估代码、配置、逐次运行的 `results.json`，以及论文用到的汇总结果。

测试集预测（`y_true` / `y_pred` / `y_prob` / `labels`）在 `outputs/predictions/` 与 `outputs/predictions_semeval/`。权重文件 `best_model.pt` 不在本包内。

## 目录

```
├── configs/                  # 训练配置（主实验、λ₁ 扫描、第二数据集、第二编码器）
├── data/                     # GoEmotions 与 SemEval-2018 官方划分
├── src/                      # 数据、模型、损失、训练
├── scripts/                  # 训练、评估、作图、重叠统计、打包
└── outputs/
    ├── key_results.json      # 正文表格所用的五种子汇总
    ├── experiment_summary.json
    ├── paper_table_fragments.json
    ├── predictions_manifest.json
    ├── analysis/             # 队列补偿、剂量与显著性汇总
    ├── figures/              # 正文与补充材料中的图
    ├── checkpoints/          # GoEmotions：每次运行一个 results.json
    ├── checkpoints_semeval/  # SemEval：每次运行一个 results.json
    ├── predictions/          # GoEmotions 测试集预测
    └── predictions_semeval/  # SemEval 测试集预测
```

## 环境

```bash
pip install -r requirements.txt
```

GoEmotions 与 SemEval-2018 的划分已在 `data/`。若需重新下载：

```bash
python scripts/download_data.py
python scripts/download_semeval2018.py
```

## 训练与评估

```bash
python scripts/train.py --config configs/default.yaml --model_type baseline --epochs 5
python scripts/train.py --config configs/default.yaml --model_type clfer_lt --epochs 5
python scripts/eval_checkpoint.py --help
python scripts/export_predictions.py --help
```

Jaccard 正样本重叠统计：

```bash
python scripts/positive_pair_overlap_stats.py
```

## 结果文件

正文数字以 `outputs/key_results.json` 为准。每次运行的测试分数在 `outputs/checkpoints/<run>/results.json` 与 `outputs/checkpoints_semeval/<run>/results.json`。队列补偿与剂量分析的汇总在 `outputs/analysis/`。

## 测试集预测

`outputs/predictions/` 与 `outputs/predictions_semeval/` 各含一次测试运行的 npz。逐文件大小与 SHA-256 见 `outputs/predictions_manifest.json`。只重新打包预测文件时：

```bash
python scripts/pack_predictions.py
```
