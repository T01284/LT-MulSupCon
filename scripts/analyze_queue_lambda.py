#!/usr/bin/env python3
"""Queue effect (Full - w/o queue) across lambda_1 on SemEval and GoEmotions.

Per lambda: five-seed paired deltas, mean, 95% t-CI, one-sample t p-value.
Trend: one OLS slope of the tail delta on log(lambda_1) per seed, then a
one-sample t test on the five slopes (same procedure as the GoEmotions text).
Writes outputs/analysis/queue_lambda_<dataset>.json.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44, 45, 46)
LAMBDAS = (0.02, 0.05, 0.10, 0.20, 0.50)
FIELDS = ("tail", "macro_f1", "medium", "head", "micro_f1")


def key(base: str, seed: int) -> str:
    return base if seed == 42 else f"{base}_s{seed}"


def semeval_metrics() -> dict[str, dict]:
    out = {}
    for d in (ROOT / "outputs" / "checkpoints_semeval").glob("*/results.json"):
        out[d.parent.name] = json.loads(d.read_text(encoding="utf-8"))["test_metrics"]
    return out


def goemotions_metrics() -> dict[str, dict]:
    p = ROOT / "outputs" / "experiment_summary.json"
    if not p.exists():
        return {}
    return {k: v["test_metrics"] for k, v in json.loads(p.read_text(encoding="utf-8")).items() if "test_metrics" in v}


def pair_names(dataset: str, lam: float, metrics: dict | None = None) -> tuple[str, str]:
    tag = f"{int(round(lam * 100)):03d}"
    if dataset == "semeval":
        full = f"semeval_clfer_lt_l{tag}"
        if lam == 0.10 and metrics is not None and full not in metrics:
            full = "semeval_clfer_lt"  # earlier-hardware runs, until the same-box rerun lands
        return full, f"semeval_clfer_lt_l{tag}_no_queue"
    if lam == 0.50:
        return "clfer_lt", "clfer_lt_no_queue"
    return f"clfer_lt_l{tag}", f"clfer_lt_l{tag}_no_queue"


def ci_p(ds: list[float]) -> dict:
    m = statistics.mean(ds)
    se = statistics.stdev(ds) / math.sqrt(len(ds))
    h = float(stats.t.ppf(0.975, len(ds) - 1)) * se
    return {"mean": m, "ci": [m - h, m + h], "p": float(stats.ttest_1samp(ds, 0.0).pvalue), "deltas": ds}


def analyze(dataset: str, metrics: dict[str, dict]) -> dict:
    res: dict = {"dataset": dataset, "per_lambda": {}, "missing": []}
    for lam in LAMBDAS:
        full, nq = pair_names(dataset, lam, metrics)
        row = {}
        for f in FIELDS:
            ds = []
            for s in SEEDS:
                a, b = metrics.get(key(full, s)), metrics.get(key(nq, s))
                if a is None or b is None:
                    if f == FIELDS[0]:
                        res["missing"].append(key(full, s) if a is None else key(nq, s))
                    continue
                ds.append(float(a[f]) - float(b[f]))
            if len(ds) >= 3:
                row[f] = ci_p(ds)
        row["n"] = len(row.get("tail", {}).get("deltas", []))
        res["per_lambda"][str(lam)] = row

    slopes = []
    for s in SEEDS:
        xs, ys = [], []
        for lam in LAMBDAS:
            full, nq = pair_names(dataset, lam, metrics)
            a, b = metrics.get(key(full, s)), metrics.get(key(nq, s))
            if a and b:
                xs.append(math.log(lam))
                ys.append(float(a["tail"]) - float(b["tail"]))
        if len(xs) == len(LAMBDAS):
            slopes.append(stats.linregress(xs, ys).slope)
    if len(slopes) >= 3:
        res["tail_log_slope"] = ci_p(slopes)
    return res


def show(r: dict) -> None:
    print(f"\n=== {r['dataset']} : Full - w/o queue (5 seeds, paired) ===")
    print(f"{'lambda':>6} | {'tail delta':>10} {'95% CI':>20} {'p':>7} | {'macro delta':>11} {'95% CI':>20} {'p':>7}")
    for lam, row in r["per_lambda"].items():
        if "tail" not in row:
            print(f"{lam:>6} | missing")
            continue
        t, m = row["tail"], row["macro_f1"]
        print(
            f"{lam:>6} | {t['mean']:+.4f}    [{t['ci'][0]:+.4f},{t['ci'][1]:+.4f}] {t['p']:7.3f} | "
            f"{m['mean']:+.4f}      [{m['ci'][0]:+.4f},{m['ci'][1]:+.4f}] {m['p']:7.3f}"
        )
    if "tail_log_slope" in r:
        s = r["tail_log_slope"]
        print(f"tail-delta slope on log(lambda): {s['mean']:+.4f} CI [{s['ci'][0]:+.4f},{s['ci'][1]:+.4f}] p={s['p']:.4f}")
    if r["missing"]:
        print("missing runs:", ", ".join(r["missing"]))


def main() -> None:
    out_dir = ROOT / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, m in (("semeval", semeval_metrics()), ("goemotions", goemotions_metrics())):
        if not m:
            continue
        r = analyze(name, m)
        show(r)
        (out_dir / f"queue_lambda_{name}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
