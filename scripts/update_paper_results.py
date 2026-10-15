#!/usr/bin/env python3
"""Update main.tex tables from experiment_summary.json (dev-based lambda protocol)."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "experiment_summary.json"
TEX = ROOT / "paper" / "main.tex"
BOOT_FOCAL = ROOT / "outputs" / "bootstrap_focal_vs_l010.json"
BOOT_BASE = ROOT / "outputs" / "bootstrap_baseline_vs_l010.json"
BOOT_SEM = ROOT / "outputs" / "bootstrap_semeval_focal_vs_clfer.json"

LAMBDA_KEYS = [
    ("clfer_lt_l002", "0.02"),
    ("clfer_lt_l005", "0.05"),
    ("clfer_lt_l010", "0.10"),
    ("clfer_lt_l015", "0.15"),
    ("clfer_lt_l020", "0.20"),
    ("clfer_lt", "0.50 (default)"),
]


def fmt(v, nd=3):
    if v is None:
        return "--"
    return f"{float(v):.{nd}f}"


def signed(x: float, nd: int = 3) -> str:
    return f"{x:+.{nd}f}"


def fmt_ci(lo: float, hi: float, nd: int = 3) -> str:
    return f"$[{signed(lo, nd)},{signed(hi, nd)}]$"


def paired_deltas(a_keys: list[str], b_keys: list[str], summary: dict, field: str):
    import math
    import statistics

    ds = []
    for ak, bk in zip(a_keys, b_keys):
        if ak not in summary or bk not in summary:
            continue
        av = summary[ak]["test_metrics"].get(field)
        bv = summary[bk]["test_metrics"].get(field)
        if av is None or bv is None:
            continue
        ds.append(float(av) - float(bv))
    if len(ds) < 2:
        return None
    m = statistics.mean(ds)
    sd = statistics.stdev(ds)
    se = sd / math.sqrt(len(ds))
    tcrit = 2.7764451051977987 if len(ds) == 5 else None
    try:
        from scipy import stats as st

        p = float(st.ttest_1samp(ds, 0.0).pvalue)
        h = float(st.t.ppf(0.975, len(ds) - 1)) * se
    except Exception:
        p = None
        h = (tcrit * se) if tcrit else None
    return {"deltas": ds, "mean": m, "ci": (m - h, m + h) if h is not None else None, "p": p}


def fmt_p(p: float | None) -> str:
    if p is None:
        return "--"
    if p < 0.001:
        return "$<$0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


def mean_std(keys: list[str], summary: dict, field: str) -> str:
    vals = [
        summary[k]["test_metrics"].get(field)
        for k in keys
        if k in summary and summary[k].get("test_metrics", {}).get(field) is not None
    ]
    if not vals:
        return "--"
    if len(vals) == 1:
        return fmt(vals[0])
    import statistics

    return f"{statistics.mean(vals):.{3}f} $\\pm$ {statistics.stdev(vals):.{3}f}"


def seed_keys(base: str, summary: dict) -> list[str]:
    """Seed 42 is stored without a suffix (baseline, clfer_lt_l010, ...)."""
    keys = []
    for s in (42, 43, 44, 45, 46):
        k = base if s == 42 else f"{base}_s{s}"
        if k in summary:
            keys.append(k)
    return keys


def best_dev_epoch(history: list[dict]) -> dict:
    return max(history, key=lambda r: r.get("macro_f1", 0.0))


def replace_block(text: str, begin: str, end: str, body: str) -> str:
    """Replace content between whole-line markers (exact match, no substring collision)."""
    begin_line = re.escape(begin) + r"\s*$"
    end_line = re.escape(end) + r"\s*$"
    pattern = begin_line + r"\n.*?\n" + end_line
    block = begin + "\n" + body + "\n" + end
    new_text, n = re.subn(pattern, lambda _m: block, text, count=1, flags=re.DOTALL | re.MULTILINE)
    if n == 0:
        print(f"[warn] failed to replace block {begin} ... {end}", file=sys.stderr)
    return new_text


def main() -> None:
    if not SUMMARY.exists():
        print(f"Missing {SUMMARY}", file=sys.stderr)
        sys.exit(1)

    summary = json.load(SUMMARY.open(encoding="utf-8"))
    # λ₁=0.50 drop-one seeds 43–46 live in checkpoints but were never folded into the summary.
    for key in (
        "clfer_lt_no_proto_s43",
        "clfer_lt_no_proto_s44",
        "clfer_lt_no_proto_s45",
        "clfer_lt_no_proto_s46",
        "clfer_lt_no_supcon_s43",
        "clfer_lt_no_supcon_s44",
        "clfer_lt_no_supcon_s45",
        "clfer_lt_no_supcon_s46",
    ):
        if key not in summary:
            src = ROOT / "outputs" / "checkpoints" / key / "results.json"
            if src.exists():
                summary[key] = json.loads(src.read_text(encoding="utf-8"))

    lambda_dev_rows = []
    best_key, best_dev = "clfer_lt_l010", -1.0
    for key, lam in LAMBDA_KEYS:
        if key not in summary:
            continue
        dev = best_dev_epoch(summary[key]["history"])
        lambda_dev_rows.append(
            f"{lam} & {fmt(dev.get('macro_f1'))} & {fmt(dev.get('tail'))} & {fmt(dev.get('micro_f1'))} \\\\"
        )
        if dev["macro_f1"] > best_dev:
            best_dev = dev["macro_f1"]
            best_key = key

    grid_key = best_key
    # Isolation study is locked to λ1=0.10 (five-seed ablations). The grid is reported separately.
    best_key = "clfer_lt_l010" if "clfer_lt_l010" in summary else best_key
    selected_lam = dict(LAMBDA_KEYS).get(best_key, "0.10")
    print(f"Isolation setting: {best_key} (lambda={selected_lam}); grid winner {grid_key} dev_macro={best_dev:.4f}")

    b = summary["baseline"]["test_metrics"]
    bf = summary["baseline_focal"]["test_metrics"]
    tuned = summary[best_key]["test_metrics"]

    bert_seed_keys = seed_keys("bert_bce", summary)
    if len(bert_seed_keys) >= 3:
        main_rows = [
            "BERT+BCE$^{\\dagger}$ & "
            + " & ".join(mean_std(bert_seed_keys, summary, f) for f in ("micro_f1", "macro_f1", "mAP", "head", "tail"))
            + " \\\\",
        ]
    elif "bert_bce" in summary:
        bb = summary["bert_bce"]["test_metrics"]
        main_rows = [
            f"BERT+BCE$^{{\\dagger}}$ & {fmt(bb.get('micro_f1'))} & {fmt(bb.get('macro_f1'))} & "
            f"{fmt(bb.get('mAP'))} & {fmt(bb.get('head'))} & {fmt(bb.get('tail'))} \\\\",
        ]
    else:
        main_rows = [
            "BERT+BCE$^{\\dagger}$ & 0.588 & 0.484 & 0.509 & 0.610 & 0.389 \\\\",
        ]

    ablation_rows = []
    ablation_l050_rows = []
    l050_full_keys = seed_keys("clfer_lt", summary)
    l050_nq_keys = seed_keys("clfer_lt_no_queue", summary)
    abl_fields = ("macro_f1", "head", "medium", "tail")
    if len(l050_full_keys) >= 3:
        ablation_l050_rows.append(
            "Full & "
            + " & ".join(mean_std(l050_full_keys, summary, f) for f in abl_fields)
            + " \\\\"
        )
    elif "clfer_lt" in summary:
        m = summary["clfer_lt"]["test_metrics"]
        ablation_l050_rows.append(
            f"Full & {fmt(m.get('macro_f1'))} & {fmt(m.get('head'))} "
            f"& {fmt(m.get('medium'))} & {fmt(m.get('tail'))} \\\\"
        )
    for key, name in {
        "clfer_lt_no_supcon": "w/o LT-MulSupCon",
        "clfer_lt_no_proto": "w/o LPC",
    }.items():
        keys = seed_keys(key, summary)
        if len(keys) >= 3:
            ablation_l050_rows.append(
                f"{name} & "
                + " & ".join(mean_std(keys, summary, f) for f in abl_fields)
                + " \\\\"
            )
        elif key in summary:
            m = summary[key]["test_metrics"]
            ablation_l050_rows.append(
                f"{name}$^{{\\ddagger}}$ & {fmt(m.get('macro_f1'))} & {fmt(m.get('head'))} "
                f"& {fmt(m.get('medium'))} & {fmt(m.get('tail'))} \\\\"
            )
    if len(l050_nq_keys) >= 3:
        ablation_l050_rows.append(
            "w/o queue & "
            + " & ".join(mean_std(l050_nq_keys, summary, f) for f in abl_fields)
            + " \\\\"
        )
    elif "clfer_lt_no_queue" in summary:
        m = summary["clfer_lt_no_queue"]["test_metrics"]
        ablation_l050_rows.append(
            f"w/o queue & {fmt(m.get('macro_f1'))} & {fmt(m.get('head'))} "
            f"& {fmt(m.get('medium'))} & {fmt(m.get('tail'))} \\\\"
        )
    # Legacy single block (unused if ablation_l050_rows populated)
    for key, name in {
        "clfer_lt": "\\method{} (Full)",
        "clfer_lt_no_supcon": "w/o LT-MulSupCon",
        "clfer_lt_no_proto": "w/o LPC",
        "clfer_lt_no_queue": "w/o Memory Queue",
    }.items():
        if key not in summary:
            continue
        m = summary[key]["test_metrics"]
        ablation_rows.append(
            f"{name} & {fmt(m.get('macro_f1'))} & {fmt(m.get('head'))} "
            f"& {fmt(m.get('medium'))} & {fmt(m.get('tail'))} \\\\"
        )

    focal_seed_keys = seed_keys("baseline_focal", summary)
    hard_seed_keys = seed_keys("hard_supcon", summary)
    bce_seed_keys = seed_keys("baseline", summary)
    jaccard_seed_keys = seed_keys("clfer_lt_l010", summary)
    hardmatch_seed_keys = seed_keys("clfer_lt_l010_hard_match", summary)
    noqueue_seed_keys = seed_keys("clfer_lt_l010_no_queue", summary)
    noproto_seed_keys = seed_keys("clfer_lt_l010_no_proto", summary)
    hard_focal_seed_keys = seed_keys("hard_supcon_focal", summary)

    def _main_full_row(label: str, keys: list[str], source: str = "Reproduced") -> str | None:
        fields = ("micro_f1", "macro_f1", "mAP", "head", "tail")
        if len(keys) >= 3:
            return (
                f"{label} & "
                + " & ".join(mean_std(keys, summary, f) for f in fields)
                + f" & {source} \\\\"
            )
        if len(keys) == 1:
            m = summary[keys[0]]["test_metrics"]
            return (
                f"{label} & "
                + " & ".join(fmt(m.get(f)) for f in fields)
                + f" & {source} \\\\"
            )
        return None

    for label, keys in [
        ("R + BCE", bce_seed_keys),
        ("R + CB-Focal", focal_seed_keys),
        ("R + Hard SC", hard_seed_keys),
        ("R + Hard SC + CB-Focal", hard_focal_seed_keys),
        ("\\method{}", jaccard_seed_keys),
    ]:
        row = _main_full_row(label, keys)
        if row:
            main_rows.append(row)

    ablation_l010_rows = []
    abl_fields_l010 = ("macro_f1", "head", "medium", "tail")
    if len(jaccard_seed_keys) >= 3:
        ablation_l010_rows.append(
            "Full & "
            + " & ".join(mean_std(jaccard_seed_keys, summary, f) for f in abl_fields_l010)
            + " \\\\"
        )
    else:
        full_l010 = summary.get(best_key, summary.get("clfer_lt_l010", {})).get("test_metrics", {})
        ablation_l010_rows.append(
            f"Full & {fmt(full_l010.get('macro_f1'))} & {fmt(full_l010.get('head'))} "
            f"& {fmt(full_l010.get('medium'))} & {fmt(full_l010.get('tail'))} \\\\"
        )
    if len(noproto_seed_keys) >= 3:
        ablation_l010_rows.append(
            "w/o LPC & "
            + " & ".join(mean_std(noproto_seed_keys, summary, f) for f in abl_fields_l010)
            + " \\\\"
        )
    elif "clfer_lt_l010_no_proto" in summary:
        m = summary["clfer_lt_l010_no_proto"]["test_metrics"]
        ablation_l010_rows.append(
            f"w/o LPC$^{{\\dagger}}$ & {fmt(m.get('macro_f1'))} & {fmt(m.get('head'))} "
            f"& {fmt(m.get('medium'))} & {fmt(m.get('tail'))} \\\\"
        )
    else:
        ablation_l010_rows.append("w/o LPC & -- & -- & -- & -- \\\\")
    if len(noqueue_seed_keys) >= 3:
        ablation_l010_rows.append(
            "w/o queue & "
            + " & ".join(mean_std(noqueue_seed_keys, summary, f) for f in abl_fields_l010)
            + " \\\\"
        )
    elif "clfer_lt_l010_no_queue" in summary:
        m = summary["clfer_lt_l010_no_queue"]["test_metrics"]
        ablation_l010_rows.append(
            f"w/o queue & {fmt(m.get('macro_f1'))} & {fmt(m.get('head'))} "
            f"& {fmt(m.get('medium'))} & {fmt(m.get('tail'))} \\\\"
        )
    else:
        ablation_l010_rows.append("w/o queue & -- & -- & -- & -- \\\\")

    semeval_rows = []
    for key, name in [
        ("semeval_baseline", "RoBERTa + BCE"),
        ("semeval_baseline_focal", "RoBERTa + CB-Focal"),
        ("semeval_clfer_lt", f"\\method{{}} ($\\lambda_1{{=}}{selected_lam}$)"),
    ]:
        if key not in summary:
            continue
        m = summary[key]["test_metrics"]
        semeval_rows.append(
            f"{name} & {fmt(m.get('micro_f1'))} & {fmt(m.get('macro_f1'))} & {fmt(m.get('mAP'))} \\\\"
        )

    semeval_group_rows = []
    for base, name in [
        ("semeval_baseline", "RoBERTa + BCE"),
        ("semeval_baseline_focal", "RoBERTa + CB-Focal"),
        ("semeval_clfer_lt", f"\\method{{}}"),
    ]:
        sk = seed_keys(base, summary)
        if len(sk) >= 3:
            semeval_group_rows.append(
                f"{name} & {mean_std(sk, summary, 'head')} & "
                f"{mean_std(sk, summary, 'medium')} & {mean_std(sk, summary, 'tail')} \\\\"
            )
        elif base in summary:
            m = summary[base]["test_metrics"]
            semeval_group_rows.append(
                f"{name} & {fmt(m.get('head'))} & {fmt(m.get('medium'))} & {fmt(m.get('tail'))} \\\\"
            )

    rare_classes = ["anticipation", "trust", "surprise"]
    semeval_rare_rows = []
    for cls in rare_classes:
        cells = [cls.replace("_", "\\_")]
        for base in ["semeval_baseline", "semeval_baseline_focal", "semeval_clfer_lt"]:
            sk = seed_keys(base, summary)
            vals = [
                summary[k].get("per_class_f1", {}).get(cls)
                for k in sk
                if summary[k].get("per_class_f1", {}).get(cls) is not None
            ]
            if len(vals) >= 3:
                import statistics as _st

                cells.append(f"{_st.mean(vals):.3f} $\\pm$ {_st.stdev(vals):.3f}")
            else:
                pc = summary.get(base, {}).get("per_class_f1", {})
                cells.append(fmt(pc.get(cls)))
        semeval_rare_rows.append(" & ".join(cells) + " \\\\")

    if len(bce_seed_keys) >= 3:
        bce_line = (
            f"R + BCE & {mean_std(bce_seed_keys, summary, 'macro_f1')} & "
            f"{mean_std(bce_seed_keys, summary, 'tail')} \\\\"
        )
    else:
        bce_line = "R + BCE & -- & -- \\\\"

    if any(k in summary for k in focal_seed_keys):
        focal_line = (
            f"R + CB-Focal & {mean_std(focal_seed_keys, summary, 'macro_f1')} & "
            f"{mean_std(focal_seed_keys, summary, 'tail')} \\\\"
        )
    else:
        focal_line = "R + CB-Focal & -- & -- \\\\"

    if len(hard_seed_keys) >= 3:
        hard_line = (
            f"R + Hard SC & {mean_std(hard_seed_keys, summary, 'macro_f1')} & "
            f"{mean_std(hard_seed_keys, summary, 'tail')} \\\\"
        )
    else:
        hard_line = "R + Hard SC & -- & -- \\\\"

    if len(hard_focal_seed_keys) >= 3:
        hard_focal_line = (
            f"R + Hard SC + CB-Focal & {mean_std(hard_focal_seed_keys, summary, 'macro_f1')} & "
            f"{mean_std(hard_focal_seed_keys, summary, 'tail')} \\\\"
        )
    else:
        hard_focal_line = "R + Hard SC + CB-Focal & -- & -- \\\\"

    if len(jaccard_seed_keys) >= 3:
        jaccard_line = (
            f"Jaccard (LT-MulSupCon) & {mean_std(jaccard_seed_keys, summary, 'macro_f1')} & "
            f"{mean_std(jaccard_seed_keys, summary, 'tail')} \\\\"
        )
    else:
        full_l010 = summary.get(best_key, summary.get("clfer_lt_l010", {})).get("test_metrics", {})
        jaccard_line = (
            f"Jaccard (LT-MulSupCon) & {fmt(full_l010.get('macro_f1'))} & "
            f"{fmt(full_l010.get('tail'))} \\\\"
        )

    if len(hardmatch_seed_keys) >= 3:
        exact_line = (
            f"Exact match (hard) & {mean_std(hardmatch_seed_keys, summary, 'macro_f1')} & "
            f"{mean_std(hardmatch_seed_keys, summary, 'tail')} \\\\"
        )
    elif "clfer_lt_l010_hard_match" in summary:
        hm = summary["clfer_lt_l010_hard_match"]["test_metrics"]
        exact_line = (
            f"Exact match (hard) & {fmt(hm.get('macro_f1'))} & {fmt(hm.get('tail'))} \\\\"
        )
    else:
        exact_line = "Exact match (hard) & -- & -- \\\\"

    if len(noqueue_seed_keys) >= 3:
        noqueue_line = (
            f"w/o queue & {mean_std(noqueue_seed_keys, summary, 'macro_f1')} & "
            f"{mean_std(noqueue_seed_keys, summary, 'tail')} \\\\"
        )
    elif "clfer_lt_l010_no_queue" in summary:
        nq = summary["clfer_lt_l010_no_queue"]["test_metrics"]
        noqueue_line = (
            f"w/o queue & {fmt(nq.get('macro_f1'))} & {fmt(nq.get('tail'))} \\\\"
        )
    else:
        noqueue_line = "w/o queue & -- & -- \\\\"

    if len(noproto_seed_keys) >= 3:
        noproto_line = (
            f"w/o LPC & {mean_std(noproto_seed_keys, summary, 'macro_f1')} & "
            f"{mean_std(noproto_seed_keys, summary, 'tail')} \\\\"
        )
    elif "clfer_lt_l010_no_proto" in summary:
        np_ = summary["clfer_lt_l010_no_proto"]["test_metrics"]
        noproto_line = (
            f"w/o LPC & {fmt(np_.get('macro_f1'))} & {fmt(np_.get('tail'))} \\\\"
        )
    else:
        noproto_line = "w/o LPC & -- & -- \\\\"

    multiseed_rows = [
        bce_line,
        hard_line,
        hard_focal_line,
        focal_line,
        jaccard_line,
        noproto_line,
        noqueue_line,
        exact_line,
    ]

    stack_rows = [
        f"RoBERTa + CB-Focal & {mean_std(focal_seed_keys, summary, 'macro_f1')} & "
        f"{mean_std(focal_seed_keys, summary, 'tail')} \\\\",
        f"{{}}+ hard SupCon & {mean_std(hard_focal_seed_keys, summary, 'macro_f1')} & "
        f"{mean_std(hard_focal_seed_keys, summary, 'tail')} \\\\",
        f"{{}}+ Jaccard LT-MulSupCon + LPC (no queue) & {mean_std(noqueue_seed_keys, summary, 'macro_f1')} & "
        f"{mean_std(noqueue_seed_keys, summary, 'tail')} \\\\",
        f"{{}}+ queue, $K{{=}}4096$ (full \\method{{}}) & {mean_std(jaccard_seed_keys, summary, 'macro_f1')} & "
        f"{mean_std(jaccard_seed_keys, summary, 'tail')} \\\\",
    ]

    paired_rows = []
    q_m = paired_deltas(jaccard_seed_keys, noqueue_seed_keys, summary, "macro_f1")
    q_t = paired_deltas(jaccard_seed_keys, noqueue_seed_keys, summary, "tail")
    p_m = paired_deltas(jaccard_seed_keys, noproto_seed_keys, summary, "macro_f1")
    p_t = paired_deltas(jaccard_seed_keys, noproto_seed_keys, summary, "tail")
    j_m = paired_deltas(jaccard_seed_keys, hardmatch_seed_keys, summary, "macro_f1")
    j_t = paired_deltas(jaccard_seed_keys, hardmatch_seed_keys, summary, "tail")
    if q_m and q_t and p_m and p_t and j_m and j_t:
        seeds = (42, 43, 44, 45, 46)
        for i, seed in enumerate(seeds):
            paired_rows.append(
                f"{seed} & {signed(q_m['deltas'][i])} & {signed(q_t['deltas'][i])} & "
                f"{signed(p_m['deltas'][i])} & {signed(p_t['deltas'][i])} & "
                f"{signed(j_m['deltas'][i])} & {signed(j_t['deltas'][i])} \\\\"
            )
        paired_mean = (
            f"Mean & {signed(q_m['mean'])} & {signed(q_t['mean'])} & "
            f"{signed(p_m['mean'])} & {signed(p_t['mean'])} & "
            f"{signed(j_m['mean'])} & {signed(j_t['mean'])} \\\\"
        )
        paired_p = (
            f"Paired $t$ $p$ & {fmt_p(q_m['p'])} & {fmt_p(q_t['p'])} & "
            f"{fmt_p(p_m['p'])} & {fmt_p(p_t['p'])} & {fmt_p(j_m['p'])} & {fmt_p(j_t['p'])} \\\\"
        )
        print(
            "paired CIs:",
            "queue", q_m["ci"], q_t["ci"],
            "lpc", p_m["ci"], p_t["ci"],
            "jac", j_m["ci"], j_t["ci"],
        )
    else:
        paired_mean = paired_p = ""

    queue_lambda_rows = []
    component_lambda_rows = []

    def _append_pair_rows(label: str, full_k: list[str], comp_k: list[str], lam: str, n_label: str | None = None):
        qm = paired_deltas(full_k, comp_k, summary, "macro_f1")
        qt = paired_deltas(full_k, comp_k, summary, "tail")
        if not (qm and qt):
            return
        pos = sum(1 for d in qt["deltas"] if d > 1e-6)
        n_show = n_label or f"{pos}/{len(qt['deltas'])}"
        if qt["ci"]:
            ci_str = fmt_ci(qt["ci"][0], qt["ci"][1], nd=4)
            p_str = fmt_p(qt["p"])
        else:
            ci_str = "--"
            p_str = "--"
        row = (
            f"{label} & {lam} & {signed(qm['mean'])} & {signed(qt['mean'])} & "
            f"{ci_str} & {p_str} & {n_show} \\\\"
        )
        component_lambda_rows.append(row)
        if label == "Queue":
            queue_lambda_rows.append(
                f"{lam} & {signed(qm['mean'])} & {signed(qt['mean'])} & "
                f"{ci_str} & {p_str} & {n_show} \\\\"
            )
            print(f"queue@{lam} tail {qt['mean']:.4f} CI {qt['ci']} p {qt['p']} pos {pos}/{len(qt['deltas'])}")

    for lam, base in [("0.02", "clfer_lt_l002"), ("0.05", "clfer_lt_l005"), ("0.10", "clfer_lt_l010")]:
        _append_pair_rows("Queue", seed_keys(base, summary), seed_keys(f"{base}_no_queue", summary), lam)

    l020_full = seed_keys("clfer_lt_l020", summary)
    l020_nq = seed_keys("clfer_lt_l020_no_queue", summary)
    if len(l020_full) >= 3 and len(l020_nq) >= 3:
        _append_pair_rows("Queue", l020_full, l020_nq, "0.20")

    l050_full = seed_keys("clfer_lt", summary)
    l050_nq = seed_keys("clfer_lt_no_queue", summary)
    if len(l050_full) >= 3 and len(l050_nq) >= 3:
        _append_pair_rows("Queue", l050_full, l050_nq, "0.50")

    l050_full_hm = seed_keys("clfer_lt", summary)
    l050_hm = seed_keys("clfer_lt_hard_match", summary)

    for lam, base in [("0.02", "clfer_lt_l002"), ("0.10", "clfer_lt_l010")]:
        _append_pair_rows(
            "Jaccard",
            seed_keys(base, summary),
            seed_keys(f"{base}_hard_match", summary),
            lam,
        )

    if len(l050_full_hm) >= 3 and len(l050_hm) >= 3:
        _append_pair_rows("Jaccard", l050_full_hm, l050_hm, "0.50")
        jm = paired_deltas(l050_full_hm, l050_hm, summary, "tail")
        if jm:
            print(f"jaccard@0.50 tail {jm['mean']:.4f} CI {jm['ci']} p {jm['p']}")

    for lam, base in [("0.02", "clfer_lt_l002"), ("0.10", "clfer_lt_l010")]:
        _append_pair_rows(
            "LPC",
            seed_keys(base, summary),
            seed_keys(f"{base}_no_proto", summary),
            lam,
        )

    interaction_rows = []
    for high, low, tag in [
        ("clfer_lt_l010", "clfer_lt_l005", "0.10 vs 0.05"),
        ("clfer_lt_l010", "clfer_lt_l002", "0.10 vs 0.02"),
    ]:
        ds = []
        for s in (42, 43, 44, 45, 46):
            fk = high if s == 42 else f"{high}_s{s}"
            nk = f"{high}_no_queue" if s == 42 else f"{high}_no_queue_s{s}"
            fk_l = low if s == 42 else f"{low}_s{s}"
            nk_l = f"{low}_no_queue" if s == 42 else f"{low}_no_queue_s{s}"
            if all(k in summary for k in (fk, nk, fk_l, nk_l)):
                ds.append(
                    (float(summary[fk]["test_metrics"]["tail"]) - float(summary[nk]["test_metrics"]["tail"]))
                    - (float(summary[fk_l]["test_metrics"]["tail"]) - float(summary[nk_l]["test_metrics"]["tail"]))
                )
        if len(ds) >= 3:
            import statistics as _st
            import math as _math
            m = _st.mean(ds)
            sd = _st.stdev(ds)
            se = sd / _math.sqrt(len(ds))
            try:
                from scipy import stats as _stats
                p = float(_stats.ttest_1samp(ds, 0.0).pvalue)
                h = float(_stats.t.ppf(0.975, len(ds) - 1)) * se
            except Exception:
                p = None
                h = 2.7764451051977987 * se if len(ds) == 5 else None
            interaction_rows.append(
                f"{tag} & {signed(m)} & $[{signed(m - h)},{signed(m + h)}]$ & {p:.3f} \\\\"
            )
            print(f"interaction {tag} mean {m:.4f} p {p}")

    full_vs_focal_rows = []
    for lam, base in [
        ("0.02", "clfer_lt_l002"),
        ("0.05", "clfer_lt_l005"),
        ("0.10", "clfer_lt_l010"),
        ("0.20", "clfer_lt_l020"),
        ("0.50", "clfer_lt"),
    ]:
        fk = seed_keys(base, summary)
        bf = seed_keys("baseline_focal", summary)
        if len(fk) < 3 or len(bf) < 3:
            continue
        qm = paired_deltas(fk, bf, summary, "macro_f1")
        qt = paired_deltas(fk, bf, summary, "tail")
        if not (qm and qt and qm["ci"] and qt["ci"]):
            continue
        full_vs_focal_rows.append(
            f"{lam} & {signed(qm['mean'])} & {fmt_ci(qm['ci'][0], qm['ci'][1], nd=4)} & "
            f"{signed(qt['mean'])} & {fmt_ci(qt['ci'][0], qt['ci'][1], nd=4)} & {fmt_p(qt['p'])} \\\\"
        )
        print(f"full-focal@{lam} macro {qm['mean']:.4f} tail {qt['mean']:.4f}")

    log_trend_line = ""
    trend_seeds = (42, 43, 44, 45, 46)
    trend_pairs = [
        (0.02, "clfer_lt_l002"),
        (0.05, "clfer_lt_l005"),
        (0.10, "clfer_lt_l010"),
        (0.20, "clfer_lt_l020"),
        (0.50, "clfer_lt"),
    ]
    per_seed_slopes: list[float] = []
    import math as _math

    for sd in trend_seeds:
        xs, ys = [], []
        for lam, base in trend_pairs:
            fk = base if sd == 42 else f"{base}_s{sd}"
            if base == "clfer_lt":
                nk = "clfer_lt_no_queue" if sd == 42 else f"clfer_lt_no_queue_s{sd}"
            else:
                nk = f"{base}_no_queue" if sd == 42 else f"{base}_no_queue_s{sd}"
            if fk not in summary or nk not in summary:
                continue
            xs.append(_math.log(lam))
            ys.append(
                float(summary[fk]["test_metrics"]["tail"]) - float(summary[nk]["test_metrics"]["tail"])
            )
        if len(xs) == len(trend_pairs):
            n = len(xs)
            xbar = sum(xs) / n
            ybar = sum(ys) / n
            sxx = sum((x - xbar) ** 2 for x in xs)
            sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
            per_seed_slopes.append(sxy / sxx if sxx else 0.0)
    if len(per_seed_slopes) == 5:
        import statistics as _st
        import math as _math2

        m_slope = _st.mean(per_seed_slopes)
        sd_slope = _st.stdev(per_seed_slopes)
        se_slope = sd_slope / _math2.sqrt(len(per_seed_slopes))
        tcrit = 2.7764451051977987
        lo, hi = m_slope - tcrit * se_slope, m_slope + tcrit * se_slope
        try:
            from scipy import stats as _stats

            p_slope = float(_stats.ttest_1samp(per_seed_slopes, 0.0).pvalue)
        except Exception:
            p_slope = None
        log_trend_line = (
            "We fit one OLS slope per seed across the five $\\lambda_1$ settings and test the five slopes "
            f"with a one-sample $t$ test ($\\mathrm{{df}}{{=}}4$); slope ${signed(m_slope)}$ "
            f"(95\\% CI $[{signed(lo)},{signed(hi)}]$, $p={fmt_p(p_slope)}$)."
        )
        print(f"log trend slope {m_slope:.4f} CI ({lo:.4f},{hi:.4f}) p {p_slope}")

    per_class = summary.get(best_key, {}).get("per_class_f1", {})
    focal_pc = summary.get("baseline_focal", {}).get("per_class_f1", {})
    train_counts = {
        "neutral": 14219, "admiration": 4130, "approval": 2939, "gratitude": 2662,
        "annoyance": 2470, "amusement": 2328, "curiosity": 2191, "love": 2086,
        "disapproval": 2022, "optimism": 1581, "anger": 1567, "joy": 1452,
        "confusion": 1368, "sadness": 1326, "disappointment": 1269, "realization": 1110,
        "caring": 1087, "surprise": 1060, "excitement": 853, "disgust": 793,
        "desire": 641, "fear": 596, "remorse": 545, "embarrassment": 303,
        "nervousness": 164, "relief": 153, "pride": 111, "grief": 77,
    }
    labels_sorted = sorted(
        set(per_class) | set(focal_pc) | set(train_counts),
        key=lambda x: train_counts.get(x, 0),
        reverse=True,
    )
    full_f1_rows = [
        f"{lab} & {train_counts.get(lab, 0)} & {fmt(focal_pc.get(lab))} & {fmt(per_class.get(lab))} \\\\"
        for lab in labels_sorted
        if lab in train_counts
    ]

    def split_main_row(row: str) -> tuple[str, str]:
        body = row.strip().removesuffix("\\\\").strip()
        parts = [part.strip() for part in body.split("&")]
        label = parts[0]
        if len(parts) < 6:
            raise ValueError(f"Unexpected main-results row: {row!r}")
        row_a = f"{label} & {parts[1]} & {parts[2]} & {parts[3]} \\\\"
        row_b = f"{label} & {parts[4]} & {parts[5]} \\\\"
        return row_a, row_b

    main_rows_a: list[str] = []
    main_rows_b: list[str] = []
    for row in main_rows:
        row_a, row_b = split_main_row(row)
        main_rows_a.append(row_a)
        main_rows_b.append(row_b)

    text = TEX.read_text(encoding="utf-8")
    text = replace_block(text, "% BEGIN_MAIN_RESULTS_A", "% END_MAIN_RESULTS_A", "\n".join(main_rows_a))
    text = replace_block(text, "% BEGIN_MAIN_RESULTS_B", "% END_MAIN_RESULTS_B", "\n".join(main_rows_b))
    text = replace_block(text, "% BEGIN_ABLATION_L010", "% END_ABLATION_L010", "\n".join(ablation_l010_rows))
    l050_body = "\n".join(ablation_l050_rows) if ablation_l050_rows else "\n".join(ablation_rows)
    text = replace_block(text, "% BEGIN_ABLATION_L050", "% END_ABLATION_L050", l050_body)
    text = replace_block(text, "% BEGIN_LAMBDA_DEV", "% END_LAMBDA_DEV", "\n".join(lambda_dev_rows))
    text = replace_block(text, "% BEGIN_SEMEVAL", "% END_SEMEVAL", "\n".join(semeval_rows))
    text = replace_block(text, "% BEGIN_SEMEVAL_GROUP", "% END_SEMEVAL_GROUP", "\n".join(semeval_group_rows))
    text = replace_block(text, "% BEGIN_SEMEVAL_RARE", "% END_SEMEVAL_RARE", "\n".join(semeval_rare_rows))
    text = replace_block(text, "% BEGIN_MULTISEED", "% END_MULTISEED", "\n".join(multiseed_rows))
    text = replace_block(text, "% BEGIN_STACK", "% END_STACK", "\n".join(stack_rows))
    if paired_rows:
        paired_body = "\n".join(paired_rows + ["\\midrule", paired_mean, paired_p])
        text = replace_block(text, "% BEGIN_PAIRED", "% END_PAIRED", paired_body)
    if queue_lambda_rows:
        text = replace_block(text, "% BEGIN_QUEUE_LAMBDA", "% END_QUEUE_LAMBDA", "\n".join(queue_lambda_rows))
    if component_lambda_rows:
        text = replace_block(
            text, "% BEGIN_COMPONENT_LAMBDA", "% END_COMPONENT_LAMBDA", "\n".join(component_lambda_rows)
        )
    if interaction_rows:
        text = replace_block(text, "% BEGIN_INTERACTION", "% END_INTERACTION", "\n".join(interaction_rows))
    if log_trend_line:
        text = replace_block(text, "% BEGIN_LOG_TREND", "% END_LOG_TREND", log_trend_line)
    if full_vs_focal_rows:
        text = replace_block(
            text, "% BEGIN_FULL_VS_FOCAL", "% END_FULL_VS_FOCAL", "\n".join(full_vs_focal_rows)
        )
    text = replace_block(text, "% BEGIN_FULL_F1", "% END_FULL_F1", "\n".join(full_f1_rows))

    TEX.write_text(text, encoding="utf-8")
    print(f"Updated {TEX}")


if __name__ == "__main__":
    main()
