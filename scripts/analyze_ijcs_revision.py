#!/usr/bin/env python3
"""Full analysis of the IJCS revision experiments (A1, A2, E1-E6).

Paired five/ten-seed deltas with 95% t-CIs and one-sample t tests; Holm
correction across the lambda settings of each family; per-seed OLS slopes on
log(lambda) or log(queue size). Writes outputs/analysis/ijcs_revision_report.md
and ijcs_revision_stats.json.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis"
S5 = (42, 43, 44, 45, 46)
S_EXTRA = (47, 48, 49, 50, 51)
S10 = S5 + S_EXTRA
LAMS = ("002", "005", "010", "020", "050")
LAMV = {t: int(t) / 100 for t in LAMS}

lines: list[str] = []
stats_out: dict = {}


def emit(s: str = "") -> None:
    lines.append(s)
    print(s)


def load() -> dict[str, dict]:
    m: dict[str, dict] = {}
    summ = ROOT / "outputs" / "experiment_summary.json"
    if summ.exists():
        for k, v in json.loads(summ.read_text(encoding="utf-8")).items():
            if "test_metrics" in v:
                m[k] = v["test_metrics"]
    for sub in ("checkpoints", "checkpoints_semeval"):
        for p in (ROOT / "outputs" / sub).glob("*/results.json"):
            m[p.parent.name] = json.loads(p.read_text(encoding="utf-8"))["test_metrics"]
    return m


M = load()


def k(base: str, s: int) -> str:
    return base if s == 42 else f"{base}_s{s}"


def deltas(a: str, b: str, seeds, field: str = "tail") -> list[float]:
    out = []
    for s in seeds:
        x, y = M.get(k(a, s)), M.get(k(b, s))
        if x is not None and y is not None:
            out.append(float(x[field]) - float(y[field]))
    return out


def summarize(ds: list[float]) -> dict | None:
    if len(ds) < 3:
        return None
    m = statistics.mean(ds)
    se = statistics.stdev(ds) / math.sqrt(len(ds))
    h = float(stats.t.ppf(0.975, len(ds) - 1)) * se
    p = float(stats.ttest_1samp(ds, 0.0).pvalue) if se > 0 else 1.0
    return {"n": len(ds), "mean": m, "lo": m - h, "hi": m + h, "p": p}


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def fmt(r: dict | None, holm_p: float | None = None) -> str:
    if r is None:
        return "n/a"
    p = f"{r['p']:.3f}" if r["p"] >= 0.001 else "<0.001"
    hp = "" if holm_p is None else (f", Holm {holm_p:.3f}" if holm_p >= 0.001 else ", Holm <0.001")
    return f"{r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] (n={r['n']}, p={p}{hp})"


def slope(pairs: list[tuple[float, str, str]], seeds, field: str = "tail") -> dict | None:
    """pairs: (x, full_base, comp_base); x is already log-transformed."""
    slopes = []
    for s in seeds:
        xs, ys = [], []
        for x, a, b in pairs:
            d = deltas(a, b, (s,), field)
            if d:
                xs.append(x)
                ys.append(d[0])
        if len(xs) == len(pairs):
            slopes.append(stats.linregress(xs, ys).slope)
    return summarize(slopes)


def family(title: str, rows: list[tuple[str, str, str]], seeds, key: str) -> None:
    """rows: (label, full_base, comp_base). Prints tail and macro deltas with Holm across rows."""
    emit(f"\n#### {title}\n")
    emit("| setting | tail Δ [95% CI] | macro-F1 Δ [95% CI] |")
    emit("|---|---|---|")
    tails = [summarize(deltas(a, b, seeds, "tail")) for _, a, b in rows]
    macs = [summarize(deltas(a, b, seeds, "macro_f1")) for _, a, b in rows]
    ok = [i for i, r in enumerate(tails) if r]
    th = holm([tails[i]["p"] for i in ok]) if ok else []
    mh = holm([macs[i]["p"] for i in ok if macs[i]]) if ok else []
    hmap_t = dict(zip(ok, th))
    hmap_m = dict(zip([i for i in ok if macs[i]], mh))
    for i, (label, _, _) in enumerate(rows):
        emit(f"| {label} | {fmt(tails[i], hmap_t.get(i))} | {fmt(macs[i], hmap_m.get(i))} |")
    stats_out[key] = {label: {"tail": tails[i], "macro_f1": macs[i]} for i, (label, _, _) in enumerate(rows)}


def mean_abs(base: str, seeds, field: str) -> str:
    vals = [float(M[k(base, s)][field]) for s in seeds if k(base, s) in M]
    if len(vals) < 2:
        return "n/a"
    return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"


# ---------------------------------------------------------------- names
def sem_full(t: str) -> str:
    return f"semeval_clfer_lt_l{t}"


def goe_full(t: str) -> str:
    return "clfer_lt" if t == "050" else f"clfer_lt_l{t}"


def goe_variant(t: str, v: str) -> str:
    if t == "050":
        return {"no_queue": "clfer_lt_no_queue", "hard_match": "clfer_lt_hard_match", "no_proto": "clfer_lt_no_proto"}[v]
    return f"clfer_lt_l{t}_{v}"


def main() -> None:
    emit("# IJCS revision — full analysis\n")
    emit("Δ = Full − variant, paired by seed; tail = tail-group macro-F1. 95% t-CI; Holm across settings in each table.")

    # S0 hardware
    emit("\n## 0. Hardware check (same config and seed, old box vs new box)\n")
    hw_t = summarize(deltas("semeval_clfer_lt", "semeval_clfer_lt_l010", S5, "tail"))
    hw_m = summarize(deltas("semeval_clfer_lt", "semeval_clfer_lt_l010", S5, "macro_f1"))
    emit(f"SemEval λ₁=0.10 Full, old − new: tail {fmt(hw_t)}; macro {fmt(hw_m)}")
    absd = [abs(d) for d in deltas("semeval_clfer_lt", "semeval_clfer_lt_l010", S5, "tail")]
    emit(f"mean |tail difference| per seed = {statistics.mean(absd):.4f}" if absd else "")
    stats_out["hardware"] = {"tail": hw_t, "macro_f1": hw_m}

    # S1 main grid
    emit("\n## 1. Queue effect across λ₁ (batch 32)\n")
    for name, full, seedsets in (
        ("SemEval", sem_full, (("seeds 42–46", S5), ("seeds 42–51", S10))),
        ("GoEmotions", goe_full, (("seeds 42–46", S5), ("seeds 42–51", S10))),
    ):
        nq = (lambda t: f"semeval_clfer_lt_l{t}_no_queue") if name == "SemEval" else (lambda t: goe_variant(t, "no_queue"))
        for lab, seeds in seedsets:
            rows = [(f"λ₁={LAMV[t]:.2f}", full(t), nq(t)) for t in LAMS]
            family(f"{name}, {lab}", rows, seeds, f"queue_{name}_{lab}")
            sl = slope([(math.log(LAMV[t]), full(t), nq(t)) for t in LAMS], seeds)
            emit(f"\nTail-Δ slope on log λ₁ ({lab}): {fmt(sl)}")
            stats_out[f"slope_{name}_{lab}"] = sl

    # S2 batch
    emit("\n## 2. Batch size (in-batch tail positives)\n")
    for b, lams in (("b8", LAMS), ("b16", ("002", "010", "050"))):
        rows = [(f"λ₁={LAMV[t]:.2f}", f"semeval_{b}_l{t}", f"semeval_{b}_l{t}_no_queue") for t in lams]
        family(f"SemEval batch {b[1:]}", rows, S5, f"batch_semeval_{b}")
        sl = slope([(math.log(LAMV[t]), f"semeval_{b}_l{t}", f"semeval_{b}_l{t}_no_queue") for t in lams], S5)
        emit(f"\nTail-Δ slope on log λ₁: {fmt(sl)}")
        stats_out[f"slope_semeval_{b}"] = sl
    emit("\nSemEval λ₁=0.50 tail Δ by batch size: " + "; ".join(
        f"b{bs}: {fmt(summarize(deltas(a, b_, S5)))}"
        for bs, a, b_ in ((8, "semeval_b8_l050", "semeval_b8_l050_no_queue"),
                          (16, "semeval_b16_l050", "semeval_b16_l050_no_queue"),
                          (32, "semeval_clfer_lt_l050", "semeval_clfer_lt_l050_no_queue"))))

    rows = [
        ("λ₁=0.10, batch 32 (seeds 42–46)", "clfer_lt_l010", "clfer_lt_l010_no_queue"),
        ("λ₁=0.10, batch 128", "goemo_b128_l010", "goemo_b128_l010_no_queue"),
        ("λ₁=0.50, batch 32 (seeds 42–46)", "clfer_lt", "clfer_lt_no_queue"),
        ("λ₁=0.50, batch 64", "goemo_b64_l050", "goemo_b64_l050_no_queue"),
        ("λ₁=0.50, batch 128", "goemo_b128_l050", "goemo_b128_l050_no_queue"),
    ]
    family("GoEmotions batch size", rows, S5, "batch_goemo")
    emit("\nAbsolute Full test macro-F1 (mean ± SD): " + "; ".join(
        f"{lab}: {mean_abs(a, S5, 'macro_f1')}" for lab, a, _ in rows))

    # S3 queue size
    emit("\n## 3. Queue capacity at λ₁=0.50\n")
    rows = [
        ("queue 1024", "goemo_q1024_l050", "clfer_lt_no_queue"),
        ("queue 4096 (default)", "clfer_lt", "clfer_lt_no_queue"),
        ("queue 16384", "goemo_q16384_l050", "clfer_lt_no_queue"),
    ]
    family("GoEmotions (seeds 47–51, vs w/o queue)", rows, S_EXTRA, "qsize_goemo")
    sl = slope([(math.log(q), a, b) for q, (_, a, b) in zip((1024, 4096, 16384), rows)], S_EXTRA)
    emit(f"\nTail-Δ slope on log queue size: {fmt(sl)}")
    stats_out["slope_qsize_goemo"] = sl
    rows = [
        ("queue 512", "semeval_q512_l050", "semeval_clfer_lt_l050_no_queue"),
        ("queue 2048 (default)", "semeval_clfer_lt_l050", "semeval_clfer_lt_l050_no_queue"),
        ("queue 8192", "semeval_q8192_l050", "semeval_clfer_lt_l050_no_queue"),
    ]
    family("SemEval (seeds 42–46, vs w/o queue)", rows, S5, "qsize_semeval")
    sl = slope([(math.log(q), a, b) for q, (_, a, b) in zip((512, 2048, 8192), rows)], S5)
    emit(f"\nTail-Δ slope on log queue size: {fmt(sl)}")
    stats_out["slope_qsize_semeval"] = sl

    # S4 BERT
    emit("\n## 4. BERT-base backbone\n")
    for ds, p in (("GoEmotions", "goemo"), ("SemEval", "semeval")):
        rows = [(f"λ₁={LAMV[t]:.2f}", f"{p}_bert_l{t}", f"{p}_bert_l{t}_no_queue") for t in LAMS]
        family(f"{ds}: queue effect with BERT-base", rows, S5, f"bert_{ds}")
        sl = slope([(math.log(LAMV[t]), f"{p}_bert_l{t}", f"{p}_bert_l{t}_no_queue") for t in LAMS], S5)
        emit(f"\nTail-Δ slope on log λ₁: {fmt(sl)}")
        stats_out[f"slope_bert_{ds}"] = sl
        emit(f"\n{ds} BERT λ₁=0 anchor (w/o SupCon): macro {mean_abs(f'{p}_bert_no_supcon', S5, 'macro_f1')}, "
             f"tail {mean_abs(f'{p}_bert_no_supcon', S5, 'tail')}")
        for t in LAMS:
            emit(f"  λ₁={LAMV[t]:.2f}: Full macro {mean_abs(f'{p}_bert_l{t}', S5, 'macro_f1')}, "
                 f"tail {mean_abs(f'{p}_bert_l{t}', S5, 'tail')}; w/o queue tail "
                 f"{mean_abs(f'{p}_bert_l{t}_no_queue', S5, 'tail')}")

    # S5 component grid
    emit("\n## 5. Component grid (Full − variant), seeds 42–46\n")
    for name in ("GoEmotions", "SemEval"):
        for v, label in (("no_queue", "w/o queue"), ("hard_match", "Jaccard→exact match"), ("no_proto", "w/o LPC")):
            if name == "SemEval":
                rows = [(f"λ₁={LAMV[t]:.2f}", sem_full(t), f"semeval_clfer_lt_l{t}_{v}") for t in LAMS]
            else:
                rows = [(f"λ₁={LAMV[t]:.2f}", goe_full(t), goe_variant(t, v)) for t in LAMS]
            family(f"{name}: {label}", rows, S5, f"comp_{name}_{v}")
        anchor = "clfer_lt_no_supcon" if name == "GoEmotions" else "semeval_clfer_lt_no_supcon"
        emit(f"\n{name} λ₁=0 anchor (w/o SupCon) test macro {mean_abs(anchor, S5, 'macro_f1')}, "
             f"tail {mean_abs(anchor, S5, 'tail')}")
        for t in LAMS:
            f = sem_full(t) if name == "SemEval" else goe_full(t)
            emit(f"  Full λ₁={LAMV[t]:.2f}: macro {mean_abs(f, S5, 'macro_f1')}, tail {mean_abs(f, S5, 'tail')}")

    # S6 compensation: queue recovery vs contrastive-term damage
    emit("\n## 6. Queue recovery vs contrastive-term damage (RoBERTa, batch 32)\n")
    emit("damage = w/o SupCon tail − w/o queue tail; recovery = Full − w/o queue (tail).")
    emit("Independent-seed version: damage uses w/o queue seeds 47–51, recovery uses seeds 42–46.\n")
    emit("| dataset | λ₁ | damage (shared seeds) | damage (seeds 47–51) | recovery |")
    emit("|---|---|---|---|---|")
    xs_shared, xs_ind, ys = [], [], []
    for name in ("GoEmotions", "SemEval"):
        anchor = "clfer_lt_no_supcon" if name == "GoEmotions" else "semeval_clfer_lt_no_supcon"
        for t in LAMS:
            full = goe_full(t) if name == "GoEmotions" else sem_full(t)
            nq = goe_variant(t, "no_queue") if name == "GoEmotions" else f"semeval_clfer_lt_l{t}_no_queue"
            d_sh = statistics.mean(deltas(anchor, nq, S5))
            a_mean = statistics.mean(float(M[k(anchor, s)]["tail"]) for s in S5)
            nq_ext = statistics.mean(float(M[k(nq, s)]["tail"]) for s in S_EXTRA)
            rec = statistics.mean(deltas(full, nq, S5))
            xs_shared.append(d_sh)
            xs_ind.append(a_mean - nq_ext)
            ys.append(rec)
            emit(f"| {name} | {LAMV[t]:.2f} | {d_sh:+.4f} | {a_mean - nq_ext:+.4f} | {rec:+.4f} |")
    for lab, xs in (("shared seeds", xs_shared), ("independent seeds", xs_ind)):
        lr = stats.linregress(xs, ys)
        rho = stats.spearmanr(xs, ys)
        emit(f"\nrecovery ~ damage ({lab}): slope {lr.slope:.3f}, r={lr.rvalue:.3f}, p={lr.pvalue:.4f}; "
             f"Spearman ρ={rho.statistic:.3f}, p={rho.pvalue:.4f}")
        stats_out[f"compensation_{lab.replace(' ', '_')}"] = {
            "slope": lr.slope, "r": lr.rvalue, "p": lr.pvalue, "spearman": rho.statistic, "spearman_p": rho.pvalue,
        }

    # S7 compensation across backbones
    emit("\n## 7. Recovery vs damage across backbones (RoBERTa + BERT, batch 32)\n")
    emit("BERT has seeds 42–46 only; its split-seed version takes damage from seeds 44–46 and recovery from 42–43.\n")
    emit("| backbone | dataset | λ₁ | damage (shared) | damage (split) | recovery (shared) | recovery (split) |")
    emit("|---|---|---|---|---|---|---|")
    cells = []  # (backbone, dataset, t, damage_shared, damage_split, recovery_shared, recovery_split)
    for i, (name, t) in enumerate((n, t) for n in ("GoEmotions", "SemEval") for t in LAMS):
        cells.append(("RoBERTa", name, t, xs_shared[i], xs_ind[i], ys[i], ys[i]))
    split_d, split_r = (44, 45, 46), (42, 43)
    for name, p in (("GoEmotions", "goemo"), ("SemEval", "semeval")):
        anchor = f"{p}_bert_no_supcon"
        for t in LAMS:
            full, nq = f"{p}_bert_l{t}", f"{p}_bert_l{t}_no_queue"
            cells.append((
                "BERT", name, t,
                statistics.mean(deltas(anchor, nq, S5)),
                statistics.mean(deltas(anchor, nq, split_d)),
                statistics.mean(deltas(full, nq, S5)),
                statistics.mean(deltas(full, nq, split_r)),
            ))
    for bb, name, t, dsh, dsp, rsh, rsp in cells:
        emit(f"| {bb} | {name} | {LAMV[t]:.2f} | {dsh:+.4f} | {dsp:+.4f} | {rsh:+.4f} | {rsp:+.4f} |")
    stats_out["compensation_cells"] = [
        {"backbone": c[0], "dataset": c[1], "lambda": LAMV[c[2]], "damage_shared": c[3], "damage_split": c[4],
         "recovery_shared": c[5], "recovery_split": c[6]} for c in cells
    ]
    for subset, sel in (("BERT", lambda c: c[0] == "BERT"), ("RoBERTa + BERT", lambda c: True)):
        sub = [c for c in cells if sel(c)]
        for lab, xi, yi in (("shared seeds", 3, 5), ("split seeds", 4, 6)):
            xs, ys_ = [c[xi] for c in sub], [c[yi] for c in sub]
            lr = stats.linregress(xs, ys_)
            rho = stats.spearmanr(xs, ys_)
            emit(f"\n{subset}, {lab} ({len(sub)} cells): slope {lr.slope:.3f} ± {lr.stderr:.3f}, "
                 f"r={lr.rvalue:.3f}, p={lr.pvalue:.4g}; Spearman ρ={rho.statistic:.3f}, p={rho.pvalue:.4g}")
            stats_out[f"compensation_{subset}_{lab}".replace(" ", "_").replace("+", "and")] = {
                "n": len(sub), "slope": lr.slope, "slope_se": lr.stderr, "intercept": lr.intercept,
                "r": lr.rvalue, "p": lr.pvalue, "spearman": rho.statistic, "spearman_p": rho.pvalue,
            }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ijcs_revision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "ijcs_revision_stats.json").write_text(json.dumps(stats_out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
