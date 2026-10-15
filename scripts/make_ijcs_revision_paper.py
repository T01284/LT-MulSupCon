#!/usr/bin/env python3
"""Figures and table bodies for the cross-dataset / cross-backbone queue analysis in paper/main.tex.

Writes fig7_queue_dose and fig8_compensation to paper/figures and outputs/figures, and fills
the % BEGIN_XXX / % END_XXX blocks QUEUE_GRID, COMPONENT_GRID, COMPENSATION_FIT and
CAPACITY_BATCH in paper/main.tex, plus QUEUE_CI and COMPENSATION_CELLS in paper/supplementary.tex.
Statistics come from scripts/analyze_ijcs_revision.py.
"""

from __future__ import annotations

import math
import re
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_ijcs_revision as A  # noqa: E402

TEX = ROOT / "paper" / "main.tex"
SUPP = ROOT / "paper" / "supplementary.tex"
FIG_DIRS = (ROOT / "paper" / "figures", ROOT / "outputs" / "figures")
LAMS = A.LAMS
LAMV = A.LAMV
COLORS = {"GoEmotions": "#1f5fa8", "SemEval": "#d9731a"}
ENCODERS = {"RoBERTa": "#1f5fa8", "BERT": "#d9731a"}
MARKERS = {"RoBERTa": "o", "BERT": "^"}


def queue_pair(backbone: str, ds: str, t: str) -> tuple[str, str]:
    if backbone == "BERT":
        p = "goemo" if ds == "GoEmotions" else "semeval"
        return f"{p}_bert_l{t}", f"{p}_bert_l{t}_no_queue"
    if ds == "GoEmotions":
        return A.goe_full(t), A.goe_variant(t, "no_queue")
    return A.sem_full(t), f"semeval_clfer_lt_l{t}_no_queue"


def anchor(backbone: str, ds: str) -> str:
    if backbone == "BERT":
        return ("goemo" if ds == "GoEmotions" else "semeval") + "_bert_no_supcon"
    return "clfer_lt_no_supcon" if ds == "GoEmotions" else "semeval_clfer_lt_no_supcon"


def seeds_for(backbone: str):
    return A.S10 if backbone == "RoBERTa" else A.S5


def sci_p(p: float) -> str:
    if p >= 0.001:
        return f"{p:.3f}"
    e = math.floor(math.log10(p))
    return f"{p / 10 ** e:.0f}\\times10^{{{e}}}"


def p_tex(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def signed(v: float) -> str:
    s = f"{v:+.3f}"
    return "0.000" if s in ("+0.000", "-0.000") else s


def cell(r: dict | None, holm_p: float | None) -> str:
    if r is None:
        return "--"
    star = "$^{*}$" if holm_p is not None and holm_p < 0.05 else ""
    v = f"{r['mean']:+.3f}"
    return ("0.000" if v in ("+0.000", "-0.000") else v) + star


def replace_block(text: str, name: str, body: str, path: Path = TEX) -> str:
    begin, end = f"% BEGIN_{name}", f"% END_{name}"
    pattern = re.escape(begin) + r"[ \t]*\n.*?" + re.escape(end)
    new, n = re.subn(pattern, lambda _m: begin + "\n" + body + "\n" + end, text, count=1, flags=re.DOTALL)
    if n == 0:
        raise SystemExit(f"marker {name} not found in {path}")
    return new


# ---------------------------------------------------------------- tables
def queue_grid() -> str:
    cols = [("RoBERTa", "GoEmotions"), ("RoBERTa", "SemEval"), ("BERT", "GoEmotions"), ("BERT", "SemEval")]
    per_col = []
    for bb, ds in cols:
        rs = [A.summarize(A.deltas(*queue_pair(bb, ds, t), seeds_for(bb))) for t in LAMS]
        hp = A.holm([r["p"] for r in rs])
        sl = A.slope([(math.log(LAMV[t]), *queue_pair(bb, ds, t)) for t in LAMS], seeds_for(bb))
        per_col.append((rs, hp, sl))
    rows = []
    for i, t in enumerate(LAMS):
        rows.append(f"{LAMV[t]:.2f} & " + " & ".join(cell(rs[i], hp[i]) for rs, hp, _ in per_col) + r" \\")
    rows.append(r"\midrule")
    rows.append("Slope & " + " & ".join(f"{sl['mean']:+.3f}" for _, _, sl in per_col) + r" \\")
    rows.append("Slope $p$ & " + " & ".join(p_tex(sl["p"]) for _, _, sl in per_col) + r" \\")
    return "\n".join(rows)


def component_grid() -> str:
    cols = []
    for ds in ("GoEmotions", "SemEval"):
        for v in ("no_queue", "hard_match", "no_proto"):
            pairs = []
            for t in LAMS:
                if ds == "GoEmotions":
                    pairs.append((A.goe_full(t), A.goe_variant(t, v)))
                else:
                    pairs.append((A.sem_full(t), f"semeval_clfer_lt_l{t}_{v}"))
            rs = [A.summarize(A.deltas(a, b, A.S5)) for a, b in pairs]
            cols.append((rs, A.holm([r["p"] for r in rs])))
    return "\n".join(
        f"{LAMV[t]:.2f} & " + " & ".join(cell(rs[i], hp[i]) for rs, hp in cols) + r" \\"
        for i, t in enumerate(LAMS)
    )


def compensation_cells() -> list[dict]:
    """Damage = anchor - w/o queue; recovery = Full - w/o queue (tail-F1, batch 32)."""
    cells = []
    for bb in ("RoBERTa", "BERT"):
        for ds in ("GoEmotions", "SemEval"):
            anc = anchor(bb, ds)
            for t in LAMS:
                full, nq = queue_pair(bb, ds, t)
                d_sh = statistics.mean(A.deltas(anc, nq, A.S5))
                r_sh = statistics.mean(A.deltas(full, nq, A.S5))
                if bb == "RoBERTa":
                    a_mean = statistics.mean(float(A.M[A.k(anc, s)]["tail"]) for s in A.S5)
                    nq_ext = statistics.mean(float(A.M[A.k(nq, s)]["tail"]) for s in A.S_EXTRA)
                    d_sp, r_sp = a_mean - nq_ext, r_sh
                else:
                    d_sp = statistics.mean(A.deltas(anc, nq, (44, 45, 46)))
                    r_sp = statistics.mean(A.deltas(full, nq, (42, 43)))
                cells.append({"bb": bb, "ds": ds, "t": t, "d_sh": d_sh, "r_sh": r_sh, "d_sp": d_sp, "r_sp": r_sp})
    return cells


def fit(cells: list[dict], x: str, y: str):
    return stats.linregress([c[x] for c in cells], [c[y] for c in cells])


def interaction_p(cells: list[dict], x: str, y: str) -> tuple[float, float, float]:
    xv = np.array([c[x] for c in cells])
    yv = np.array([c[y] for c in cells])
    b = np.array([c["bb"] == "BERT" for c in cells], dtype=float)
    X = np.column_stack([np.ones_like(xv), xv, b, xv * b])
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    dof = len(yv) - X.shape[1]
    s2 = float(((yv - X @ beta) ** 2).sum()) / dof
    se = float(np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))[3])
    return float(beta[3]), se, float(2 * stats.t.sf(abs(beta[3] / se), dof))


def compensation_fit(cells: list[dict]) -> str:
    rows = []
    for lab, sub in (
        ("RoBERTa", [c for c in cells if c["bb"] == "RoBERTa"]),
        ("BERT", [c for c in cells if c["bb"] == "BERT"]),
        ("Pooled", cells),
        (r"Pooled, $\lambda_1{<}0.50$", [c for c in cells if c["t"] != "050"]),
    ):
        a, b = fit(sub, "d_sh", "r_sh"), fit(sub, "d_sp", "r_sp")
        rows.append(
            f"{lab} & {len(sub)} & {a.slope:.2f} $\\pm$ {a.stderr:.2f} & {a.rvalue:.2f} & "
            f"{b.slope:.2f} $\\pm$ {b.stderr:.2f} & {b.rvalue:.2f} \\\\"
        )
    return "\n".join(rows)


def capacity_batch() -> str:
    def row(label: str, a: str, b: str, seeds) -> str:
        r = A.summarize(A.deltas(a, b, seeds))
        return f"{label} & {r['mean']:+.3f} & $[{r['lo']:+.3f},{r['hi']:+.3f}]$ & {p_tex(r['p'])} \\\\"

    g = [
        r"\multicolumn{4}{@{}l}{\textit{GoEmotions, batch size (seeds 42--46)}} \\",
        row("$B{=}32$", "clfer_lt", "clfer_lt_no_queue", A.S5),
        row("$B{=}64$", "goemo_b64_l050", "goemo_b64_l050_no_queue", A.S5),
        row("$B{=}128$", "goemo_b128_l050", "goemo_b128_l050_no_queue", A.S5),
        r"\multicolumn{4}{@{}l}{\textit{GoEmotions, queue size (seeds 47--51)}} \\",
        row("$K{=}1024$", "goemo_q1024_l050", "clfer_lt_no_queue", A.S_EXTRA),
        row("$K{=}4096$", "clfer_lt", "clfer_lt_no_queue", A.S_EXTRA),
        row("$K{=}16384$", "goemo_q16384_l050", "clfer_lt_no_queue", A.S_EXTRA),
        r"\midrule",
        r"\multicolumn{4}{@{}l}{\textit{SemEval, batch size (seeds 42--46)}} \\",
        row("$B{=}8$", "semeval_b8_l050", "semeval_b8_l050_no_queue", A.S5),
        row("$B{=}16$", "semeval_b16_l050", "semeval_b16_l050_no_queue", A.S5),
        row("$B{=}32$", "semeval_clfer_lt_l050", "semeval_clfer_lt_l050_no_queue", A.S5),
        r"\multicolumn{4}{@{}l}{\textit{SemEval, queue size (seeds 42--46)}} \\",
        row("$K{=}512$", "semeval_q512_l050", "semeval_clfer_lt_l050_no_queue", A.S5),
        row("$K{=}2048$", "semeval_clfer_lt_l050", "semeval_clfer_lt_l050_no_queue", A.S5),
        row("$K{=}8192$", "semeval_q8192_l050", "semeval_clfer_lt_l050_no_queue", A.S5),
    ]
    return "\n".join(g)


def queue_ci() -> str:
    rows = []
    for bb in ("RoBERTa", "BERT"):
        for ds in ("GoEmotions", "SemEval"):
            seeds = seeds_for(bb)
            rs = [A.summarize(A.deltas(*queue_pair(bb, ds, t), seeds)) for t in LAMS]
            hp = A.holm([r["p"] for r in rs])
            for i, t in enumerate(LAMS):
                r = rs[i]
                head = f"{bb} & {ds} & {len(seeds)}" if i == 0 else " & & "
                rows.append(
                    f"{head} & {LAMV[t]:.2f} & {signed(r['mean'])} & $[{r['lo']:+.3f},{r['hi']:+.3f}]$ & "
                    f"{p_tex(r['p'])} & {p_tex(hp[i])} \\\\"
                )
            if (bb, ds) != ("BERT", "SemEval"):
                rows.append(r"\midrule")
    return "\n".join(rows)


def compensation_table(cells: list[dict]) -> str:
    rows = []
    for i, c in enumerate(cells):
        head = f"{c['bb']} & {c['ds']}" if i % len(LAMS) == 0 else " & "
        rows.append(
            f"{head} & {LAMV[c['t']]:.2f} & {signed(c['d_sh'])} & {signed(c['r_sh'])} & "
            f"{signed(c['d_sp'])} & {signed(c['r_sp'])} \\\\"
        )
        if i % len(LAMS) == len(LAMS) - 1 and i != len(cells) - 1:
            rows.append(r"\midrule")
    return "\n".join(rows)


# ---------------------------------------------------------------- figures
def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 120,
    })


def save(fig, name: str) -> None:
    for d in FIG_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def plot_dose() -> None:
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    x = np.array([LAMV[t] for t in LAMS])
    for ax, ds in zip(axes, ("GoEmotions", "SemEval")):
        for bb, shift in (("RoBERTa", 0.96), ("BERT", 1.04)):
            rs = [A.summarize(A.deltas(*queue_pair(bb, ds, t), seeds_for(bb))) for t in LAMS]
            m = np.array([r["mean"] for r in rs])
            err = np.array([[r["mean"] - r["lo"] for r in rs], [r["hi"] - r["mean"] for r in rs]])
            n = len(seeds_for(bb))
            ax.errorbar(x * shift, m, yerr=err, color=ENCODERS[bb], marker=MARKERS[bb], ms=5, lw=1.4, capsize=2.5,
                        ls="-" if bb == "RoBERTa" else "--", mfc="white" if bb == "BERT" else ENCODERS[bb],
                        label=f"{bb} ({n} seeds)")
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x])
        ax.minorticks_off()
        ax.set_xlabel(r"contrastive weight $\lambda_1$")
        ax.set_title(("(a) " if ds == "GoEmotions" else "(b) ") + ds, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("tail-F1 gain from queue\n(Full $-$ w/o queue)")
    fig.tight_layout()
    save(fig, "fig7_queue_dose")


def plot_compensation(cells: list[dict]) -> None:
    style()
    fig, ax = plt.subplots(figsize=(3.5, 3.05))
    for c in cells:
        ax.scatter(c["d_sh"], c["r_sh"], color=COLORS[c["ds"]], marker=MARKERS[c["bb"]], s=36,
                   facecolors=COLORS[c["ds"]] if c["bb"] == "RoBERTa" else "white", linewidths=1.2, zorder=3)
    lr = fit(cells, "d_sh", "r_sh")
    xs = np.linspace(min(c["d_sh"] for c in cells) - 0.005, max(c["d_sh"] for c in cells) + 0.005, 50)
    ax.plot(xs, lr.intercept + lr.slope * xs, color="#333", lw=1.2, label=f"OLS fit, slope {lr.slope:.2f}, $r$={lr.rvalue:.2f}")
    ax.plot(xs, xs / 2, color="#999", lw=1.0, ls="--", label=r"recovery $=$ damage$/2$")
    ax.plot(xs, xs, color="#bbb", lw=0.8, ls=":", label=r"recovery $=$ damage")
    ax.axhline(0, color="#ddd", lw=0.6, zorder=0)
    ax.axvline(0, color="#ddd", lw=0.6, zorder=0)
    from matplotlib.lines import Line2D

    handles, _ = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], color=COLORS["GoEmotions"], marker="s", ls="", label="GoEmotions"),
        Line2D([], [], color=COLORS["SemEval"], marker="s", ls="", label="SemEval"),
        Line2D([], [], color="#555", marker="o", ls="", label="RoBERTa"),
        Line2D([], [], color="#555", marker="^", mfc="white", ls="", label="BERT"),
    ]
    ax.set_xlim(-0.035, 0.125)
    ax.set_ylim(-0.02, 0.075)
    ax.set_xticks([0.0, 0.05, 0.10])
    ax.set_yticks([0.0, 0.02, 0.04, 0.06])
    ax.set_xlabel(r"damage (tail-F1): $\lambda_1{=}0$ anchor $-$ w/o queue")
    ax.set_ylabel("recovery (tail-F1):\nFull $-$ w/o queue")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=2, fontsize=7,
               bbox_to_anchor=(0.55, 0.0))
    fig.tight_layout(rect=(0.0, 0.24, 1.0, 1.0))
    save(fig, "fig8_compensation")


def _series(ax, specs, color, label, dodge: float) -> None:
    rs = [A.summarize(A.deltas(a, b, seeds)) for _, a, b, seeds in specs]
    x = np.array([v * dodge for v, *_ in specs], dtype=float)
    m = np.array([r["mean"] for r in rs])
    err = np.array([[r["mean"] - r["lo"] for r in rs], [r["hi"] - r["mean"] for r in rs]])
    ax.errorbar(x, m, yerr=err, color=color, marker="o", ms=5, lw=1.4, capsize=2.5,
                mfc="white" if label.startswith("SemEval") else color, label=label)


def plot_capacity() -> None:
    """Queue length and batch size at λ₁=0.50. Same contrasts as the capacity table."""
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    queue = {
        "GoEmotions": [
            (1024, "goemo_q1024_l050", "clfer_lt_no_queue", A.S_EXTRA),
            (4096, "clfer_lt", "clfer_lt_no_queue", A.S_EXTRA),
            (16384, "goemo_q16384_l050", "clfer_lt_no_queue", A.S_EXTRA),
        ],
        "SemEval": [
            (512, "semeval_q512_l050", "semeval_clfer_lt_l050_no_queue", A.S5),
            (2048, "semeval_clfer_lt_l050", "semeval_clfer_lt_l050_no_queue", A.S5),
            (8192, "semeval_q8192_l050", "semeval_clfer_lt_l050_no_queue", A.S5),
        ],
    }
    batch = {
        "GoEmotions": [
            (32, "clfer_lt", "clfer_lt_no_queue", A.S5),
            (64, "goemo_b64_l050", "goemo_b64_l050_no_queue", A.S5),
            (128, "goemo_b128_l050", "goemo_b128_l050_no_queue", A.S5),
        ],
        "SemEval": [
            (8, "semeval_b8_l050", "semeval_b8_l050_no_queue", A.S5),
            (16, "semeval_b16_l050", "semeval_b16_l050_no_queue", A.S5),
            (32, "semeval_clfer_lt_l050", "semeval_clfer_lt_l050_no_queue", A.S5),
        ],
    }
    for ax, spec, xlabel, ticks in (
        (axes[0], queue, "queue length $K$", [512, 1024, 2048, 4096, 8192, 16384]),
        (axes[1], batch, "batch size $B$", [8, 16, 32, 64, 128]),
    ):
        _series(ax, spec["GoEmotions"], COLORS["GoEmotions"], "GoEmotions", 0.96)
        _series(ax, spec["SemEval"], COLORS["SemEval"], "SemEval", 1.04)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(v) for v in ticks], rotation=45, ha="right")
        ax.minorticks_off()
        ax.set_xlabel(xlabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("tail-F1 gain from queue\n(Full $-$ w/o queue)")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_title(r"(a) queue length, $\lambda_1{=}0.50$", fontsize=10)
    axes[1].set_title(r"(b) batch size, $\lambda_1{=}0.50$", fontsize=10)
    fig.tight_layout()
    save(fig, "fig9_capacity")


def main() -> None:
    cells = compensation_cells()
    plot_dose()
    plot_compensation(cells)
    plot_capacity()
    text = TEX.read_text(encoding="utf-8")
    text = replace_block(text, "QUEUE_GRID", queue_grid())
    text = replace_block(text, "COMPONENT_GRID", component_grid())
    text = replace_block(text, "COMPENSATION_FIT", compensation_fit(cells))
    text = replace_block(text, "CAPACITY_BATCH", capacity_batch())
    TEX.write_text(text, encoding="utf-8")
    supp = SUPP.read_text(encoding="utf-8")
    supp = replace_block(supp, "QUEUE_CI", queue_ci(), SUPP)
    supp = replace_block(supp, "COMPENSATION_CELLS", compensation_table(cells), SUPP)
    SUPP.write_text(supp, encoding="utf-8")
    for x, y in (("d_sh", "r_sh"), ("d_sp", "r_sp")):
        lr = fit(cells, x, y)
        b, se, p = interaction_p(cells, x, y)
        print(f"{x}: pooled slope {lr.slope:.3f}±{lr.stderr:.3f} r={lr.rvalue:.3f} p={sci_p(lr.pvalue)}; "
              f"backbone interaction {b:+.3f}±{se:.3f} p={p:.2f}")
    print("updated", TEX, "and", SUPP)


if __name__ == "__main__":
    main()
