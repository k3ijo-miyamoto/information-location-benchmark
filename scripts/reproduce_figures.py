#!/usr/bin/env python3
"""Regenerate publication figures from committed evidence only."""
from pathlib import Path
import csv, json, statistics
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "evidence"
OUT = ROOT / "generated" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# Figure 1: controlled information location.
fig, ax = plt.subplots(figsize=(8.2, 3.6))
ax.axis("off")
labels = [
    ("Current state", "posterior determined by latest warning", "IS0"),
    ("Topology", "posterior determined by dependency pattern", "IS1"),
    ("Ordered history", "posterior determined by warning order", "IS2"),
]
for i, (title, desc, code) in enumerate(labels):
    x = 0.18 + 0.32 * i
    ax.text(x, 0.67, title, ha="center", va="center", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fill=False))
    ax.text(x, 0.47, desc, ha="center", va="center", fontsize=9, wrap=True)
    ax.text(x, 0.28, code, ha="center", va="center", fontsize=9)
ax.text(0.5, 0.92, "Same public semantic facts; matched two-point posterior marginal", ha="center", fontsize=11)
ax.text(0.5, 0.08, "IS3: separate topology-history interaction extension; Brier-opportunity matched only",
        ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "figure1_design.pdf", bbox_inches="tight")
fig.savefig(OUT / "figure1_design.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# Figure 2: corpus-mean learning curves.
data = rows(E / "paper" / "figure2_corpus_means.csv")
cells = [f"IS{i}-O{o}" for i in range(4) for o in (1, 2)]
families = ["tabular", "static", "static_size_control", "temporal"]
fig, axes = plt.subplots(4, 2, figsize=(9.0, 12.0))
for ax, cell in zip(axes.flat, cells):
    for fam in families:
        subset = [r for r in data if r["cell"] == cell and r["family"] == fam]
        by_n = {}
        for r in subset:
            by_n.setdefault(int(r["N"]), []).append(float(r["recovery_pct"]))
        ns = sorted(by_n)
        means = [statistics.mean(by_n[n]) for n in ns]
        sds = [statistics.stdev(by_n[n]) for n in ns]
        ax.errorbar(ns, means, yerr=sds, marker="o", capsize=2, label=fam)
    ax.set_xscale("log", base=2)
    ax.set_title(cell)
    ax.set_xlabel("training N")
    ax.set_ylabel("normalized Brier recovery x100")
    ax.axhline(0, linewidth=0.8)
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT / "figure2_learning_curves.pdf", bbox_inches="tight")
fig.savefig(OUT / "figure2_learning_curves.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# Figure 3: six corpus-level primary INR contrasts.
analysis = json.loads((E / "stage_v" / "stage_v_analysis.json").read_text(encoding="utf-8"))
keys = ["C1-O1", "C1-O2", "C2-O1", "C2-O2", "C3-O1", "C3-O2"]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
for i, key in enumerate(keys):
    vals = analysis["primary"][key]["values"]
    ax.scatter([i] * len(vals), vals, alpha=0.65)
    ax.scatter([i], [statistics.mean(vals)], marker="D", s=45)
ax.axhline(0, linewidth=0.8)
ax.axhline(0.10, linestyle="--", linewidth=0.8)
ax.axhline(-0.10, linestyle="--", linewidth=0.8)
ax.set_xticks(range(len(keys)), keys)
ax.set_ylabel("paired INR difference")
ax.set_title("Development-stage primary contrasts")
fig.tight_layout()
fig.savefig(OUT / "figure3_primary_inr.pdf", bbox_inches="tight")
fig.savefig(OUT / "figure3_primary_inr.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# Figure 4: selected D1 official trajectory, with distinguishable axes/series.
d1 = [r for r in rows(E / "paper" / "figure4_official_epochs.csv") if r["case"] == "D1"]
x = [int(r["step"]) for r in d1]
recovery = [float(r["recovery_pct"]) for r in d1]
distance = [float(r["content_h_pair_l2"]) for r in d1]
fig, ax1 = plt.subplots(figsize=(8.5, 4.4))
line1, = ax1.plot(x, recovery, marker="o", markersize=2.5, linewidth=1.2, label="recovery")
ax1.set_xlabel("global optimizer step")
ax1.set_ylabel("normalized Brier recovery x100")
ax1.tick_params(axis="y", labelcolor=line1.get_color())
ax1.yaxis.label.set_color(line1.get_color())
ax2 = ax1.twinx()
line2, = ax2.plot(x, distance, linestyle="--", linewidth=1.4, label="history-pair distance")
ax2.set_ylabel("content-history pair L2")
ax2.tick_params(axis="y", labelcolor=line2.get_color())
ax2.yaxis.label.set_color(line2.get_color())
boundary = 12417
ax1.axvline(boundary, linestyle=":", linewidth=1.2, label="selected boundary")
lines = [line1, line2] + [ax1.lines[-1]]
ax1.legend(lines, [l.get_label() for l in lines], loc="lower left")
ax1.set_title("Selected D1 official trajectory")
fig.tight_layout()
fig.savefig(OUT / "figure4_d1_trajectory.pdf", bbox_inches="tight")
fig.savefig(OUT / "figure4_d1_trajectory.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# Figure 6: applied LR normalized by the frozen one-step threshold midpoint.
b = rows(E / "paper" / "table6_boundary_context.csv")
cases = [r["case"] for r in b]
low = [float(r["normalized_low_lr"]) for r in b]
orig = [float(r["normalized_original_lr"]) for r in b]
fig, ax = plt.subplots(figsize=(6.8, 4.2))
xpos = list(range(len(cases)))
ax.scatter(xpos, low, marker="o", label="applied 1e-4")
ax.scatter(xpos, orig, marker="s", label="official 1e-3")
ax.axhline(1.0, linestyle="--", linewidth=1.0, label="one-step transition")
ax.set_yscale("log")
ax.set_xticks(xpos, cases)
ax.set_ylabel("applied LR / frozen threshold midpoint")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "figure6_boundary_normalized_lr.pdf", bbox_inches="tight")
fig.savefig(OUT / "figure6_boundary_normalized_lr.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"wrote figures to {OUT}")
