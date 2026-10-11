#!/usr/bin/env python3
"""Regenerate publication-facing Markdown tables from committed CSV evidence."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "evidence" / "paper"
OUT = ROOT / "generated" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

FILES = [
    "table1_regimes.csv",
    "table2_learners.csv",
    "table3_primary_inr.csv",
    "table4_selected_cases.csv",
    "table5_frozen_counterfactuals.csv",
    "table6_interventions_public.csv",
    "table6_boundary_context.csv",
]

for name in FILES:
    path = SRC / name
    if not path.exists():
        continue
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    widths = len(rows[0])
    md = ["| " + " | ".join(rows[0]) + " |",
          "| " + " | ".join(["---"] * widths) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    (OUT / name.replace(".csv", ".md")).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(name)
