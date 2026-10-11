#!/usr/bin/env python3
"""CPU-only consistency checks for the publication evidence."""
from __future__ import annotations
import csv, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_json(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))

def load_csv(rel):
    with (ROOT / rel).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def close(a, b, tol=1e-12):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)

posterior = load_json("evidence/design/exact_posterior_distributions.json")
assert posterior["status"] == "PASS"
cells = {(x["regime"], x["opportunity"]): x for x in posterior["cells"]}
for opp, gain, risk in [
    ("O1", "81/16384", "4015/16384"),
    ("O2", "81/4096", "943/4096"),
]:
    pure = [cells[(r, opp)] for r in ("IS0", "IS1", "IS2")]
    assert len({json.dumps(x["posterior_mass"], sort_keys=True) for x in pure}) == 1
    assert all(x["gain"] == gain and x["bayes_brier"] == risk for x in pure)
    assert cells[("IS3", opp)]["gain"] == gain
    assert cells[("IS3", opp)]["posterior_mass"] != pure[0]["posterior_mass"]

analysis = load_json("evidence/stage_v/stage_v_analysis.json")
assert analysis["development_only"] is True
expected = {
    "C1-O1": (-0.029064990920097622, 0.2133306778338138),
    "C1-O2": ( 0.018236539775400754, 0.07566871212489289),
    "C2-O1": ( 0.0535173125347381,   0.06072792130427416),
    "C2-O2": ( 0.17796939833977907,  0.12892491292885186),
    "C3-O1": (-0.07644734040692044,  0.05830396528946652),
    "C3-O2": ( 0.04867746143005792,  0.04343791073448027),
}
for key, (mean, sd) in expected.items():
    got = analysis["primary"][key]
    assert got["independent_units"] == 6
    assert close(got["mean"], mean)
    assert close(got["sample_sd"], sd)

assert close(analysis["negative_controls"]["O1:tabular-static"]["mean"], 0.6488574281577417)
assert close(analysis["negative_controls"]["O2:tabular-static"]["mean"], 0.25314964875247276)

allocation = load_json("evidence/stage_v/stage_c_allocation.json")
assert allocation["precision_feasible"] is False
assert allocation["corpora"] == 32
assert close(allocation["worst_halfwidth"], 0.1552776052, 1e-9) if "worst_halfwidth" in allocation else True

primary_rows = {r["contrast"]: r for r in load_csv("evidence/paper/table3_primary_inr.csv")}
for key, (mean, sd) in expected.items():
    assert close(primary_rows[key]["mean_INR_difference"], mean)
    assert close(primary_rows[key]["corpus_sample_sd"], sd)

d1 = load_json("evidence/diagnostics/d1_persistent_lr_summary.json")
d1_one = load_json("evidence/diagnostics/d1_one_step_lr_summary.json")
d3 = load_json("evidence/diagnostics/d3_persistent_lr_summary.json")
boundary = load_json("evidence/diagnostics/boundary_normalized_lr_summary.json")
assert close(d1["final_recovery_pct"], 97.08273892812468)
assert d1["first_content_death_step"] is None
assert d1_one["first_content_death_step"] == 12418
assert close(d1_one["final_recovery_pct"], -0.994285835561944)
assert d3["first_content_death_step"] == 2742
assert close(d3["final_recovery_pct"], -0.000521796345658655)
assert close(boundary["cases"]["D1"]["ratios"]["0.0001"]["normalized_midpoint"], 0.21615772868720054)
assert close(boundary["cases"]["D3"]["ratios"]["0.0001"]["normalized_midpoint"], 3.118757817036381)

print("PASS: exact design, Stage V primary contrasts, IS0 controls, Stage C gate, and selected diagnostic endpoints.")
