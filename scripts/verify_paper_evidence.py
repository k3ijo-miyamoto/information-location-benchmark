#!/usr/bin/env python3
import csv, json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def j(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))

def eq(a,b,tol=1e-12):
    return math.isclose(float(a),float(b),rel_tol=0.0,abs_tol=tol)

p=j("evidence/design/exact_posterior_distributions.json")
assert p["status"]=="PASS"
cells={(x["regime"],x["opportunity"]):x for x in p["cells"]}
for opp,gain in (("O1","81/16384"),("O2","81/4096")):
    pure=[cells[(r,opp)] for r in ("IS0","IS1","IS2")]
    assert len({json.dumps(x["posterior_mass"],sort_keys=True) for x in pure})==1
    assert all(x["gain"]==gain for x in pure)
    assert cells[("IS3",opp)]["gain"]==gain
    assert cells[("IS3",opp)]["posterior_mass"]!=pure[0]["posterior_mass"]

a=j("evidence/stage_v/stage_v_analysis.json")
expected={
"C1-O1":(-0.029064990920097622,0.2133306778338138),
"C1-O2":(0.018236539775400754,0.07566871212489289),
"C2-O1":(0.0535173125347381,0.06072792130427416),
"C2-O2":(0.17796939833977907,0.12892491292885186),
"C3-O1":(-0.07644734040692044,0.05830396528946652),
"C3-O2":(0.04867746143005792,0.04343791073448027)}
for k,(m,s) in expected.items():
    assert eq(a["primary"][k]["mean"],m)
    assert eq(a["primary"][k]["sample_sd"],s)
assert eq(a["negative_controls"]["O1:tabular-static"]["mean"],0.6488574281577417)
assert eq(a["negative_controls"]["O2:tabular-static"]["mean"],0.25314964875247276)

c=j("evidence/stage_v/stage_c_allocation.json")
assert c["precision_feasible"] is False
assert c["selected_R"]==32 and c["stage_c_runs"]==0 and c["stage_c_seeds_reserved"]==0
assert eq(c["candidate_half_widths"]["32"],0.1552776051938475)
print("PASS")
