import json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def close(a,b): return math.isclose(float(a),float(b),rel_tol=0.0,abs_tol=1e-12)
def test_primary_and_stage_c():
    a=json.loads((ROOT/"evidence/stage_v/stage_v_analysis.json").read_text())
    expected={"C1-O1":-0.029064990920097622,"C1-O2":0.018236539775400754,"C2-O1":0.0535173125347381,"C2-O2":0.17796939833977907,"C3-O1":-0.07644734040692044,"C3-O2":0.04867746143005792}
    for k,v in expected.items(): assert close(a["primary"][k]["mean"],v)
    c=json.loads((ROOT/"evidence/stage_v/stage_c_allocation.json").read_text())
    assert c["precision_feasible"] is False
    assert c["selected_R"]==32 and c["stage_c_runs"]==0 and c["stage_c_seeds_reserved"]==0
