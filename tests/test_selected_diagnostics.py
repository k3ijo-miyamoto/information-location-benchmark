import json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(p): return json.loads((ROOT/p).read_text())
def test_selected_diagnostics_are_bounded_and_consistent():
    b=load("evidence/diagnostics/optimizer_boundary_summary.json")
    assert b["cases"]["D1"]["first_death_step"]==12417
    assert b["cases"]["D3"]["first_death_step"]==2742
    d1=load("evidence/diagnostics/d1_persistent_lr_summary.json")
    one=load("evidence/diagnostics/d1_one_step_lr_summary.json")
    d3=load("evidence/diagnostics/d3_persistent_lr_summary.json")
    assert d1["first_content_death_step"] is None
    assert math.isclose(d1["final_recovery_pct"],97.08273892812468,abs_tol=1e-12)
    assert one["first_content_death_step"]==12418
    assert d3["first_content_death_step"]==2742
