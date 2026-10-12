#!/usr/bin/env python3
"""Verify selected diagnostic claims from committed publication evidence only."""
import json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(p): return json.loads((ROOT/p).read_text(encoding="utf-8"))
def close(a,b,tol=1e-12): return math.isclose(float(a),float(b),rel_tol=0.0,abs_tol=tol)

boundary=load("evidence/diagnostics/optimizer_boundary_summary.json")
assert boundary["cases"]["D1"]["first_death_step"]==12417
assert boundary["cases"]["D3"]["first_death_step"]==2742
assert boundary["cases"]["D1"]["boundary_status"]=="BOUNDARY-A"
assert boundary["cases"]["D3"]["boundary_status"]=="BOUNDARY-A"

d1=load("evidence/diagnostics/d1_persistent_lr_summary.json")
assert d1["source_run_id"]=="IS2-O1/c0/n8192/i1/tabular"
assert d1["first_content_death_step"] is None
assert close(d1["final_recovery_pct"],97.08273892812468)

one=load("evidence/diagnostics/d1_one_step_lr_summary.json")
assert one["first_content_death_step"]==12418
assert close(one["final_recovery_pct"],-0.994285835561944)

d3=load("evidence/diagnostics/d3_persistent_lr_summary.json")
assert d3["source_run_id"]=="IS2-O1/c0/n8192/i0/temporal"
assert d3["first_content_death_step"]==2742
assert close(d3["final_recovery_pct"],-0.000521796345658655)

thr=load("evidence/diagnostics/boundary_normalized_lr_summary.json")
assert close(thr["cases"]["D1"]["midpoint"],0.00046262514233163924)
assert close(thr["cases"]["D1"]["ratios"]["0.0001"]["normalized_midpoint"],0.21615772868720054)
assert close(thr["cases"]["D3"]["midpoint"],0.00003206404788911299)
assert close(thr["cases"]["D3"]["ratios"]["0.0001"]["normalized_midpoint"],3.118757817036381)

states=load("evidence/diagnostics/frozen_state_manifest.json")
assert states["cases"]["D1"]["decompressed_boundary_state_sha256"]==d1["binding"]["archive_sha256"]
assert states["cases"]["D3"]["decompressed_boundary_state_sha256"]==d3["binding"]["archive_sha256"]
assert states["cases"]["D1"]["corpus_sha256"]==states["cases"]["D3"]["corpus_sha256"]
print("PASS: selected diagnostic summaries, interventions, thresholds, and state identities")
