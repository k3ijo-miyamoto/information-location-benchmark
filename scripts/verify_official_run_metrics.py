#!/usr/bin/env python3
"""Verify the sanitized 2,304-run Stage V archive and recompute official analysis."""
import hashlib, json
from pathlib import Path
from construction_world.benchmark.native_stage_v_analysis_v13 import analyze

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"evidence/stage_v/official_run_metrics"
manifest=json.loads((BASE/"manifest.json").read_text(encoding="utf-8"))
records=[]
all_hasher=hashlib.sha256()
for part in manifest["parts"]:
    p=BASE/part["path"]
    raw=p.read_bytes()
    assert len(raw)==part["bytes"]
    assert hashlib.sha256(raw).hexdigest()==part["sha256"]
    all_hasher.update(raw)
    lines=[x for x in raw.splitlines() if x]
    assert len(lines)==part["records"]
    records.extend(json.loads(x) for x in lines)
assert len(records)==manifest["record_count"]==2304
assert all_hasher.hexdigest()==manifest["concatenated_jsonl_sha256"]
assert len({r["run_id"] for r in records})==2304

recomputed,allocation=analyze(records)
stored=json.loads((ROOT/"evidence/stage_v/stage_v_analysis.json").read_text(encoding="utf-8"))
stored_c=json.loads((ROOT/"evidence/stage_v/stage_c_allocation.json").read_text(encoding="utf-8"))
for k in ("primary","negative_controls","total_optimizer_steps","total_example_presentations","max_allocated_gpu_memory_bytes"):
    assert recomputed[k]==stored[k], k
for k in ("six_sd_inputs","max_sd","candidate_half_widths","selected_R","precision_feasible","stage_c_seeds_reserved","stage_c_runs"):
    assert allocation[k]==stored_c[k], k
print("PASS: 2,304 sanitized Stage V records reproduce official development analysis and Stage C allocation")
