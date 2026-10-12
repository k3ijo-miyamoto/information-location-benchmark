from pathlib import Path
import json
from construction_world.benchmark.native_stage_v_analysis_v13 import analyze

ROOT=Path(__file__).resolve().parents[1]
def test_sanitized_official_archive_recomputes_primary():
    base=ROOT/"evidence/stage_v/official_run_metrics"
    m=json.loads((base/"manifest.json").read_text())
    records=[]
    for part in m["parts"]:
        with (base/part["path"]).open(encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f if line.strip())
    assert len(records)==2304
    a,_=analyze(records)
    stored=json.loads((ROOT/"evidence/stage_v/stage_v_analysis.json").read_text())
    assert a["primary"]==stored["primary"]
    assert a["negative_controls"]==stored["negative_controls"]
