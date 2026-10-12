import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_pure_relocation_exactness():
    p=json.loads((ROOT/"evidence/design/exact_posterior_distributions.json").read_text())
    cells={(x["regime"],x["opportunity"]):x for x in p["cells"]}
    for opp,gain in (("O1","81/16384"),("O2","81/4096")):
        pure=[cells[(r,opp)] for r in ("IS0","IS1","IS2")]
        assert len({json.dumps(x["posterior_mass"],sort_keys=True) for x in pure})==1
        assert all(x["gain"]==gain for x in pure)
        assert cells[("IS3",opp)]["gain"]==gain
        assert cells[("IS3",opp)]["posterior_mass"]!=pure[0]["posterior_mass"]
