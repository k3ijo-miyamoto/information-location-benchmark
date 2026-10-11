#!/usr/bin/env python3
"""Rerun one frozen Stage V learner run on the certified official environment."""
from __future__ import annotations
import argparse, json
from pathlib import Path

from construction_world.benchmark.native_stage_v_common_v13 import FAMILIES, NS, seeds
from construction_world.benchmark.native_stage_v_sampler_v13 import catalog, sample
from construction_world.benchmark.native_stage_v_batch_v13 import templates
from construction_world.benchmark.native_stage_v_training_v13 import train

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cell", required=True, choices=[f"IS{i}-O{o}" for i in range(4) for o in (1,2)])
    p.add_argument("--corpus", required=True, type=int, choices=range(6))
    p.add_argument("--N", required=True, type=int, choices=NS)
    p.add_argument("--init", required=True, type=int, choices=range(3))
    p.add_argument("--family", required=True, choices=FAMILIES)
    p.add_argument("--output", type=Path)
    a=p.parse_args()

    entry=seeds()[f"g13/stage_v/corpus/{a.cell}/{a.corpus}"]
    rows,_=sample(entry["seed"], a.cell, 8192)
    _,publics=catalog()
    cache=templates(publics)
    for family in ("tabular","static","temporal"):
        cache[family]={k:v.to("cuda") for k,v in cache[family].items()}
    cache["static_size_control"]=cache["static"]

    identity={"cell":a.cell,"corpus":a.corpus,"N":a.N,"init":a.init,"family":a.family}
    result=train(identity, rows, cache[a.family])
    payload={"identity":identity,"corpus_seed":entry["seed"],"metadata":result}
    text=json.dumps(payload,sort_keys=True,indent=2)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(text+"\n",encoding="utf-8")
    else:
        print(text)

if __name__=="__main__":
    main()
