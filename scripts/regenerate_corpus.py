#!/usr/bin/env python3
"""Regenerate one frozen Stage V corpus deterministically from the public seed registry."""
import argparse, hashlib, json
from pathlib import Path
from construction_world.benchmark.native_stage_v_common_v13 import seeds, CELLS
from construction_world.benchmark.native_stage_v_sampler_v13 import sample

def encoded_row(row):
    return (json.dumps(row,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode("utf-8")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cell",required=True,choices=CELLS)
    p.add_argument("--corpus",required=True,type=int,choices=range(6))
    p.add_argument("--count",type=int,default=8192)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    entry=seeds()[f"g13/stage_v/corpus/{a.cell}/{a.corpus}"]
    rows,next_draw=sample(entry["seed"],a.cell,a.count)
    raw=b"".join(encoded_row(r) for r in rows)
    sha=hashlib.sha256(raw).hexdigest()
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_bytes(raw)
    print(json.dumps({"cell":a.cell,"corpus":a.corpus,"count":len(rows),"seed":entry["seed"],"next_draw":next_draw,"corpus_sha256":sha},sort_keys=True))
if __name__=="__main__":
    main()
