#!/usr/bin/env python3
"""Replay one selected D1/D3 learning-rate continuation from the published frozen prestate.

This is intentionally strict: the original selected trajectories were produced on
Python 3.11.9 / PyTorch 2.13.0+cu130 / CUDA 13.0 / RTX 4080 SUPER, float64,
deterministic execution. The script refuses automatic hardware fallback.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path

import torch

from construction_world.benchmark import native_stage_v_training_v02 as training
from construction_world.benchmark.native_device_v13 import build_model
from construction_world.benchmark.native_no_training_gate_v13 import parameter_digest
from construction_world.benchmark.native_stage_v_batch_v13 import forward, select, templates
from construction_world.benchmark.native_stage_v_common_v13 import encoded, seeds
from construction_world.benchmark.native_stage_v_sampler_v13 import catalog, sample

ROOT=Path(__file__).resolve().parents[1]

CASES={
    "d1-persistent": dict(
        state_case="D1", family="tabular", init=1, start=12417, end=25600,
        one_step_only=False, expected_first_death=None,
        expected_recovery=0.9708273892812469,
        expected_model_sha="7fc50c05dda530499da01bd849b7b5f131aaa858dbb22f49d5b61c15552d627d",
    ),
    "d1-one-step": dict(
        state_case="D1", family="tabular", init=1, start=12417, end=25600,
        one_step_only=True, expected_first_death=12418,
        expected_recovery=-0.00994285835561944,
        expected_model_sha="bb7bbcfb5f5394abf738ad8762e4ca1d0c9a617620f5c28930497f1a4183b5e4",
    ),
    "d3-persistent": dict(
        state_case="D3", family="temporal", init=0, start=2742, end=25600,
        one_step_only=False, expected_first_death=2742,
        expected_recovery=-5.217963456586549e-06,
        expected_model_sha="f8f49f61ffba763b4b95b760a1882bd1ab9a7303b4f432fd62c34a1a6126536d",
    ),
}
CELL="IS2-O1"
CORPUS=0
N=8192
LOW_LR=0.0001
OFFICIAL_LR=0.001
EXPECTED_CORPUS_SHA="e12ebb3ab92d927d10f68e9b6e1c1e3c96878c2b54c4cab55c76df3d9538b8c8"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@torch.no_grad()
def content_dead(model, cache) -> bool:
    mask=cache["mask"].bool()
    x=torch.where(mask.unsqueeze(-1),cache["x"],torch.zeros_like(cache["x"]))
    z=model.parts["content"](x)
    return bool((z[mask] <= 0).all())


def corpus_rows():
    entry=seeds()[f"g13/stage_v/corpus/{CELL}/{CORPUS}"]
    rows,_=sample(entry["seed"],CELL,N)
    raw=b"".join(encoded(r) for r in rows)
    actual=sha256(raw)
    if actual != EXPECTED_CORPUS_SHA:
        raise RuntimeError(f"corpus hash mismatch: {actual}")
    return rows,actual


def load_archive(case):
    manifest=json.loads((ROOT/"frozen_states/manifest.json").read_text(encoding="utf-8"))
    item=manifest["items"][case]
    compressed=(ROOT/item["path"]).read_bytes()
    if len(compressed)!=item["compressed_bytes"] or sha256(compressed)!=item["compressed_sha256"]:
        raise RuntimeError("compressed state identity mismatch")
    raw=gzip.decompress(compressed)
    if len(raw)!=item["decompressed_bytes"] or sha256(raw)!=item["decompressed_sha256"]:
        raise RuntimeError("decompressed state identity mismatch")
    return torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True),item


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant",choices=sorted(CASES),required=True)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    spec=CASES[args.variant]

    # This enforces the certified original CUDA/software environment.
    environment=training.environment()

    rows,corpus_sha=corpus_rows()
    _,publics=catalog()
    cache_all=templates(publics)
    cache={k:v.to("cuda") for k,v in cache_all[spec["family"]].items()}
    inputs=torch.tensor([r["public_support_index"] for r in rows],device="cuda")
    labels=torch.tensor([r["y"] for r in rows],device="cuda",dtype=torch.float64)

    archive,state_meta=load_archive(spec["state_case"])
    identity=dict(cell=CELL,corpus=CORPUS,N=N,init=spec["init"],family=spec["family"])
    if archive["identity"] != identity or archive["global_step"] != spec["start"]:
        raise RuntimeError("frozen prestate identity mismatch")

    init_seed=seeds()[f"g13/stage_v/init/{spec['init']}"]["seed"]
    model=build_model(spec["family"],init_seed,"cuda")
    model.load_state_dict(archive["model"])
    opt=training.optimizer(model)
    opt.load_state_dict(copy.deepcopy(archive["optimizer"]))
    if len(opt.param_groups)!=1 or opt.param_groups[0]["lr"]!=OFFICIAL_LR:
        raise RuntimeError("unexpected frozen optimizer group")
    if any(int(s["step"]) != spec["start"]-1 for s in opt.state.values()):
        raise RuntimeError("unexpected frozen optimizer step counters")
    model.train()

    epoch0=(spec["start"]-1)//256
    batch0=(spec["start"]-1)%256
    next_batch=training.shuffle(CELL,CORPUS,spec["init"],N,epoch0).split(32)[batch0]
    if next_batch.tolist()!=archive["minibatch_row_indices"]:
        raise RuntimeError("next minibatch does not match frozen state")

    opt.param_groups[0]["lr"]=LOW_LR
    first_death=None
    completed=spec["start"]-1

    for epoch in range(epoch0,100):
        order=training.shuffle(CELL,CORPUS,spec["init"],N,epoch).to("cuda")
        for bi,batch in enumerate(order.split(32)):
            step=epoch*256+bi+1
            if step<=completed:
                continue
            if spec["one_step_only"] and step==spec["start"]+1:
                opt.param_groups[0]["lr"]=OFFICIAL_LR
            expected_lr=LOW_LR if (not spec["one_step_only"] or step==spec["start"]) else OFFICIAL_LR
            if opt.param_groups[0]["lr"]!=expected_lr:
                raise RuntimeError("learning-rate schedule mismatch")

            opt.zero_grad(set_to_none=True)
            logits=forward(model,select(cache,inputs[batch]))
            loss=torch.nn.functional.binary_cross_entropy_with_logits(
                logits,labels[batch],reduction="mean"
            )
            loss.backward()
            if not bool(torch.isfinite(loss)) or any(
                p.grad is None or not bool(torch.isfinite(p.grad).all())
                for p in model.parameters()
            ):
                raise RuntimeError(f"nonfinite loss/gradient at step {step}")
            opt.step()
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise RuntimeError(f"nonfinite parameter at step {step}")

            dead=content_dead(model,cache)
            if dead and first_death is None:
                first_death=step
            completed=step

    if completed!=spec["end"]:
        raise RuntimeError("continuation did not reach frozen endpoint")

    model.eval()
    exact=training.posterior(CELL,"test")
    with torch.no_grad():
        probs=torch.sigmoid(forward(model,select(cache,list(exact)))).cpu().tolist()
    evaluation=training.metrics(probs,list(exact.values()),"O1")
    recovery=evaluation["normalized_recovery"]
    model_sha=parameter_digest(model)

    if first_death!=spec["expected_first_death"]:
        raise AssertionError((first_death,spec["expected_first_death"]))
    if abs(recovery-spec["expected_recovery"])>1e-12:
        raise AssertionError((recovery,spec["expected_recovery"]))
    if model_sha!=spec["expected_model_sha"]:
        raise AssertionError((model_sha,spec["expected_model_sha"]))

    result=dict(
        variant=args.variant,
        source_run_id=f"{CELL}/c{CORPUS}/n{N}/i{spec['init']}/{spec['family']}",
        environment=environment,
        corpus_sha256=corpus_sha,
        frozen_state=state_meta,
        completed_step=completed,
        first_content_death_step=first_death,
        final_test_evaluation=evaluation,
        final_model_sha256=model_sha,
        exact_expected_match=True,
    )
    text=json.dumps(result,indent=2,sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text+"\n",encoding="utf-8")
    print(text)


if __name__=="__main__":
    main()
