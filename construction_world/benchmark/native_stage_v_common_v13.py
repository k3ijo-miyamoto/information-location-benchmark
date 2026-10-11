"""Publication-facing Stage V constants and deterministic identities."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "benchmark_configs/generator_v1.3_stage_v_execution_public.json"
REGISTRY = "benchmark_configs/generator_v1.3_seed_registry_v0.1.json"
DESIGN = "benchmark_configs/generator_v1.3_learner_architecture_v0.1.json"

CELLS = tuple(f"IS{i}-O{o}" for i in range(4) for o in (1, 2))
FAMILIES = ("tabular", "static", "temporal", "static_size_control")
NS = (128, 512, 2048, 8192)

def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")

def digest(value):
    return sha256(encoded(value)).hexdigest()

def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def identities():
    return [
        dict(cell=cell, corpus=corpus, N=n, init=init, family=family)
        for cell in CELLS
        for corpus in range(6)
        for n in NS
        for init in range(3)
        for family in FAMILIES
    ]

def run_id(identity):
    if identity not in identities():
        raise ValueError("outside frozen Stage V grid")
    return "{cell}/c{corpus}/n{N}/i{init}/{family}".format(**identity)

def seeds():
    registry = read(ROOT / REGISTRY)
    return {e["namespace"]: e for e in registry["entries"]}
