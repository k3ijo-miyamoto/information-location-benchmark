# Selected diagnostic replay status

The paper's D1/D3 mechanism and learning-rate results are **selected post-hoc diagnostics**, not prevalence estimates.

## What is public now

The repository contains:

- exact selected-case identities and first content-death steps;
- compact optimizer-boundary and counterfactual summaries;
- persistent and one-step learning-rate intervention summaries;
- the frozen one-step learning-rate threshold summary;
- deterministic Stage V corpus generation code and the frozen seed registry;
- a clean manifest binding D1/D3 to the original frozen pre-boundary state identities.

Run:

```bash
python scripts/regenerate_corpus.py --cell IS2-O1 --corpus 0 --output /tmp/is2_o1_c0.jsonl
python scripts/verify_selected_diagnostics.py
```

For both D1 and D3, the regenerated 8,192-record corpus must have SHA-256:

`e12ebb3ab92d927d10f68e9b6e1c1e3c96878c2b54c4cab55c76df3d9538b8c8`.

## Frozen pre-boundary states

The original D1/D3 compressed state files are identified in
`evidence/diagnostics/frozen_state_manifest.json`, including source Git blob
identity, decompressed byte count, and decompressed SHA-256.

They are now mirrored under `frozen_states/`. The publication reconstruction
workflow verified each compressed archive and its decompressed SHA-256 against
the source transport manifest before committing the files.

The repository therefore contains the exact selected prestates needed for
D1/D3 continuation replay. Publication-facing replay scripts are provided:

```bash
python scripts/replay_d1_persistent_lr.py --output /tmp/d1-persistent.json
python scripts/replay_d1_one_step_lr.py --output /tmp/d1-one-step.json
python scripts/replay_d3_persistent_lr.py --output /tmp/d3-persistent.json
```

Each replay regenerates and hashes the selected corpus, verifies the frozen
state archive, resumes from the exact selected pre-boundary optimizer/model
state, executes the documented learning-rate schedule, and checks the final
model hash, first content-death step, and final exact normalized Brier recovery
against the committed selected-case evidence.

The remaining portability boundary is the certified GPU/software environment,
not state availability. The three full continuations are therefore not run by
hosted CPU CI.

## Environment boundary

The original selected continuations used Python 3.11.9, PyTorch 2.13.0+cu130,
CUDA 13.0, float64 deterministic execution, and an NVIDIA GeForce RTX 4080
SUPER. The paper does not claim bitwise portability of optimizer trajectories
to arbitrary hardware.
