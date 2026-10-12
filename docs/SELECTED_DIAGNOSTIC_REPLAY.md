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

They are not currently mirrored as binaries in this public repository. This is
an explicit artifact-availability limitation, not an assertion that a replay
has been performed from the public repository alone.

A fully self-contained selected-replay release remains a follow-up artifact
task. Until those state binaries (or an independently verified deterministic
state-regeneration path) are public, the repository supports **claim
verification and exact corpus regeneration**, but not one-command bitwise
replay of the D1/D3 interventions.

## Environment boundary

The original selected continuations used Python 3.11.9, PyTorch 2.13.0+cu130,
CUDA 13.0, float64 deterministic execution, and an NVIDIA GeForce RTX 4080
SUPER. The paper does not claim bitwise portability of optimizer trajectories
to arbitrary hardware.
