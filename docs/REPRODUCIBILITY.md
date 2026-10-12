# Reproducibility

This repository separates paper-level verification, selected diagnostic
verification, and expensive optimizer reruns.

## 1. Paper-level verification

Requires Python 3.11 and no GPU.

```bash
python scripts/verify_paper_evidence.py
python scripts/reproduce_tables.py
```

The verifier checks:

- exact posterior-marginal matching for IS0--IS2 at each opportunity level;
- the narrower Brier-opportunity match for IS3;
- all six Stage V primary INR means and corpus sample SDs;
- the two IS0 Tabular-minus-Static implementation-gap means;
- the Stage C precision gate, selected allocation cap, and the fact that Stage C was not executed.

Publication plotting inputs are under `evidence/paper/`.

## 2. Selected diagnostic verification

The selected D1/D3 claims can be checked from committed compact evidence:

```bash
python scripts/verify_selected_diagnostics.py
```

The exact training corpus used by D1 and D3 can also be regenerated from the
public seed registry:

```bash
python scripts/regenerate_corpus.py \
  --cell IS2-O1 --corpus 0 \
  --output /tmp/is2_o1_c0.jsonl
```

The expected SHA-256 for that 8,192-record corpus is:

`e12ebb3ab92d927d10f68e9b6e1c1e3c96878c2b54c4cab55c76df3d9538b8c8`.

The original D1/D3 pre-boundary state identities are recorded in
`evidence/diagnostics/frozen_state_manifest.json`, and the verified compressed
state archives are published under `frozen_states/`. This makes the selected
prestates independently available. Full continuation replay still requires the
certified GPU/software environment described below.
See [SELECTED_DIAGNOSTIC_REPLAY.md](SELECTED_DIAGNOSTIC_REPLAY.md).

## 3. Evidence layers

- `evidence/design/`: exact finite-support design and adapter/split audits.
- `evidence/stage_v/`: development-stage primary analysis and Stage C allocation decision.
- `evidence/diagnostics/`: selected post-hoc trajectory/intervention summaries and state identities.
- `evidence/paper/`: compact table and figure inputs used by the manuscript.
- `provenance/`: prospective benchmark specification and source-to-public artifact mapping.

Exact evaluation removes sampled evaluation-label noise. It does not remove
training, corpus, initialization, or optimization variability.

## 4. Benchmark rerun status

Generator 1.3, learner/adaptation modules, exact evaluator, deterministic
sampler, seed registry, and official Stage V training implementation are
published.

A single official run can be rerun with:

```bash
python scripts/rerun_single_official.py \
  --cell IS2-O1 --corpus 0 --N 8192 --init 1 --family tabular
```

The strict runner intentionally requires the certified original environment:
Python 3.11.9, PyTorch 2.13.0+cu130, CUDA 13.0, float64 deterministic execution,
and an NVIDIA GeForce RTX 4080 SUPER. A full 2,304-run rerun is not required for
paper-level verification and bitwise portability to arbitrary hardware is not
claimed.

A sanitized per-run Stage V archive is published as 16 JSONL parts under
`evidence/stage_v/official_run_metrics/`. Its manifest binds the 2,304 records
to the original source archive identity, and `scripts/verify_official_run_metrics.py`
recomputes the official development analysis and Stage C allocation from these
public records.

## 5. CI and tests

The GitHub Actions workflow runs:

- publication cleanliness scanning;
- paper evidence verification;
- selected diagnostic verification;
- exact selected-corpus regeneration and hash check;
- publication regression tests;
- table regeneration;
- figure regeneration.

Local equivalent:

```bash
python -m pip install -e '.[test,paper]'
python scripts/check_public_cleanliness.py
python scripts/verify_paper_evidence.py
python scripts/verify_selected_diagnostics.py
python -m pytest -q
python scripts/reproduce_tables.py
python scripts/reproduce_figures.py
```

## 6. Stage C

The planned confirmatory stage was not run. The stored allocation result selected
the cap of 32 corpora but estimated a worst planned half-width of approximately
0.1553, above the predeclared 0.10 target. No Stage C runs or seeds are
represented as completed here.
