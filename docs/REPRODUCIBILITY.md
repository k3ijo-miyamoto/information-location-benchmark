# Reproducibility

This repository separates paper-level verification from expensive reruns.

## 1. Paper-level verification

Requires Python 3.11 and no GPU.

```bash
python scripts/verify_paper_evidence.py
```

The verifier checks:

- exact posterior-marginal matching for IS0--IS2 at each opportunity level;
- the narrower Brier-opportunity match for IS3;
- all six Stage V primary INR means and corpus sample SDs;
- the two IS0 Tabular-minus-Static implementation-gap means;
- the Stage C precision gate, selected allocation cap, and the fact that Stage C was not executed.

The published tables and plotting inputs are under `evidence/paper/`.

## 2. Evidence layers

- `evidence/design/`: exact finite-support design and adapter/split audits.
- `evidence/stage_v/`: development-stage primary analysis and Stage C allocation decision.
- `evidence/diagnostics/`: selected post-hoc trajectory/intervention summaries.
- `evidence/paper/`: compact table and figure inputs used by the manuscript.

Exact evaluation removes sampled evaluation-label noise. It does not remove training, corpus, initialization, or optimization variability.

## 3. Benchmark rerun status

Generator 1.3 and the core learner/adaptation modules are being staged in this clean repository. The original official execution environment was Python 3.11.9, PyTorch 2.13.0+cu130, CUDA 13.0, float64 deterministic execution on an NVIDIA GeForce RTX 4080 SUPER.

A full 2,304-run rerun is not required for the paper-level verification above and is not claimed to be bitwise portable to arbitrary hardware.

## 4. Selected diagnostics

The committed diagnostic summaries document selected local post-hoc cases. They do not estimate mechanism prevalence and do not establish family-wide causal effects. Publication-facing replay assets will be added only after their paths/provenance have been scrubbed and independently checked.

## 5. Stage C

The planned confirmatory stage was not run. The stored allocation result selected the cap of 32 corpora but estimated a worst planned half-width of approximately 0.1553, above the predeclared 0.10 target. No Stage C runs or seeds are represented as completed here.
