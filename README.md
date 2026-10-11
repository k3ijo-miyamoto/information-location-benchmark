# Information Location Benchmark

Clean reproducibility package for the study **Information Location under Matched Predictive Opportunity**.

This repository contains only publication-facing code, frozen configurations, compact evidence, and selected diagnostic states needed to inspect or reproduce the paper's claims. It intentionally excludes private authoring history and unrelated project materials.

## What can be reproduced

The public artifact is organized in three levels:

1. **Paper verification (CPU-friendly).** Recompute the exact finite-support design checks, verify the six development-stage INR contrasts, regenerate the paper tables/figures from committed evidence, and confirm that the planned confirmatory stage was not executed.
2. **Selected diagnostic replay.** Inspect the selected D1/D3 pathway-collapse evidence, frozen one-step boundaries, and bounded learning-rate interventions from published frozen states.
3. **Benchmark rerun components.** Generator 1.3, public-fact adapters, specified learner implementations, exact evaluator, deterministic seed registry, sampler, and frozen training protocol are included so the development benchmark can be rerun with suitable compute.

The paper-level reproduction path does **not** require rerunning all 2,304 learner runs.

## Scientific scope

The benchmark controls which audited public statistic determines a known binary posterior. In the three pure-relocation conditions, the full posterior marginal is held fixed while the posterior-determining statistic is moved among current state, dependency topology, and ordered history. A fourth topology-history interaction condition matches Bayes Brier opportunity but not the full posterior marginal.

All learner organizations have access to the same public semantic facts. This does not imply identical tensors, effective capacity, conditioning, or optimization difficulty.

Reported learner comparisons are development-stage results for the specified learner-optimizer systems. Selected D1/D3 diagnostics are local post-hoc mechanism probes; they are not estimates of mechanism prevalence or family-wide causal effects.

## Repository status

This is the publication-clean repository. The first public build is being assembled and verified before it is cited from the arXiv manuscript.

- Paper title: *Information Location under Matched Predictive Opportunity*
- Author: Yukitake Miyamoto
- Affiliation: Independent Researcher, Japan
- Confirmatory Stage C: not executed
- License: MIT

See `docs/REPRODUCIBILITY.md` and `docs/EVIDENCE_SCOPE.md` once the initial publication bundle is complete.
