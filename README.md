# Information Location Benchmark

Publication-clean reproducibility package for **Information Location under Matched Predictive Opportunity**.

This repository contains only paper-facing code, frozen configurations, compact evidence, and selected diagnostic summaries. Its Git history starts with publication assembly rather than the private research/authoring history.

## Study in one paragraph

The benchmark asks whether specified learning systems behave differently when an equally predictable binary target is determined by a current observation, dependency topology, or ordered history. In IS0--IS2, the full posterior marginal is identical within each opportunity level while the posterior-determining public statistic changes. IS3 is a separate topology-history interaction extension that matches Bayes Brier opportunity but not the full posterior marginal. All learner organizations receive the same public semantic facts.

## Quick paper verification

The core paper checks are CPU-only and use the committed evidence:

```bash
python scripts/verify_paper_evidence.py
python scripts/reproduce_tables.py
```

For the publication figures:

```bash
python -m pip install -r requirements-paper.txt
python scripts/reproduce_figures.py
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Evidence layers

- `evidence/design/` — exact finite-support/posterior and adapter/split evidence.
- `evidence/stage_v/` — development-stage primary analysis and Stage C allocation decision.
- `evidence/paper/` — paper table/figure inputs.
- `evidence/diagnostics/` — selected post-hoc diagnostic and intervention summaries.

The primary learner results are development-stage results for the specified learner-optimizer systems. Stage C was not executed. The selected diagnostic cases are local post-hoc probes and do not establish mechanism prevalence or family-wide causal effects.

## Benchmark implementation

The repository includes Generator 1.3, public-fact adapters, the specified tabular/static/temporal learner implementations, exact expected-risk evaluation, deterministic corpus sampling, the frozen seed registry, and the official Stage V training implementation.

A single frozen run can be rerun with:

```bash
python scripts/rerun_single_official.py \
  --cell IS2-O1 --corpus 0 --N 8192 --init 1 --family tabular
```

The official training implementation intentionally requires the certified original CUDA environment; paper-level verification does not.

## Reproduction levels

1. **Paper verification:** exact design and reported aggregate values from committed evidence; CPU-friendly.
2. **Selected diagnostics:** committed selected-case summaries and publication-facing trajectory data.
3. **Official training rerun:** source/configuration is provided; requires the pinned official GPU/software environment for the strict path.

## Scope boundaries

This repository does not claim that one representation family is universally superior, that the observed rankings isolate causal architecture effects, or that selected diagnostic cases explain the whole benchmark. The construction-inspired vocabulary is synthetic and does not establish real construction-domain performance.

## Metadata

- Author: Yukitake Miyamoto
- Affiliation: Independent Researcher, Japan
- License: MIT
- Confirmatory Stage C: not executed
