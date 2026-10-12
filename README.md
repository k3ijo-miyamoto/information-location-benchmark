# Information Location Benchmark

[![verify-publication-artifacts](https://github.com/k3ijo-miyamoto/information-location-benchmark/actions/workflows/verify.yml/badge.svg)](https://github.com/k3ijo-miyamoto/information-location-benchmark/actions/workflows/verify.yml)

Publication-clean reproducibility package for **Information Location under Matched Predictive Opportunity**.

## Preprint

The paper is publicly archived on Zenodo:

- **Miyamoto, Yukitake (2026). _Information Location under Matched Predictive Opportunity_.** Zenodo. https://doi.org/10.5281/zenodo.23297061
- Version-specific DOI (v1): `10.5281/zenodo.23297061`
- Concept DOI (all versions): `10.5281/zenodo.23297060`

For reproducibility or priority-sensitive references to the October 11, 2026 release, cite the version-specific DOI above.

This repository contains only paper-facing code, frozen configurations, compact evidence, and selected diagnostic summaries. Its Git history starts with publication assembly rather than the private research/authoring history.

## Study in one paragraph

The benchmark asks whether specified learning systems behave differently when an equally predictable binary target is determined by a current observation, dependency topology, or ordered history. In IS0--IS2, the full posterior marginal is identical within each opportunity level while the posterior-determining public statistic changes. IS3 is a separate topology-history interaction extension that matches Bayes Brier opportunity but not the full posterior marginal. All learner organizations receive the same public semantic facts.

## Quick paper verification

The core paper checks are CPU-only and use the committed evidence:

```bash
python scripts/verify_paper_evidence.py
python scripts/verify_selected_diagnostics.py
python scripts/verify_official_run_metrics.py
python scripts/reproduce_tables.py
```

For the publication figures:

```bash
python -m pip install -r requirements-paper.txt
python scripts/reproduce_figures.py
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and [docs/SOURCE_PROVENANCE.md](docs/SOURCE_PROVENANCE.md).

## Reproducibility v0.2 status

The public artifact now includes:

- a sanitized per-run archive for all **2,304** Stage V development runs;
- a verifier that recomputes the stored primary INR analysis and Stage C allocation from those public records;
- published compressed D1/D3 pre-boundary optimizer/model states with compressed and decompressed SHA-256 identities;
- deterministic regeneration of the selected IS2-O1 corpus;
- strict D1/D3 continuation replay scripts for the certified original CUDA environment;
- source-to-public provenance manifests and publication-cleanliness checks;
- CPU CI covering paper evidence, per-run archive reconstruction, frozen-state identity checks, tests, and table/figure regeneration.

The full selected optimizer continuations are not run in hosted CPU CI because the paper does not claim bitwise portability beyond the certified original GPU/software environment.

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

Selected intervention replays use the published frozen prestates:

```bash
python scripts/replay_d1_persistent_lr.py --output /tmp/d1-persistent.json
python scripts/replay_d1_one_step_lr.py --output /tmp/d1-one-step.json
python scripts/replay_d3_persistent_lr.py --output /tmp/d3-persistent.json
```

These strict replays require the same certified GPU/software environment as the original selected continuations.

## Reproduction levels

1. **Paper verification:** exact design and reported aggregate values from committed evidence; CPU-friendly.
2. **Selected diagnostics:** committed selected-case summaries, exact corpus regeneration, published D1/D3 frozen prestates, and strict continuation replay scripts for the certified GPU environment.
3. **Official training rerun:** source/configuration and sanitized 2,304-run public metrics are provided; strict training/replay requires the pinned official GPU/software environment.

## Scope boundaries

This repository does not claim that one representation family is universally superior, that the observed rankings isolate causal architecture effects, or that selected diagnostic cases explain the whole benchmark. The construction-inspired vocabulary is synthetic and does not establish real construction-domain performance.

## Metadata

- Author: Yukitake Miyamoto
- Affiliation: Independent Researcher, Japan
- License: MIT
- Confirmatory Stage C: not executed
