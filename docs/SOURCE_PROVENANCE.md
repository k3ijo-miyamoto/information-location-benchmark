# Source provenance

This repository is a publication-clean reconstruction of the research artifact
used for *Information Location under Matched Predictive Opportunity*.

## Development source anchor

Scientifically material files were selected from the private development state
anchored at commit:

`d2170862637668300a337e561aef77e3c62ba0b2`

The machine-readable mapping is:

`provenance/source_manifest.json`

For each mapped artifact it records the original source path, the source Git
blob SHA, the public path, the public Git blob SHA, and whether the copy is
byte-identical.

## Publication adaptations

A small set of files was intentionally adapted for publication rather than
copied byte-for-byte. Adaptations are limited to packaging and provenance
boundaries, for example:

- package initializers no longer expose unrelated legacy modules;
- the seed helper retains the deterministic Stage V derivation while removing
  historical registry dependencies not needed for this paper;
- the Stage V common helper retains public identities/constants while removing
  private historical authority-chain checks;
- the public Stage V execution configuration removes machine-local path
  metadata and unrelated deferred-realism bookkeeping;
- one canonical dependency comment was generalized without changing executable
  semantics.

The scientific objects that must not change under these adaptations are the
posterior formulas, finite support, split policy, seed values/derivation,
learner dimensions, optimizer protocol, exact evaluator, INR estimands, and
reported evidence values.

## Evidence identity

The original complete Stage V results object contained 2,304 successful records.
Its compressed Git object and decompressed JSON identity are recorded in
`evidence/stage_v/source_hashes.json`.

The full raw archive is intentionally not exposed from the private development
repository. A sanitized per-run publication archive remains a follow-up item.
The current public repository therefore makes aggregate Stage V evidence and
paper plotting/table inputs available, and says so explicitly.

Selected D1/D3 optimizer prestates are similarly identified in
`evidence/diagnostics/frozen_state_manifest.json`. Their binary archives are
not currently mirrored publicly.

## Clean-history rule

This repository was created with fresh Git history. Private authoring notes,
assistant instructions, interview-origin files, superseded manuscript history,
and unrelated development tracks are not part of the publication artifact.
