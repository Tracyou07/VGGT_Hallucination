# Unified AutoDL Results Root Design

## Objective

Store every experiment artifact below `/root/autodl-tmp/results` while keeping
datasets, checkpoints, Hugging Face caches, and Git checkouts in their current
locations. Every active experiment branch must write the canonical path
directly rather than depending on compatibility links.

## Canonical Layout

Shell entry points define:

```bash
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
RESULTS_ROOT="${RESULTS_ROOT:-$AUTODL_TMP/results}"
```

Each experiment owns `$RESULTS_ROOT/{experiment}/`. Existing internal
`results/`, `state/`, `pointers/`, and auxiliary directories remain unchanged,
so migration does not rewrite immutable run artifacts. Python CLI defaults use
the same `RESULTS_ROOT` convention. Explicit user overrides continue to win.

## Migration Contract

`scripts/autodl/migrate_results_root.py` migrates an explicit allowlist of
experiment roots. It supports `--dry-run`, takes an exclusive lock, and is
idempotent after success or interruption. Before moving any data it scans every
source/destination pair. A same-path file collision is accepted only when both
files have the same SHA-256 digest; type mismatches or different content abort
the whole migration before mutation.

When only the legacy root exists, it is moved atomically. When both roots
exist, unique source entries are moved and identical duplicates are removed.
The emptied legacy root becomes an absolute symlink to the canonical root.
These links exist only so historical metadata containing legacy absolute paths
can still be inspected. Updated scripts write canonical paths directly.

The allowlist includes Camera Context, Camera Iteration, Camera Head
Amplification, Local-Global Consistency, all Camera hidden studies, Camera
Refiner data/training, and phenomenon characterization (`vggt_hallucination`).
The tool never discovers directories by glob and never touches `datasets`,
`ckpt`, `hf_home`, or repository checkouts.

## Branch Rollout

Update and independently test `main` plus all active experiment branches:

- `camera-context-consistency-preexperiment`
- `camera-iteration-preexperiment`
- `camera-head-amplification-preexperiment`
- `local-global-consistency-preexperiment`
- `camera-hidden-state-attribution-preexperiment`
- `phenomenon-characterization`
- `camera-refiner-data-construction`
- `camera-refiner-training`

Each branch receives only path-default, documentation, migration, and focused
test changes appropriate to files present on that branch. Commits are pushed
normally without force-updating branch history.

## Verification

Tests cover dry-run behavior, clean moves, partial merge recovery, identical
duplicate handling, conflicting-file refusal before mutation, symlink
idempotence, explicit allowlisting, and exclusion of non-result roots. Every
modified shell script receives syntax validation, and each branch's existing
focused CPU suite is run before push.
