# Agent Instructions

## Mission

This repository develops reproducible HLT-only JetClass classifiers and
training methods that may use richer offline information only through
explicitly declared training-time supervision or oracle controls.

The canonical initial model is a from-scratch HLT Particle Transformer. The
first planned research campaign is HLT Offline Structure Distillation (HOSD).

## Required reading order

Before changing the repository:

1. read `README.md`;
2. read `docs/HANDOFF.md`;
3. read `REPOSITORY_TRANSFER_PLAN.md` and the active block or campaign plan;
4. read the relevant reusable contracts;
5. inspect `git status --short --untracked-files=all`;
6. run the focused tests for the area being changed.

## Documentation authority

When documents differ, use this order and reconcile the conflict explicitly:

1. active scientific implementation plan;
2. versioned reusable contracts;
3. immutable `campaign_spec.json` for a particular execution;
4. this `AGENTS.md`;
5. the living status in `docs/HANDOFF.md`.

`docs/HANDOFF.md` reports implementation state. It may not silently redefine
scientific meaning.

## Scientific invariants

- Deployable inference consumes HLT inputs only.
- Offline inputs may appear only in roles explicitly authorized by a plan:
  training target, teacher, control, or oracle diagnostic.
- The initial `fixed_hlt_v3_track_dominant_proxy` is a controlled HLT-like
  proxy, not genuine detector HLT.
- All models in a comparison consume identical authenticated HLT views.
- Degradation construction/source indices are forbidden from deployable model
  inputs.
- Final-test inference is sealed until the required finalist and execution
  locks exist.
- Poor accuracy, rejection, calibration, or target quality is a scientific
  result. It must never fail a job, cancel later registered experiments, or
  skip a row.
- Invalid inputs, forbidden data access, nonfinite required quantities, stale
  source, corrupt artifacts, and lineage mismatches fail closed.

## Implementation rules

- The new package must not import `Fresh_check` at runtime.
- Put reusable semantics under `src/hlt_classification/`; keep CLIs and Slurm
  workers thin.
- Record every migrated donor file and donor commit in
  `docs/LEGACY_SOURCE_MAP.md`.
- Version schemas and contracts. If scientific semantics change, bump the
  appropriate version and tests rather than making old and new artifacts look
  interchangeable.
- Reusable artifacts require content hashes, parent hashes, source lineage,
  validation before reuse, and atomic publication.
- Path existence alone never proves an artifact is reusable.
- Do not use timestamps as scientific identities.
- Preserve unrelated user work. Never reset, clean, or overwrite an existing
  dirty worktree to simplify a task.

## Testing expectations

Use the ladder in `docs/TESTING.md`. At minimum, each change receives focused
local tests. Production readiness additionally requires authoritative
installed-Weaver parity and a genuine Tigris miniature using production
workers.

Standard local command:

```powershell
python -m pytest -q
```

Use `PYTHONDONTWRITEBYTECODE=1` when source snapshots or tracked bytecode could
otherwise change during tests.

## Research-compute rules

- Tigris account: `reu-aisocial`.
- Tigris partition: `tigris`.
- Repository path: `/home/ryreu/atlas/HLT_Classification`.
- Conda environment: `atlas_kd_tigris`.
- JetClass data are read-only under `/home/ryreu/atlas/PracticeTagging/data`.
- Every worker must use `PYTHONNOUSERSITE=1` and prepend
  `${CONDA_PREFIX}/lib` to `LD_LIBRARY_PATH`.
- Slurm workers source helpers from the absolute `${PROJECT_DIR}` path, never
  relative to the spool copy of a submitted script.
- Do not submit a full campaign without explicit authorization, a full dry
  run, measured resources, exact pushed source, and a real miniature.
- Cancel jobs by exact campaign-bound IDs, never broad ambiguous names.

## Completion report

After each transfer block or material campaign step, report:

- files changed;
- donor files and donor commit;
- contracts added or versioned;
- local test results;
- real-Weaver/Tigris validation, when applicable;
- remaining blocker or exact next task.

Update `docs/HANDOFF.md` with evidence, not aspiration.
