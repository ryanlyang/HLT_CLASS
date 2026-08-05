# PMARD Implementation Mandate

Open this repository and implement the complete campaign specified in:

```text
docs/plans/SCOUTING_ALPHA_REPAIR_DISTILLATION_OPTIMAL_CAMPAIGN.md
```

Act as the lead ML research engineer. Do not build a toy, notebook-only proof,
or disconnected scaffold. Continue until PMARD is locally implementation-
complete and ready for its first genuine research-compute miniature.

## Authority

For this task, PMARD is the active scientific implementation plan. This prompt
explicitly authorizes updating its status and `docs/plans/README.md` accordingly.
Preserve all unrelated campaign work.

Use this authority order:

1. the PMARD implementation plan;
2. versioned reusable contracts;
3. an immutable campaign specification;
4. `AGENTS.md`;
5. `docs/HANDOFF.md`.

PMARD must remain independent of PRAD-specific modules, artifacts, and
scientific semantics. Extract only genuinely reusable neutral behavior, without
repairing or changing PRAD as part of this task.

Before editing code, read completely:

- `AGENTS.md`;
- `README.md`;
- `docs/HANDOFF.md`;
- `REPOSITORY_TRANSFER_PLAN.md`;
- the PMARD plan and reusable data/experiment/testing contracts;
- `C:\Users\22rya\ComputerScience\FCV\SCOUTING_AK8_USAGE_GUIDE.md`;
- relevant files in `C:\Users\22rya\ComputerScience\FCV\scouting_ak8_tools`.

Inspect `git status --short --untracked-files=all`, preserve existing work, run
focused tests, then create and maintain a complete implementation plan.

Treat `scouting_ak8_tools` as a donor/reference, not a runtime dependency.
Record every migrated donor file and exact donor commit in
`docs/LEGACY_SOURCE_MAP.md`. If exact provenance is unavailable, do not copy
the file or invent provenance.

## Non-negotiable requirements

- Deployable inference accepts only canonical HLT features, four-vectors, and
  masks and produces 15 logits.
- Offline data, matches, repair metadata, alpha, labels, and teacher outputs
  are training-only and structurally inaccessible at inference.
- Constituent matching follows the plan exactly: type-compatible candidates,
  global assignment, dustbins, abstention, matching-only selection, and
  five-fold out-of-file assignments for teacher-training rows.
- Never force complete matching coverage or silently admit split/merge,
  cross-category, charge-flip, or lost-track matches.
- Implement the exact `P4_ONLY/v1` endpoints and 21-channel repair/retain map.
- Preserve HLT particle count, ordering, mask, padding, categorical fields,
  and unmatched particles for every alpha.
- Every alpha uses the same locked matches. `alpha = 0` must be byte-identical
  to canonical HLT input.
- All deployable comparisons consume the same authenticated HLT view.
- Final-test access remains sealed behind finalist and execution locks.
- Poor scientific performance is a valid result and never cancels a
  registered experiment. Invalid inputs, nonfinite required quantities,
  corrupt lineage, and forbidden access fail closed.

Do not silently simplify or change scientific meaning. If a genuine semantic
change is required, revise/version the plan and contract before implementing it.

## Implementation scope

Implement the whole pipeline under the repository’s normal structure:

1. **Contracts and provenance** — Scouting schema, identity, splits, inputs,
   matching, repair, checkpoints, reports, campaign specs, locks, hashes,
   atomic publication, and validation.
2. **Streaming data foundation** — exact selection/classes, file-disjoint
   splits, projected Uproot/Awkward reads, deterministic worker/rank sharding,
   class weights, masks, and synthetic ROOT tests.
3. **Scouting ParT** — exact 21-channel CMSSW preprocessing, 15-class canonical
   Particle Transformer, training, inference, metrics, exact resume, and an
   installed-Weaver FP32 parity command.
4. **Matching** — transparent physics baselines, likelihood scoring,
   Hungarian and unbalanced optimal transport, contextual matcher, synthetic
   known-correspondence validation, cross-fitting, calibration, abstention,
   operating points, and immutable matcher lock.
5. **Alpha repair** — exact on-the-fly repair plus all required matching,
   corruption, direction, response, track, and shuffled controls.
6. **PMARD training** — alpha teachers, CE/self-KD/dual-KD losses, `K0`–`K6`,
   representation variants `R0`–`R5`, matched-capacity controls, and three
   fixed generations with exact resume.
7. **Evaluation and selection** — every metric, stratification, privilege
   recovery statistic, five-seed paired bootstrap, validation selector,
   deployability audit, finalist lock, and final-test claim required by the
   plan.
8. **Execution layer** — configuration-driven registry, thin CLIs, RAM-first
   storage profile, storage/resource measurement, Slurm DAG, dry run,
   monitoring, exact resume, and exact-ID recovery/cancellation.

Keep reusable logic under `src/hlt_classification/`; keep CLIs and Slurm workers
thin. Do not create a separate script for every experiment.

## Storage and compute

ROOT remains authoritative. Do not create a second full dataset, persistent
padded tensors, repaired dataset, dense pair cache, full assignment cache, or
durable training teacher-output cache.

Use bounded projected streaming. Keep compact match assignments and teacher
targets in shared RAM or a validated job-local RAM-backed directory, joined by
canonical row identity and deleted on job exit. Retain only declared manifests,
reports, locks, compact checkpoints, predictions, and execution evidence.

Tigris workers must use:

```text
account        reu-aisocial
partition      tigris
project        /home/ryreu/atlas/HLT_Classification
data           /home/ryreu/cms/data/ScoutingAK8_native_compact/2024/train
environment    atlas_kd_tigris
```

They must set `PYTHONNOUSERSITE=1`, set `PYTHONDONTWRITEBYTECODE=1`, prepend
`${CONDA_PREFIX}/lib` to `LD_LIBRARY_PATH`, and source helpers through absolute
`${PROJECT_DIR}` paths.

## Verification and working style

Work through the implementation continuously. Do not stop after scaffolding or
ask about minor choices that the plan already resolves. After each material
block:

- run focused tests;
- update `docs/HANDOFF.md` with evidence;
- report changed files, donor provenance, contracts, tests, external
  validation, and the exact next task;
- proceed while safe in-scope work remains.

Complete the full verification ladder:

- contract/numerical tests;
- synthetic ROOT, streaming, matching, repair, model, KD, and resume tests;
- corruption, forbidden-access, failure-injection, and source-drift tests;
- CLI, shell, campaign, storage, lock, and dry-run tests;
- bounded synthetic end-to-end teacher/student smoke;
- complete local pytest suite with caches disabled;
- `git diff --check` and source-hygiene checks;
- standalone installed-Weaver validation when available;
- genuine Tigris miniature only after explicit submission authorization.

Never fabricate real data, checkpoints, manifests, resource measurements, or
scientific results. Do not weaken tests to make code pass.

## Authorization boundary and completion

This prompt authorizes complete local implementation and preparation of the
research-compute execution layer. It does not authorize modifying the immutable
dataset, deleting unrelated work, pushing/committing, submitting the full
campaign, or unsealing final test.

Prepare the genuine Tigris miniature and exact command, but submit it only with
explicit user authorization. Full production requires separate authorization
after the miniature, measured resources/storage, exact committed source, and a
complete dry run.

Do not finish while safe local implementation work remains. The terminal state
is:

> Locally implementation-complete and research-compute execution-ready; real
> installed-Weaver/Tigris acceptance pending if not yet authorized.

At that point, provide a concise readiness report and the exact next command
requiring authorization.

Begin by completing the required reading and repository audit, then proceed
directly into contracts and the Scouting data foundation.
