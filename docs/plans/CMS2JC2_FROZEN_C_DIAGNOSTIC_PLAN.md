# Frozen C response diagnostic (2026-09-24)

This is an authorized, isolated development extension, not a new response fit,
production selection or JetClass2 transfer. It supersedes the CPU development
plan only for the new `DEV_C_DIAGNOSTIC/v1` registration. Existing studies,
models, source snapshots and scheduler jobs are read-only donors.

## Question and fixed controls

Find whether continuous residual noise, angular noise, momentum noise or
shared-jet residual correlation explains C's distorted jet mass, width and pT.
Import the completed comparison's authenticated C_L model, fit/report receipts,
frozen samples and histogram ranges. Require the same CMS inputs, physical
conventions, numerical environment and unchanged scientific donor source.
No association search, calibration record construction or refit is performed.
Use all 10,000 development evaluation jets and replicas 0/1/2, including empty
outputs and jets that might not be resolvable by the association algorithm.
Original response_select, response_confirm and final test remain sealed. No
JetClass2 particles are read; intended later transfer remains the dzfix release.

Five fixed interventions, with all topology, count, PID, charge and validity
draws unchanged:

| Variant | Continuous emission change |
| --- | --- |
| FULL | Exact replay of current C |
| CENTRAL | Set every sampled continuous residual to zero |
| NO_ANGLE | Zero residual delta-eta/delta-phi coordinates only |
| NO_PT | Zero residual log-pT-ratio coordinates only |
| INDEPENDENT | Replace shared latent noise by independent per-object noise with the same total latent covariance |

CENTRAL retains learned conditional central predictions, state choices,
response-envelope clipping and identity fallbacks. It is not a conditional
physical mean after nonlinear decoding. NO_ANGLE retains predicted angular
offsets; NO_PT retains predicted momentum response. Each emission may contain
one or two output particles; interventions apply to each eight-coordinate block.
INDEPENDENT uses the square root of the sum of the stored shared and independent
factor covariances. This preserves each emission's latent marginal covariance
(including correlations between split siblings), not just its independent
variance. Residual quantile tables and scale/envelope limits are unchanged.
Frozen label-independent keys are reused; no global mutable random generator.

## Diagnostics and interpretation

Compare offline, real HLT and each intervention using the same location-fitted
histogram bins. Retain existing jet/particle/PID kinematic and tracking metrics,
conditional multiplicity, momentum, crowding and displacement cohorts, overflow
counts, empty jets and jet correlations. Report raw means and histogram TV, not
an invented production six-block score. No automatic winner or physics gate.
Replicas and constituent particles are not independent statistical samples.

Add compact instrumentation grouped by emission mechanism (singleton, merged,
split, additional), output PID and coordinate. Record scale and response clipping
counts, total coordinate exposure, pre/post sums and extrema, absolute clipping
correction sums/maxima, residual-bin backoff, missing categorical/value module,
missing emission state backend/category and low-statistics cells separately.
Support excursions are counted per input particle and feature; they indicate
out-of-support input, not necessarily actual feature clipping. Counts use
explicit emission/coordinate/input-particle/jet-replica denominators. The old
boolean-per-jet flag rates remain separately available for comparison.
Quantile-tail counters describe the residual draw before intervention; response
clipping describes the actual intervened output. Module summaries record counts,
momentum, angles and valid tracking values. Never persist per-jet traces or raw
particle/proxy arrays. Small JSON aggregates and SVG figures only.

## Execution and isolation

Fresh disjoint root and clean pushed source. Reuse the CPU development journal,
exclusive worker claims, allocation and source/environment checks, immutable
receipts, atomic publication, 12-GiB total and 2-GiB report/figure caps. All 27
jobs use tier3, reu-aisocial/qos_tier3, one task/node, **no GPU** and one CPU
with native numerical pools fixed to one. Parallelism is across jobs:

1. One acceptance job (32 GiB, 2h) exercises 32 evaluation jets x 3 replicas.
   It verifies exact FULL parity with the unchanged production generator and
   unchanged topology/PID across all five interventions. This is a correctness
   check, not performance acceptance or additional training.
2. Twenty independent evaluation jobs (32 GiB, 12h), five variants x the same
   four frozen 2,500-jet shards, afterok acceptance. Every evaluated jet-replica
   repeats production-generator parity/topology checks in RAM. FULL also checks
   its aggregate histogram against the original completed shard: bins, counts
   and identities are exact; saved floating moments allow rtol/atol 1e-10 for
   cross-node last-bit differences. In-job FULL particle replay stays exact.
3. Five reports (16 GiB, 2h), each after its four shards, with conditional plots.
4. One summary (16 GiB, 2h) after all five reports, providing a side-by-side table
   and report/figure paths. No automatic selection, follow-up fit or transfer.

The measured original C shards took about 19 minutes; diagnostic replay and
instrumentation add overhead. These are conservative requested limits, not
promised runtimes. Site test-only admission precedes any live submission. Print
progress every 100 jets. Do not allocate 36 CPUs to a serial evaluator.
Use a reviewed immutable dry plan and the explicit CDIAG authorization phrase.
No existing jobs are cancelled, resized or resubmitted. If interrupted, retain
the root and use a fresh registration, not overwritten receipts.

## Acceptance and next decision

Local tests cover numerical FULL parity, deterministic intervention masks,
marginal-covariance preservation, topology/PID invariance, missing states,
clipping denominators, authenticated frozen-model reuse, corruption, forbidden
population changes, exact DAG, dry/live journal semantics, and tiny ROOT data
through evaluation/reports. The first registered acceptance job provides the
real SPORC check; local tests do not assert it has already passed. Weaver/GPU
parity is inapplicable to this CPU-only response diagnostic.
Inspect all variants, including bad outcomes, before proposing a separate
calibration change. These already-inspected evaluation data cannot provide
independent final confirmation of the eventual mapping.

## Queue interface after committing/pushing

Use `scripts/cms2jc2_response_dev.py` in a **fresh clean detached worktree** at
the exact pushed commit. Keep the original worktree and study root available.
Activate `/home/ryreu/miniconda3/envs/atlas_kd_sporc` using its conda.sh, set
PYTHONNOUSERSITE=1 and prepend CONDA_PREFIX/lib to LD_LIBRARY_PATH. PROJECT_DIR
is the new worktree, C_COMMIT its exact commit, and C_ROOT a fresh sibling root.
Creation does not submit anything:

```bash
CLI="${PROJECT_DIR}/scripts/cms2jc2_response_dev.py"
python -s "${CLI}" create-c-diagnostic \
  --parent-spec /home/ryreu/atlas/HLT_Classification/checkpoints/cms2jc2_dev_de9890b2_r1/stages/compare_search_b1_r1/stage_spec.json \
  --project-dir "${PROJECT_DIR}" --source-commit "${C_COMMIT}" --root "${C_ROOT}"
SPEC="${C_ROOT}/stages/frozen_c_r1/stage_spec.json"
python -s "${CLI}" dry-run --spec "${SPEC}"
```

Review all 27 commands and their tier3/no-GPU allocations before submitting:

```bash
python -s "${CLI}" submit --spec "${SPEC}" --execute \
  --authorization-phrase "AUTHORIZE CMS2JC2 CPU DEVELOPMENT CDIAG EXACT PLAN" \
  --reviewed-plan-hash "${REVIEWED_PLAN_HASH}"
python -s "${CLI}" monitor --spec "${SPEC}"
python -s "${CLI}" c-results --spec "${SPEC}"
```

The reviewed hash is the `content_hash` printed by dry-run and stored in
`stages/frozen_c_r1/command_plan.json`. Progress is in that directory's
slurm-JOBID.out. Reports are `reports/frozen_c_r1/c_report_VARIANT.json`;
plots are `figures/frozen_c_r1/VARIANT/`. No large dataset is generated.
