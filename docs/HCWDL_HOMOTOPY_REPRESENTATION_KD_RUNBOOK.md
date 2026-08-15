# HCWDL Homotopy Representation-KD Runbook

This runbook operates the twenty-point v2 RSET/RREL U/D supplement (22 fits,
21 logical target banks, 47 command-plan tasks). It never submits unless
the operator supplies both the immutable creation authorization and the
separate submission phrase.

## Prerequisites

Use a clean detached worktree at the pushed commit. The parent must have
published its authenticated coupling, coordinate, endpoint-equality, and
graph-recipe locks. Parent completion and factorized logit-only reports are
not launch prerequisites: their canonical paths are frozen in the child spec
and authenticated when the final comparative aggregate runs. If the parent is
still running, only that aggregate is given an external dependency on the
parent's exact `campaign_complete` job from its authenticated submission
ledger; representation training starts immediately. Reuse a
source-compatible authorized
`HCWDL_REPRESENTATION_RECIPE/v5`, architecture attestation, full numerical
acceptance, and committed kernel envelope; path existence alone is not enough.

If those reusable assets are absent, create all four with the non-training
preparation command below. It authenticates the training-ready U/J parent and the
frozen historical unweighted HCWDL campaign. It requires their complete v4
execution policies to agree and permits only evidence/count/row-selection
lineage to differ, then runs installed-Weaver parity and numerical checks and
publishes the deterministic kernels through the committed binary envelope.
It runs no fit and reads no final-test role. A recipe mismatch fails before
the output root is created; it is not treated as approximately compatible.

```bash
python -s "${PROJECT_DIR}/scripts/prepare_hcwdl_homotopy_representation_prerequisites.py" \
  --parent-homotopy-spec "${PARENT_UJ_SPEC}" \
  --historical-campaign-root "${HISTORICAL_HCWDL_ROOT}" \
  --historical-project-dir "${HISTORICAL_PROJECT_DIR}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${SOURCE_COMMIT}" \
  --output-root "${PREREQUISITE_ROOT}" \
  --device cpu
```

The command writes `prerequisite_bundle.json`. Its `paths` mapping supplies
`REP_RECIPE`, `KERNEL_REF`, `ARCH`, `NUMERICAL`, and the already source-pinned
integration and recipe-compatibility attestations. Preserve a partially published failed root and retry
under a new root rather than overwriting it.

The prerequisite helper persists the committed-envelope coordinate in its
portable campaign-planning form:

```json
{"committed_directory":"/absolute/.../committed/<64-hex-envelope-id>"}
```

The closed HCWDL-U-RKD adapter promotes only that absolute coordinate to the
production loader's internal registered-input path type, after which the
existing commit, parent, and member validators authenticate the envelope.
Generic production adapters continue to reject unregistered raw paths.

## Local nonauthorizing closure

```bash
export PROJECT_DIR=/absolute/clean/HLT_Classification
export PYTHONPATH="${PROJECT_DIR}/src"
python -s "${PROJECT_DIR}/scripts/run_hcwdl_homotopy_representation_local_smoke.py" \
  --output /tmp/hcwdl_u_rkd_local_smoke.json
```

## Genuine smoke candidate

After installed-Weaver parity and the full numerical acceptance exist, publish
the candidate. Replace only the explicit paths. When the prerequisite helper
was used, reuse its integration attestation instead of rebuilding it:

```bash
export PROJECT_DIR=/home/ryreu/atlas/HLT_Classification_hcwdl_u_rkd_${SHORT}
export SOURCE_COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
export PARENT_UJ_SPEC=/absolute/completed/hcwdl_uj_smoke/campaign_spec.json
export REP_RECIPE=/absolute/representation_recipe_v5.json
export KERNEL_REF=/absolute/kernel_envelope_reference.json
export ARCH=/absolute/architecture_attestation.json
export NUMERICAL=/absolute/full_numerical_acceptance.json
export RECIPE_COMPATIBILITY=/absolute/recipe_compatibility.json
export CAMPAIGN_ROOT=/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_u_rkd_smoke_${SOURCE_COMMIT:0:8}_r1

python -s "${PROJECT_DIR}/scripts/validate_hcwdl_homotopy_representation_integration.py" \
  --source-commit "${SOURCE_COMMIT}" \
  --architecture-attestation "${ARCH}" \
  --numerical-acceptance "${NUMERICAL}" \
  --recipe-compatibility "${RECIPE_COMPATIBILITY}" \
  --output /tmp/hcwdl_u_rkd_integration.json

python -s "${PROJECT_DIR}/scripts/create_hcwdl_homotopy_representation_campaign.py" \
  --parent-homotopy-spec "${PARENT_UJ_SPEC}" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${SOURCE_COMMIT}" \
  --representation-recipe "${REP_RECIPE}" \
  --kernel-envelope "${KERNEL_REF}" \
  --architecture-attestation "${ARCH}" \
  --numerical-acceptance "${NUMERICAL}" \
  --recipe-compatibility "${RECIPE_COMPATIBILITY}" \
  --integration-attestation /tmp/hcwdl_u_rkd_integration.json \
  --authorize-live-submission \
  --authorization-phrase "AUTHORIZE HCWDL U RKD VALIDATION CAMPAIGN EXACT SPEC"

python -s "${PROJECT_DIR}/scripts/dry_run_hcwdl_homotopy_representation_campaign.py" \
  --campaign-spec "${CAMPAIGN_ROOT}/campaign_spec.json"
```

The dry run is nonmutating. Live smoke submission is a separate action:

```bash
python -s "${PROJECT_DIR}/scripts/submit_hcwdl_homotopy_representation_campaign.py" \
  --campaign-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --command-plan "${CAMPAIGN_ROOT}/command_plan.json" \
  --output "${CAMPAIGN_ROOT}/submission_ledger.json" \
  --submission-phrase "SUBMIT HCWDL U RKD VALIDATION CAMPAIGN EXACT SPEC" \
  --execute
```

Monitor without job-name discovery:

```bash
python -s "${PROJECT_DIR}/scripts/monitor_hcwdl_homotopy_representation_campaign.py" \
  --campaign-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --submission-ledger "${CAMPAIGN_ROOT}/submission_ledger.json" \
  --output "${CAMPAIGN_ROOT}/monitor.json"
```

If an interrupted submission has durable events but no final ledger, assemble
the authenticated submitted prefix without contacting Slurm:

```bash
python -s "${PROJECT_DIR}/scripts/assemble_hcwdl_homotopy_representation_submission_ledger.py" \
  --campaign-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --command-plan "${CAMPAIGN_ROOT}/command_plan.json" \
  --event-root "${CAMPAIGN_ROOT}/submission_events" \
  --output "${CAMPAIGN_ROOT}/submission_ledger.json"
```

`complete_submission=false` is valid only for the exact contiguous command-plan
prefix proven by those events. Monitoring marks the absent suffix as
`NOT_SUBMITTED`; failed-closure recovery replaces it. The assembler never calls
`sbatch`.

## Ragged-view preprocessing

The representation ladder uses the repaired linear-time homotopy builder.
Every required offline and HLT ragged branch is materialized a fixed number
of times per source chunk rather than once per selected row. Student train and
validation views are then cached once in process-local RAM and replayed for
all passes. Non-TOFF predecessor target jobs use the same repaired builder
while retaining their exact source-index partitioning.

The bounded ordered worker pool defaults to `SLURM_CPUS_PER_TASK`, which is
eight for the registered target/training resource class. An operator may set
`HCWDL_UJ_VIEW_BUILD_WORKERS` to a smaller positive value, but the worker
rejects a value above the Slurm allocation. Worker count changes execution
speed only: emitted identities and bytes remain in canonical source order.

Each GPU log prints these phase boundaries:

```text
HCWDL-U-RKD phase=student_view_cache ... status=started|complete
HCWDL-U-RKD phase=teacher_targets ... status=started|complete
HCWDL-U-RKD phase=optimizer_training ... status=started|complete
```

Absence of `optimizer_training ... status=started` means the job is still in
one-time preprocessing or target handling, not that an optimizer epoch has
silently begun.

## Pilot boundary

After the genuine smoke, build its measured resource profile with
`build_hcwdl_homotopy_representation_resource_profile.py`. A 300k candidate
uses an authenticated training-ready 300k U/J parent plus `--resource-profile`.
Run the complete dry run again and obtain a new explicit creation and
submission authorization. Smoke permission never carries over to the pilot.

## Failure and cancellation

Build source/resource recovery with
`recover_hcwdl_homotopy_representation_campaign.py`, then submit only that
closure with `submit_hcwdl_homotopy_representation_recovery.py`. Resource
recovery changes only the resource table; source recovery changes only the
clean pushed execution source. Both preserve the output root, graph, recipes,
targets, and seeds.

Cancellation previews exact commands by default. Execution additionally
requires `--authorization-phrase "CANCEL HCWDL U RKD EXACT IDS"`; never cancel
by broad job name.
