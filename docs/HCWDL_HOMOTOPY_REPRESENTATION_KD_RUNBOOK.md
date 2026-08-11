# HCWDL Homotopy Representation-KD Runbook

This runbook operates the RSET/RREL U/D supplement. It never submits unless
the operator supplies both the immutable creation authorization and the
separate submission phrase.

## Prerequisites

Use a clean detached worktree at the pushed commit. The parent must be a
completed HCWDL structural-feature-homotopy smoke or pilot with the exact
factorized logit-only reports. Reuse a source-compatible authorized
`HCWDL_REPRESENTATION_RECIPE/v5`, architecture attestation, full numerical
acceptance, and committed kernel envelope; path existence alone is not enough.

The kernel reference JSON has one of the existing loader forms:

```json
{"committed_directory":"/absolute/.../committed/<64-hex-envelope-id>"}
```

## Local nonauthorizing closure

```bash
export PROJECT_DIR=/absolute/clean/HLT_Classification
export PYTHONPATH="${PROJECT_DIR}/src"
python -s "${PROJECT_DIR}/scripts/run_hcwdl_homotopy_representation_local_smoke.py" \
  --output /tmp/hcwdl_u_rkd_local_smoke.json
```

## Genuine smoke candidate

After installed-Weaver parity and the full numerical acceptance exist, publish
the integration attestation and candidate. Replace only the explicit paths:

```bash
export PROJECT_DIR=/home/ryreu/atlas/HLT_Classification_hcwdl_u_rkd_${SHORT}
export SOURCE_COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
export PARENT_UJ_SPEC=/absolute/completed/hcwdl_uj_smoke/campaign_spec.json
export REP_RECIPE=/absolute/representation_recipe_v5.json
export KERNEL_REF=/absolute/kernel_envelope_reference.json
export ARCH=/absolute/architecture_attestation.json
export NUMERICAL=/absolute/full_numerical_acceptance.json
export CAMPAIGN_ROOT=/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_u_rkd_smoke_${SOURCE_COMMIT:0:8}_r1

python -s "${PROJECT_DIR}/scripts/validate_hcwdl_homotopy_representation_integration.py" \
  --source-commit "${SOURCE_COMMIT}" \
  --architecture-attestation "${ARCH}" \
  --numerical-acceptance "${NUMERICAL}" \
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

## Pilot boundary

After the genuine smoke, build its measured resource profile with
`build_hcwdl_homotopy_representation_resource_profile.py`. A 300k candidate
uses an authenticated completed 300k U/J parent plus `--resource-profile`.
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
