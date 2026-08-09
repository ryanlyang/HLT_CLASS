# HCWDL corrected-parent prefix runbook

This runbook covers the non-final 60-pass HCWDL workflow used to produce a
corrected train/validation parent for a separately authorized downstream
import. It does not authorize a launch. It never registers an execution lock
or final-test work and it is not a completed HCWDL pilot.

The immutable identities are:

```text
campaign:       HCWDL_CAMPAIGN_SPEC/v8
execution scope: parent_prefix_through_finalist_lock
command plan:    HCWDL_COMMAND_PLAN/v4
authorization:   HCWDL_SUBMISSION_AUTHORIZATION/v8
terminal task:   finalist_lock
```

Existing `HCWDL_CAMPAIGN_SPEC/v7` artifacts retain their original full-DAG
meaning. Do not edit or relabel one as a prefix; v7 and smoke artifacts are
categorically ineligible for HCWDL-RKD parent import. The RKD import also
requires `preauthorized_automatic`; a manual-continuation v8 prefix cannot be
used as its parent.

## Boundary

The v8 registry contains the canonical source, split, train/validation row
selection, matching, assignment, cache miniature, v4 recipe, endpoint
qualification, all 23 screening nodes, screen aggregate, canonical
confirmation registry and runs, confirmation aggregate, and finalist lock.
Every non-smoke training run uses 60 complete natural train-role passes and
validates after every pass.

The registry contains none of:

```text
execution_lock
test_row_selection
assign_test
test_assignment_manifest
sealed_final_evaluation
aggregate_report
```

The campaign may retain the registered mode's sealed `final_test` count as
population metadata. Both `execution_lock_authorized` and
`final_test_access_authorized` are false.

## Candidate and dry run

Use a dedicated clean pushed worktree and a new campaign root. A non-smoke
candidate requires the exact primary `HCWDL_RECIPE/v4` and measured resource
profile before it can be authorized.

```bash
python -s scripts/create_hcwdl_campaign.py \
  --mode <pilot-or-named-midscale-mode> \
  --campaign-root <new-parent-root> \
  --source-manifest <source-manifest.json> \
  --split-manifest <split-manifest.json> \
  --data-root <read-only-data-root> \
  --source-commit <full-pushed-commit> \
  --recipe <primary-v4-recipe.json> \
  --project-dir <exact-clean-worktree> \
  --resource-measurement-sha256 <resource-profile-content-hash> \
  --resource-profile <resource-profile.json> \
  --endpoint-continuation preauthorized_automatic \
  --execution-scope parent_prefix_through_finalist_lock \
  --output <new-parent-root>/review/campaign_candidate.json

python -s scripts/dry_run_hcwdl_campaign.py \
  --campaign-spec <new-parent-root>/review/campaign_candidate.json \
  --output <new-parent-root>/review/dry_run_ledger.json
```

Before requesting authority, independently reopen the candidate and command
plan and require:

```text
contract == HCWDL_CAMPAIGN_SPEC/v8
mode != smoke
execution_scope == parent_prefix_through_finalist_lock
training_passes == 60
validation_every_passes == 1
tasks[-1].task_id == finalist_lock
no forbidden task ID is present
```

## Separate authorization and submission gates

Only after the user explicitly authorizes the reviewed exact candidate, build
the authorization with this literal phrase:

```text
AUTHORIZE EXACT HCWDL PARENT PREFIX FOR TIGRIS
```

```bash
python -s scripts/build_hcwdl_submission_authorization.py \
  --campaign-spec <new-parent-root>/review/campaign_candidate.json \
  --authorization-phrase "AUTHORIZE EXACT HCWDL PARENT PREFIX FOR TIGRIS" \
  --output <new-parent-root>/review/submission_authorization.json
```

Recreate the same identity at the canonical
`<new-parent-root>/campaign_spec.json` with
`--authorize-live-submission` and
`--submission-authorization <.../submission_authorization.json>`. Validation
must show that the candidate and live command-plan hashes are identical.

Authorization-artifact creation is not submission authority. An automatic
endpoint-continuation prefix may be submitted only after the user separately
issues this exact instruction:

```text
SUBMIT HCWDL EXACT PARENT PREFIX WITH PREAUTHORIZED ENDPOINT CONTINUATION
```

The corresponding submit command is:

```bash
python -s scripts/submit_hcwdl_campaign.py \
  --campaign-spec <new-parent-root>/campaign_spec.json \
  --output <new-parent-root>/submission_ledger.json \
  --execute \
  --authorization-phrase \
    "SUBMIT HCWDL EXACT PARENT PREFIX WITH PREAUTHORIZED ENDPOINT CONTINUATION"
```

Do not run that command merely because this runbook exists.

For `manual_posthoc`, initial submission instead requires the exact phrase
`SUBMIT HCWDL EXACT PARENT PREFIX`. After the endpoint evidence has been
reviewed and its acknowledgement published, the continuation command requires
the separate exact phrase:

```text
CONTINUE HCWDL PARENT PREFIX AFTER ENDPOINT ACK
```

That phrase releases only the remainder of the already authenticated v8
prefix. It cannot introduce an execution lock or final-test task.

## Monitoring and exact recovery

Monitoring reopens the v8 spec and exact submission ledger:

```bash
python -s scripts/monitor_hcwdl_campaign.py \
  --campaign-spec <new-parent-root>/campaign_spec.json \
  --submission-ledger <new-parent-root>/submission_ledger.json \
  --query-slurm \
  --output <new-parent-root>/recovery/monitor.json
```

Render a nonmutating recovery plan first:

```bash
python -s scripts/resume_hcwdl_campaign.py \
  --campaign-spec <new-parent-root>/campaign_spec.json \
  --submission-ledger <new-parent-root>/submission_ledger.json \
  --monitor-report <new-parent-root>/recovery/monitor.json \
  --output <new-parent-root>/recovery/resume_dry_run.json
```

An actual exact-ID recovery additionally requires the separate phrase:

```text
RESUME HCWDL EXACT PARENT PREFIX TASKS
```

The recovery closure is reconstructed only from the v8 prefix graph, so it
cannot introduce execution-lock or final-test tasks.

## Completion evidence

Prefix completion requires every registered task and array row to be terminal
and artifact-valid, the canonical confirmation aggregate to validate, and
`locks/finalist.json` to bind the exact corrected campaign. The root must not
contain a prefix-produced execution lock or final-test artifact. Downstream
consumers must explicitly accept the v8 prefix contract and independently
reopen the corrected loss-semantics, reports, checkpoints, and complete lock
chain; path existence alone is insufficient.
