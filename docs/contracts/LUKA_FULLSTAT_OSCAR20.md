# Luka FullStat Oscar20 contracts v1

Authority: [scientific plan](../plans/LUKA_FULLSTAT_OSCAR20_PLAN.md).

New `LUKA_FULLSTAT_{GATE,FOUNDATION,STORAGE,FILE_PREPARED,CACHE,PREFLIGHT,SPEC,PLAN,TASK,
RESULTS,SUBMISSION_CLAIM,LOCAL_WITNESS}/v1` artifacts use canonical content hashes, explicit
parent hashes and immutable publication. No old trial artifact is admissible.
LOCAL_WITNESS is explicitly non-production and cannot admit a scientific plan.

Foundation binds transfer hashes, common ROOT schema, all source hashes/latest
tree cycles, scalar cutflow, source-bin/class quotas, seed/block algorithm and
per-file scalar membership NPZs. Roles: train0, validation1, final_test2, unused3;
ineligible255. Row identity is a hash of fullstat domain/source hash/path/tree/
entry, independent of locator and role. Membership payloads contain labels,
roles and both scalar particle counts; test particles are never decoded.
Validation replays quotas and assignments from authenticated scalar metadata.
Nonfinite scalar values, unexpected codes/schema/path, duplicates and drift fail.

Preparation binds foundation/source/conventions/view/input contracts, canonical
per-source bounded chunk paths and ordered rows, labels, identities and mapping.
Each coordinate has offsets int64, features float32(P,17), vectors float32(P,4),
labels int64, identities uint8(N,32). Offsets increase by 1..512. Blocks are
completion-last and reject wrong role, source, coordinate, row joins, hashes,
shapes, nonfinite data, duplicates or missing coverage. No padded disk arrays.

`LUKA_FULLSTAT_20E_RECIPE/v1` explicitly changes the training budget/schedule;
the shared runner dispatches only this exact registered recipe (128 or256),
never mutating the old 60--100-pass recipe. Scientific reports must execute20
passes and bind the recipe, node, cache and validation population. Technical
reports are acceptance-only and can never satisfy scientific dependencies.
Checkpoint bytes and ordered selected-validation predictions are receipt-bound.
TRAIN banks bind selected teacher report/checkpoint, identities, temperature and
cache parent. No validation/test bank can supervise a student.

Preflight binds gate/cache/source, real Oscar Slurm job/runtime, installed parity,
all seven coordinate stresses, sampled miniature counts, checkpoint/bank readback,
maxima, measured timing/peaks and deterministic projections. Admission requires
every bound payload and successful exact gate accounting. No quality threshold.

Plans contain complete afterok dependencies and exact resources. Dry and live
plans must match byte-semantic content hashes. Exclusive live claims and durable
job journals prevent duplicate/ambiguous retries. Runtime source/environment and
allocation match accepted gate; completion does not depend on model accuracy.

Final test remains sealed. `event_independence_verified=false` and
`old_trial_overlap_verified=false` must remain visible in splits and reports;
row disjointness is not evidence of either claim.

## Sparse recovery / automatic follow-up extension v1

Additional canonical immutable contracts:
`LUKA_FULLSTAT_{RECOVERY_GATE,RECOVERY_PLAN,RECOVERY_SPEC,RECOVERY_CLAIM,
FOLLOWUP_REVIEW,FOLLOWUP_AUTHORIZATION,FOLLOWUP_SCIENCE_REVIEW,
FOLLOWUP_SUBMITTED}/v1`. Existing GATE/SPEC/v1 interpretation is unchanged.

RECOVERY_GATE binds the original gate/ledger/foundation file hashes, original
source592a6dac, thirteen receipt hashes, fixed retry indices12--14, node1804
exclusion, new clean pushed execution source, and unchanged scientific policy.
Preparation source compatibility is enforced against producer592a6dac and
reviewed parentbf04bb89; only named recovery orchestration files may differ
from that parent in src/scripts/sbatch. Drift fails, not silently waived.

The reused CACHE retains its ORIGINAL gate/foundation parent. The new preflight
binds RECOVERY_GATE plus that cache; RECOVERY_SPEC binds the new gate/preflight,
the old foundation locator, and the exact replacement submission ledger.
Cross-pairing an ordinary SPEC with a RECOVERY_GATE, a wrong preflight job, or
an alien source/cache is rejected. All physical payload checks still apply.

FOLLOWUP_REVIEW contains a complete science command template using a clearly
marked placeholder science identity and48h training/reducer ceilings; this is
not an admitted science spec. FOLLOWUP_AUTHORIZATION binds that review hash.
The actual admitted plan must match every template argument except its comment
identity and the measured walltimes (bounded by48h). Full exact dry/live plan
checks and journals remain mandatory. Final science submission records all16
IDs and `training_complete=false`; submission is not successful training.

Recovery admission requires exact Slurm identities (IDs, job names, owner,
account), original successful/cancelled states, and successful replacement
array elements/assembly/preflight. An array parent is never a substitute for
its elements. Unknown/missing states cannot admit science. Failures stop the
controller without cancelling unrelated work or silently retrying any task.
