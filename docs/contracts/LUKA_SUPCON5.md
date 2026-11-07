# OFFLINE SupCon-5 transfer contracts v1

Authority: [scientific plan](../plans/LUKA_SUPCON5_DEBUG_PLAN.md). This is a
separate two-fit SPORC/debug execution, not a revision of existing campaigns.

The `LUKA_SUPCON5_<KIND>/v1` family uses schema version 1, canonical content
hashes, exact parent hashes and the existing ALLJETS evaluation disclosures.
Kinds are RECIPE, SPEC, PLAN, REVIEW, DISPATCH, CLAIM, RUN_CLAIM, RESOURCES,
ACCEPTANCE, TECHNICAL_BANK, STARTUP, EPOCH, TRAINING_REPORT, FIT, TASK and RESULTS.
The original U100 bank and exported OFFLINE bank retain their existing versioned
contracts; they must pass the original validators before reuse.

| Artifact | Required binding and checks |
|---|---|
| RECIPE | Original U100 recipe hash; teacher CE1+SupCon5; student CE1+U1005+OFFLINE5; exact projection, temperature, reductions, seeds and native export semantics. |
| SPEC | Clean pushed source snapshot, authenticated completed control job21993056, original OFFLINE node, original populations, two fresh fit nodes, immutable policy and separate output root. |
| RESOURCES / ACCEPTANCE | Installed environment, A100 identity, real two-epoch worker miniatures, native initialization/reload parity, longest-support/full/partial steps and warmed measurements. Acceptance is recomputed from the diagnostic, not trusted by existence. |
| PLAN / REVIEW | Exact commands, resources and dependencies. Setup and complete upper-bound science plans have dry journals before any submission. Measured release may only reduce fit walltimes. |
| EPOCH / TRAINING_REPORT | Spec/recipe/target-bank parents, task and node, ordered epoch/update schedule, weighted losses, validation selection, population and technical/scientific distinction. |
| FIT / TASK | Selected native/full checkpoints, predictions, report and required setup/bank/startup artifacts. Every file is size/hash checked. Task receipt also binds the exact accepted journal job ID and installed environment. |
| STARTUP | Real student target-bank hashes, memory measurements and explicit disposal of technical weights before scientific training. |
| RESULTS | Authenticated prediction arrays on identical ordered validation identities/labels; oracle labels; historical M0HLT/OFFLINE recovery and U100 deltas. Original50k is overlapping, report-only validation. |

Teacher targets come only from selected-checkpoint OFFLINE TRAIN inference in
FP32. The transfer tap is the native 128D preclassifier embedding. Its projection
is not exported as the target. TRAIN-only mean and population standard deviation
use FP64 with the existing 0.001 scale floor. Exact identity and label joins bind
the OFFLINE targets to D000 TRAIN. No validation or final-test target may enter
student training.

Native inference accepts HLT only. Projection/prediction heads are training-only;
native checkpoint reload must reproduce selected validation predictions and
metrics. Target quality, classification quality and recovery have no admission
threshold. Invalid inputs, nonfinite execution, failed replay, corrupt artifacts
and lineage/resource mismatches stop execution.

The setup uses zero-valued technical OFFLINE student targets solely to exercise
the extra head before the teacher exists. The scientific student additionally
runs a miniature with the actual selected-teacher targets, then resets to its
registered fresh initialization. Technical fits are never scientific results.

Live submissions use the existing exact-DAG journal contract. An exclusive live
claim precedes scheduler contact; ambiguous partial submission must be reconciled
from its journal, never blindly repeated. The student has an `afterok` teacher
dependency and cancellation on invalid dependency. The report follows both fits
with `afterany` and can list incomplete results. Historical files and jobs are
never modified by this campaign.
