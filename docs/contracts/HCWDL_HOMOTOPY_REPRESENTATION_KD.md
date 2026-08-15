# HCWDL Homotopy Representation-KD Contract

This document is the reusable contract summary for the validation-only
campaign specified by
[`HCWDL_HOMOTOPY_REPRESENTATION_KD_IMPLEMENTATION_PLAN.md`](../plans/HCWDL_HOMOTOPY_REPRESENTATION_KD_IMPLEMENTATION_PLAN.md).
The plan remains the scientific authority.

## Immutable scientific identity

- This document describes the twenty-point v2 family. It supersedes the
  unexecuted ten-point v1 candidate without making v1 artifacts reusable as
  v2.
- Two cold, factorized tracks are registered: `F_RSET` and `F_RREL`.
- Each track has exactly 11 fits:
  `U020,U040,U060,U080,U100,D80,D60,D40,D20,D0,M1`.
- `U000` is a semantic projected-offline endpoint and is never trained.
- The two `U020` fits share one native-TOFF target bank. Every later fit uses
  a compact bank produced by its own immediate predecessor. There are exactly
  22 fits and 21 logical banks.
- Student/input/sampler seeds are paired with the imported logit-only
  factorized track. Representation-head RNG is strategy-specific.
- Every fit is a fresh initialization. U/D fits use unweighted
  `0.25 CE + 0.75 KD` at temperature 2; M1 uses temperature 1. RSET and RREL
  each add the v5 scheduled representation objective with coefficient 0.10.
- Scientific mode uses 300,000 train and 100,000 validation jets, 60 passes,
  validation after every pass, no performance early stop, and macro-OVR-AUC
  first checkpoint selection. Smoke uses the authenticated parent smoke's
  4,096/4,096 population and exactly two optimizer updates.
- No final-test task or final-role capability exists. Only D0 and M1 are
  HLT-compatible; the registered screen outputs are the two M1 models.

## Lineage and storage

Artifacts use the `HCWDL_HOMOTOPY_REPRESENTATION_*/v2` contracts enumerated
in the plan and implemented in
`hcwdl_homotopy_representation_contracts.py`. The campaign imports an exact,
training-ready U/J parent and reopens its coupling, coordinate, endpoint,
graph, recipe, M0, and TOFF evidence. Complete factorized logit-control reports
and parent completion are required by the comparative aggregate, not by the
two representation-training paths.

Homotopy student views are reconstructed once per job into process-local RAM.
No repaired particle dataset is durable. Target banks contain FP32 logits,
jet embeddings, fixed-kernel set/relation sketches, eligibility/support
metadata, and canonical identities only. Target shards are byte-hashed and
their manifest binds the complete train population. Validation and final-test
targets are forbidden. Payload cleanup requires every registered consumer's
completed report and leaves its authorization, manifest, and completion
evidence.

Endpoint construction materializes each required ragged branch a fixed number
of times per selected source chunk. Bounded parallel chunk construction may
use no more than the authenticated Slurm CPU allocation and must emit the same
canonical identity order and bytes as the one-worker reference path. Exact
source-index target partitions remain unchanged.

## Execution and recovery

The immutable graph contains 47 tasks: authentication, graph/recipe lock,
21 target generations, 22 fits, aggregate, and completion. RSET and RREL
start together after the shared TOFF bank and remain sequential internally.
Finite poor metrics never prune descendants.

Training/target jobs request 8 CPUs, 96 GiB, six hours, and one GH200;
checkpointable jobs request `B:USR1@120`. Pilot creation additionally requires
a measured genuine-smoke resource profile. Source recovery and resource-only
recovery replace only the exact failed/downstream closure and preserve valid
artifacts and the scientific graph. Cancellation is by campaign-ledger IDs
only.

Implementation, dry run, and local smoke do not authorize Slurm submission.
Creation and submission require the two distinct exact phrases defined in the
contract module. Final-test access is never authorized by this campaign.
