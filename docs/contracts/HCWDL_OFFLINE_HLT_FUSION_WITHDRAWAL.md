# HCWDL offline/HLT fusion-withdrawal contract v1

This contract binds the complete eleven-fit experiment described by
`docs/plans/HCWDL_OFFLINE_HLT_CONCATENATION_FUSION_WITHDRAWAL_IMPLEMENTATION_PLAN.md`.

The privileged oracle input is the authenticated native-order sequence of all
offline particles followed by all raw HLT particles, padded to 496 without
matching, deduplication, sorting, or truncation. Concatenation and symmetric
oracle controls may use raw HLT particles beyond index 199. Every anchored
primary route, direct-KD student, validation endpoint, and extracted endpoint
uses only the first 200 canonical native-order HLT particles. Matching indices,
degradation coordinates, and content-source codes are never ordinary model
features.

The graph contains exactly eight CE-only 60-pass oracle fits and three
100-pass Study-C fits. `ANCHORED_FUSION_OH` is the fixed teacher regardless of
its measured performance. The direct student uses C25/P75 temperature-2 logit
KD. The cosine and step withdrawal students use the exact alpha schedules and
six loss terms frozen in the plan, validate and select only on alpha zero, and
publish a separate ordinary three-input HLT checkpoint after a parity audit.
Scientific metrics never gate later registered rows or completion.

All particle views and hidden states are RAM/device-only and regenerated once
per job. Only selected/final model envelopes, identity digests, 15-class
teacher probabilities, reports, locks, ledgers, inventories, attestations, and
logs are durable. Optimizer state and rolling resumes are forbidden. Recovery
restarts incomplete tasks from update zero and cancels or addresses jobs only
by exact campaign-bound IDs.

The three-task gate is `authenticate -> capacity_audit -> preflight`. The
preflight must use installed Weaver on a genuine GH200, exercise all eight
oracle architectures, a nonzero withdrawal transition, cross-residual
gradients, and exact alpha-zero extraction. Only a pushed, source-pinned
worktree whose gate artifacts and attestation validate may submit the
sixteen-task science stage. Train and validation are the only accessible
roles; final test remains inaccessible.
