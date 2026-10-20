# LUKA_EMBEDDING_LADDER_TIER3_* /v1

Authority: [transfer plan](../plans/LUKA_EMBEDDING_LADDER_TIER3_PLAN.md).
Original ladder v1 artifacts and scientific recipe remain unchanged.

SPEC binds original gate/preflight hashes and file references, active source
snapshot, arm, measured time, exact tier3 execution site, CPUs, memory and
`sporc_debug_to_tier3_same_a100_environment_resources_v1` policy.
Only the partition differs from the accepted debug resource profile.

PLAN binds the complete tier3 command and explicit
`AUTHORIZE_LUKA_EMBEDDING_LADDER_TIER3_V1` authorization. One job runs an entire
five-rung arm; each arm requires its own review. Submission uses durable claims,
exact-DAG journals and canonical dry-run ledgers. No automatic retry/resume.

SCIENCE and RESULTS bind the transfer spec. Per-rung STEP/TRAINING_REPORT/LOGITS
remain original ladder-v1 artifacts bound to the original gate and target
identities. Final selected weights and compact results are retained. Owned
intermediate checkpoint retirement is unchanged.

Both creation and worker startup replay the original source's gate acceptance.
Actual tier3 allocation must have the same exact measured GPU, environment,
CPUs and memory. Results authenticate compact outputs and their original gate/
report references without needing retired intermediate model files.
