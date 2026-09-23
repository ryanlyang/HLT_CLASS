# JetClass2 dz-fix fusion chain contracts

Authority: `plans/JETCLASS2_DZFIX_FUSION_CHAIN_500K_PLAN.md`.

Artifacts use `JETCLASS2_DELPHES_DZFIX_FUSION_CHAIN_*` and canonical content
hashes. LAUNCH_SPEC and CAMPAIGN_SPEC are **v6**, SOURCE_IMPORT stays **v3**,
and ACCEPTANCE is **v4**; EARLY_PARITY and other artifact kinds are v1. Old execution roots
are not silently upgraded, including the capacity-correct v3 run that OOMed.

LAUNCH_SPEC/v6 binds the explicit producer SALIENCE_SCREEN_SPEC/v2, its exact
eight-task live ledger and `complete` job, new source checkout, independent
launch/campaign roots, tier3 resources, graph and population. Parent commands
and dependency closure must equal the source screen plan after exact-ID
resolution. Dry, incomplete, foreign, or duplicated-job ledgers fail closed.
SOURCE_IMPORT/v3 additionally binds screen/selection/completion/profile and
all eight task receipts, selected foundation spec/lock, and current semantic
preparation file hashes. Foundation location and hash must match the screen's
candidate registry as well as selection. No continuation/production preview is
required. All compact payloads are authenticated before reuse.

Registration locks `inventory_max_selected_round_up_16_no_truncation_v1`:
capacity is `max(16, ceil(max_selected_particle_count / 16) * 16)`, using
authenticated inventory metadata exactly as both foundation builders do.
All four screen foundations must match the canonical 17-feature/11-class
input contract at that capacity, not merely an arbitrary larger allocation.
The current dzfix foundations are capacity 320. No particle truncation,
foundation rewriting, raw final-test access, or GPU-gate bypass is permitted.
The embedded foundation hash binds capacity through cache sizing and real
longest-batch stress; CPU/GPU headroom, batch size and CPU/GPU resources are unchanged.

The producer's debug-screen/tier3-production separation remains intact. The
consumer does not inherit that production resource profile: it registers its
own tier3 execution and fresh acceptance. The current completion boundary is
21748725, read from the live ledger, not hardcoded in the reusable Python code.

CAMPAIGN_SPEC embeds that import, immutable graph/recipe/resource registration,
validation firewall and 25-task graph (4 gates including storage + 21 science).
TASK_REPORT binds campaign, source, task, dependency attestations and every output
byte hash. TRAINING_REPORT wraps the unmodified shared training-kernel report
and adds report-subset metrics, teacher lineage, selected checkpoint and native
input semantics. Banks retain the existing exact-row probability-bank contract
(train T2, validation T1), never unlabelled/raw-logit mixtures.

VALIDATION_PARTITION binds ordered identities/labels and deterministic stratified
50/25/25 assignments. ACCEPTANCE binds the exact campaign/source/runtime and
actual single/paired execution, bank/weight round trips, memory and time evidence.
No acceptance-only weights count as scientific fits. All ordinary artifacts
record `final_test_accessed=false`; ordinary tasks have no test capability.

CAMPAIGN_SPEC/v6 retains `fusion.saved_tensor_storage` as
`pair_saved_tensors_cpu_v2`: full native pair populations at context, primary
and cross sites; lossless pinned CPU storage only for CUDA training autograd
saves; no recomputation or scientific changes. The existing batch 256,
320-capacity foundation, BF16 training and compact-mask transformation remain.
CPU tensors are transient, never persisted. Offload is bypassed for inference.
Counter data hold no tensor references and do not enter model state dictionaries.
Copies preserve dtype, shape and strides, including gaps/overlap, with storage
offset rebased to zero. The physical storage span is copied synchronously to
pinned CPU memory, then restored when backward needs it.

ACCEPTANCE/v4 requires installed-Weaver CUDA FP32/BF16 storage on/off parity
over three optimizer updates, including both feature gradients, parameter gradients, BN running state
and AdamW state. Each paired three-step longest-population batch-256 stress
must record actual saved AND restored CUDA tensors at all three pair sites.
The science gate rejects absent, CPU-only, no-op, wrong-policy or incomplete
parity/stress evidence; old v1/v2/v3 gates cannot authorize this execution.
Native-mask parity, unchanged measured memory thresholds and registered runtime limits
also remain mandatory. Scientific fit reports record the storage policy and
transfer counters; their sum is not an estimate of unique/live memory savings.

`fusion.parity_backend` locks deterministic comparison settings, including
TF32 disabled and cuDNN benchmarking disabled. cuBLAS workspace configuration
is set before Python only in preflight workers. All PyTorch flags, including
the full float32 matmul precision policy, are restored on success or exception
before resource stress and training. Tolerances are unchanged:
FP32 (rtol=2e-5, atol=2e-6), BF16 (rtol=.01, atol=5e-4).

Four EARLY_PARITY/v1 reports bind the campaign, the same four unique registered
train identities/file indices, acceptance-only scope, compact-mask parity and
three-step storage comparison for U050/U000 and D000/D000 in both precisions.
They are written before the full-population caches; a failed comparison cannot
publish a passing report. ACCEPTANCE/v4 embeds and validates their hashes,
precision/coordinate coverage, provenance, storage policy and backend policy.
Early reports alone never authorize scientific work or replace full-data stress.

The v6 execution registration is `sporc_a100` (tier3/qos_tier3), with 4320-minute
fit requests. Other task limits and CPU/RAM/GPU requests remain unchanged.
`runtime_projection_margin=1.30` and `runtime_shutdown_reserve_seconds=3600`
are immutable. `fit_runtime_budget` in ACCEPTANCE/v4 binds partition, fit
request, reserve, derived projection ceiling and margin to the campaign spec.
Both preflight publication and science release require a positive finite
maximum-budget projection <= (fit request - reserve), currently 71 hours.
Measured runtime already includes 30% margin plus cache preparation; early
stopping is not assumed. The worker checks the actual scheduler TimeLimit and
partition against the submitted task registration, including launchers.
No pending-job debug portability, gate bypass, rolling resume, reduced batch,
or reuse of failed debug acceptance is introduced. The source screen remains
the authenticated v2 debug producer and its matching outputs remain read-only.

Publication is immutable/atomic; existing roots are not overwritten. Exact DAG
submission uses canonical dry runs, exclusive submission claims, journalled IDs
and afterok only. An existing live ledger is verified and returned, not resubmitted.
Each task has one immutable output directory; a partial failed task is not
silently overwritten or resumed. Recovery needs operator review and a new
registered execution, not an automatic retry. No cancellation of other
campaigns is authorized by these contracts.
