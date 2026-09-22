# JetClass2 dz-fix fusion chain contracts

Authority: `plans/JETCLASS2_DZFIX_FUSION_CHAIN_500K_PLAN.md`.

Artifacts use `JETCLASS2_DELPHES_DZFIX_FUSION_CHAIN_*` and canonical content
hashes. LAUNCH_SPEC and CAMPAIGN_SPEC are **v4**, SOURCE_IMPORT stays **v3**,
and ACCEPTANCE is **v2**; other artifact kinds retain v1. Old execution roots
are not silently upgraded, including the capacity-correct v3 run that OOMed.

LAUNCH_SPEC/v4 binds the explicit producer SALIENCE_SCREEN_SPEC/v2, its exact
eight-task live ledger and `complete` job, new source checkout, independent
launch/campaign roots, debug resources, graph and population. Parent commands
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
longest-batch stress; CPU/GPU headroom, batch size and resources are unchanged.

The producer's debug-screen/tier3-production separation remains intact. The
consumer does not inherit that production resource profile: it registers its
own debug execution and fresh acceptance. The current completion boundary is
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

CAMPAIGN_SPEC/v4 additionally freezes `fusion.saved_tensor_storage` as
`pair_saved_tensors_cpu_v1`: full native pair populations at context, primary
and cross sites; lossless pinned CPU storage only for CUDA training autograd
saves; no recomputation or scientific changes. The existing batch 256,
320-capacity foundation, BF16 training and compact-mask transformation remain.
CPU tensors are transient, never persisted. Offload is bypassed for inference.
Counter data hold no tensor references and do not enter model state dictionaries.

ACCEPTANCE/v2 requires installed-Weaver CUDA FP32/BF16 storage on/off parity
over three optimizer updates, including parameter gradients, BN running state
and AdamW state. Each paired three-step longest-population batch-256 stress
must record actual saved AND restored CUDA tensors at all three pair sites.
The science gate rejects absent, CPU-only, no-op, wrong-policy or incomplete
parity/stress evidence; the old v1 gate cannot authorize this execution.
Native-mask parity, unchanged measured resource thresholds and runtime limits
also remain mandatory. Scientific fit reports record the storage policy and
transfer counters; their sum is not an estimate of unique/live memory savings.

Publication is immutable/atomic; existing roots are not overwritten. Exact DAG
submission uses canonical dry runs, exclusive submission claims, journalled IDs
and afterok only. An existing live ledger is verified and returned, not resubmitted.
Each task has one immutable output directory; a partial failed task is not
silently overwritten or resumed. Recovery needs operator review and a new
registered execution, not an automatic retry. No cancellation of other
campaigns is authorized by these contracts.
