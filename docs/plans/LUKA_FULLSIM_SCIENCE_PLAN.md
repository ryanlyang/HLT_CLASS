# Luka FullSim stage 3: SPORC fusion campaign

User authorized implementation after successful technical job 21835630 on
2026-10-09. This plan supersedes the historical dataset, schedule, validation
partitioning and resource defaults in `MULTIVIEW_FUSION_LADDER_OVERVIEW.md` for
this campaign only. Stage-1 selection and stage-2 physical inputs remain fixed.

## Frozen comparison

Use the existing 100,000 TRAIN / 50,000 validation rows, natural eleven-class
proportions, offline pT strictly above 200 GeV, and QCD from QCD files only.
Reuse prepared assignments and every input fingerprint. Provisional mm on both
sides, error-only unavailability at zero uncertainty, signed finite tails and
nontruncating full cardinality are unchanged. Event independence across source
files remains unverified. No proxy dataset or previous scientific fit is reused.

Train twelve fresh models:

- M0HLT (D000 CE), OFFLINE (pure offline CE), U000 (CE anchor).
- DIRECT_D000: single-HLT KD from U000.
- FUSION_U050: primary U050, context U000, teacher U000.
- FUSION_U100: primary U100, context U050, teacher FUSION_U050.
- FUSION_D066: primary D066, context U100, teacher FUSION_U100.
- FUSION_D033: primary D033, context D066, teacher FUSION_D066.
- FUSION_D000: primary D000, context D033, teacher FUSION_D033.
- FINAL_DIRECT_D000: single HLT, teacher FUSION_D000.
- FUSION_D000_D000: two HLT encoders, teacher FUSION_D000.
- FINAL_BRIDGE_D000: single HLT, teacher FUSION_D000_D000.

Seven TRAIN-only probability reducers and aggregate/completion make **21 jobs**.
There is no additional non-fusion coarse or dense branch in this registration.
Fusion direction is context-to-primary at blocks 2/4/6/8 with alpha=1. Every fit
is cold-start; only teacher probabilities cross edges. The two final single-HLT
models and DIRECT are comparable deployable endpoints. Intermediate fusion with
offline context is an oracle; HLT/HLT fusion has greater inference capacity.

Use the unchanged `jetclass2_delphes.campaign.recipe()` and `runner.train_kernel`:
batch 256, BF16 forward/FP32 loss, AdamW, 60--100 passes, patience 15 counted from
the first validation, 3-pass warmup, hold through 45, decay through 60, then floor.
KD uses 25% CE + 75% forward KL at T=2 with one T-squared factor. All 50k validation
jets select checkpoints and supply exploratory reporting, as in current CONTEXT
experiments; do not import the older overview's three-way validation split or
pass-60-start patience. Coordinate-paired initialization/sampler seeds follow
the current CMS-proxy ladder donor, including M0HLT versus all single D000s.
Do not tune, skip or cancel a row based on its metrics. Final test stays sealed.

## Admission and execution

Bind exact prepared-v1, preflight-v2 report hash
`ffaa43d6d83b4587a9e8a6fca0995ad30238ccc7936898022642e6856b53274c`
and preflight source `bcccd498ee66fdee6fd3731b823d09ac2bb5df8b`.
Authenticate both historical source worktrees, all report output bytes, installed
Weaver/environment, parity/stress evidence and new clean pushed source. Existing
tracked `src/` files must be byte-identical; only the five named stage-3 modules
may be added. Old preparation and reports are never rewritten or relabelled.
Probe checkpoints/banks are evidence only, never loaded as scientific teachers.

Reuse the measured SPORC debug A100 envelope: 6 CPUs, 256000 MiB, one 40GB A100,
1386 minutes per training job, 131 per reducer. These are conservative requests,
not ETAs. CPU summaries request 1 CPU / 16 GiB / 60 minutes. No cross-cluster or
partition transfer is authorized. Retain CUDA workspace and isolated conda setup.

Each GPU worker prepares at most its two required views in one ROOT pass per
role using unchanged physical/view/feature functions, then verifies every
prepared fingerprint. D000/D000 shares the same cache. Pure D000 never decodes
offline particles. Generalized cache construction is compared against the v1
cache for every coordinate locally; maximum support and reserved memory must
fit the accepted U000-dominated stress/cache envelope before running. No old
source verifier is weakened. Phase heartbeats expose cache versus training time.

New orchestration receives synthetic end-to-end/failure-injection tests. The
already completed genuine SPORC full-population preflight is the GPU evidence
for the unchanged kernels, not proof that the new campaign has already run.
No additional technical fit is automatically queued. Revalidate all evidence
before exact-plan dry review and live submission; require completed Slurm
accounting for the accepted preflight when creating the campaign.

One new root; immutable source/preflight-bound spec, full DAG dry ledger,
explicit plan-hash authorization, exclusive live claim and per-job journal.
An interrupted/ambiguous live submission stops for inspection, never silently
duplicates jobs. Failed task directories remain intact; no automatic retry or
rolling resume. Publish receipts completion-last with hashes of all outputs and
dependency receipts. Results may inspect completed rows before the graph ends.

Recovery uses M0HLT=0%, pure OFFLINE=100% for accuracy/AUC and available rejection
metrics; zero background passing is censored, not infinite measured rejection.
Report single-seed, reused-validation and unequal-capacity/compute caveats.
