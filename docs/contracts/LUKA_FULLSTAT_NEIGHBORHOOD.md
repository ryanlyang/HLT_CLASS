# FullStat smooth-neighborhood contracts v1

Authority: [active plan](../plans/LUKA_FULLSTAT_NEIGHBORHOOD_PLAN.md).
Namespace `LUKA_FULLSTAT_NEIGHBORHOOD_<kind>/v1`, schema1. Historical contracts
remain unchanged. Every artifact is canonical-content-hashed with parent bindings
and FullStat sealed-role flags; publication is immutable. No existence-only reuse.

- REUSE: replay completed FINALIZE/FINALIZE_RETRY DE000 in its original source,
  verify native models, receipts, scheduler accounting, original preflight and
  compact TRAIN banks. Borrow exactly OFFLINE/UE100/DE066/DE033. Native donor
  bytes must match the original execution. References are M0HLT, OFFLINE and
  OLD_SPLITHEAD_DE000; M0HLT is never an auxiliary target.
- TARGET_RECIPE: physical smooth Gaussian .02/.05/.10, six categories, 36 channels,
  FP64 reductions and canonical offline order; log1p(count),
  log1p(100*offline-weighted-pT/total-HLT-pT), equal-jet TRAIN normalization floor0.1.
  No matches, cutoffs, truncation, clipping or label-conditioned targets.
- RECIPE: native D000; 20 epochs/batch128; paired seeds and existing schedule;
  CE.5 on deployed head, total K1 on four independent T2 logit heads,
  total E40 on four independent native-representation predictors; smooth10
  on isolated-seed94271 block8 head. No control, teacher fits or relational term.
- SPEC: exact clean pushed source, immutable REUSE, two distinct named output
  roots, separate personal result/target quota declarations, authenticated
  actual TRAIN support and conservative host-memory budgets. Result reserve8GiB;
  target requirement `144*tokens + 48*rows + 8 + 2GiB`. No FP16 or deletion fallback.
- BANK: TRAIN only, D000 order, FP32 `[total_HLT_tokens,36]` `smooth.npy` plus
  `index.npz`: uint8[N,32] identities, int64 labels and offsets, FP64 mean/scale36.
  Single family only. Hash both files; exact joined IDs/labels/support are checked
  again against the authenticated native cache. Load one target RAM copy for
  vectorized gathers; normalize FP64 then FP32 on batch request, padding zero.
- TARGET_TASK: registered CPU job, source/spec and bank hashes. Role-safe ROOT
  spans exclude final-test/unused entries. Ordinary validation rows can be
  decoded by the inherited reader but never create targets or enter moments.
  Each TRAIN chunk replays the original HLT features/vectors/order exactly.
  Six spawned workers maximum and six pending chunks, ordered publication.
- BUNDLE: four original embedding and logit bank refs plus neighborhood manifest,
  recipe/spec/prepared hashes, exact teacher order. Original bank validators retain
  original contracts, FP64 TRAIN moments floor1e-3, T2 native probabilities.
- STARTUP: inside the authorized fit, not a standalone preflight: native FP32/BF16
  equality, CE-only main-head gradient, connected finite heads/backbone,
  physical128 maximum-support train/validation, three stress updates and bounded
  32-update/16-validation-batch timing. Include preparation elapsed and margin1.5,
  120min allowance in the20-epoch budget. Not a measured full epoch or guarantee.
  Discard technical model and reset scientific RNG.
- TRAINING_REPORT/FIT: all20 epochs and exact loss weights/schedule, AUC checkpoint
  selection, native-only selected state, strict native reload and selected
  validation-probability/metric equality. Save completed_native.pt and
  completed_training.json before publication checks. No automatic retraining.
- TASK: successful target receipt plus complete fit, integrated checks and exact
  submitted job/source. Descendants and results authenticate saved outputs.
  Poor finite AUC/rejection never rejects technical completion.
- RESULTS: historical comparison on same500k rows, same AUC-selected checkpoints
  for R30/R50, per-class recovery and censored zero-passing QCD. Old DE000 has
  different K/E; not a matched neighborhood-only ablation. No inference/test access.
- PLAN/SUBMISSION_CLAIM: exactly CPU `targets` and L40S `train_SMOOTH_W10`, the
  latter `afterok` the first. Six CPUs, CPU90000MiB/GPU180000MiB, 24h each,
  no-requeue/kill-on-invalid-dep. No automatic retries or cancellation. Exact dry
  hash and explicit authorization precede live claim and per-command journal.
- LAUNCH_INTENT/APPROVAL/COMPLETE/FAILURE: detached bounded two-job authorization,
  save full dry plan before submission. Exclusive launch/process claims; no
  guessing fresh roots on failure. Foreground tail is expendable; launcher uses
  nohup/setsid with stdin closed. Does not protect against node reboot/site cleanup.

Compatibility baseline and internal donor: `62f9773aabd150004be9c319d6ed06c412fcc57a`.
Only new namespace/wrappers may alter runtime relative to it. Exact parent source
checks are additional. New scratch bank is reproducible but keep it for this run's
authenticated completion/reuse; never delete borrowed banks or parent worktrees.
