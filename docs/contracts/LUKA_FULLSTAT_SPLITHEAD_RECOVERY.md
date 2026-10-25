# FullStat split-head post-fit recovery v1

User-authorized recovery, 2026-10-25. Extends the
[split-head scientific plan](../plans/LUKA_FULLSTAT_SPLITHEAD_PLAN.md) and
[v1 scientific contracts](LUKA_FULLSTAT_SPLITHEAD.md). No scientific objective,
selection, seed, loss coefficient, view, population or training budget changes.

## Failure and correction

Original execution a7efcef6a8399cdcde80b2941e1a21910ba1bf67, DE066 job 7910169,
completed twenty epochs and published FIT receipt, native selected.pt, report
and validation predictions. Post-fit revalidation failed because JSON key sorting
changed floating-point accumulation order. Exact equality now recomputes the total
in registered teacher order, as the training writer did. No tolerance is widened,
no saved report is rewritten, and the differentiable training objective is unchanged.
DE033/DE000 use the same correction. Old pinned source remains untouched.

## Explicit recovery, not a successful-old-job fiction

New artifacts are `LUKA_FULLSTAT_SPLITHEAD_RECOVERY_<kind>/v1`: SPEC, EVIDENCE,
PLAN, CLAIM, READBACK and TASK. All carry normal FullStat sealed-role flags and
canonical hashes. Old v1 scientific receipts/reports retain original identity.

Before creating/reviewing/submitting/working, authenticate the failed campaign
in its original clean pushed source, completed UE000/UE100 ancestry and exact
original acceptance. Require exact old job ledger, job 7910169 FAILED/1:0, and
DE033/DE000 CANCELLED/0:0 with zero elapsed time and no attempt or fit directories.
Active/unknown/started descendants fail closed; never cancel them automatically.
Require the known dictionary-order discrepancy, otherwise stop for new diagnosis.

Verify the original FIT receipt and every payload hash, saved validation identities,
labels, recomputed metrics, selected epoch, all twenty schedule/loss rows, startup
record, original TRAIN bundle, imported manifest/payload refs and M0HLT probability
export. Bind these exact bytes in recovery evidence. No new scientific choices.

One new three-job afterok graph:

1. `recover_DE066`: no optimizer, startup updates or training calls. Restore the
   saved native checkpoint, require byte-identical predictions on the same ordinary
   validation role, validate original TRAIN target joins, export selected native
   TRAIN embeddings and T2 probabilities into the new recovery root, then publish
   a RECOVERY_TASK linking the old FIT and new recovery job. No test access.
2. `train_DE033`: fresh original twenty-epoch recipe, using recovered DE066 targets.
3. `train_DE000`: fresh original twenty-epoch recipe, using new DE033 targets.

Descendants require successful exact recovery-job accounting, never successful
accounting for the failed original. Historical DE066 fit files are referenced in
place. The completed M0HLT probability bank is reused byte-for-byte. Native
checkpoint re-evaluation is a technical readback, not checkpoint reselection.

## Execution safety

No standalone preflight or login controller. Inherit original Oscar L40S/6 CPU/
180000 MiB resource envelope and walltime ceilings, including for export-only
recovery; the recovery's duration is not remotely measured. Require 16 GiB personal
durable headroom. New roots cannot overlap old campaign/data/source roots.

Exact pushed recovery source: integration baseline 234b029ec14a8d4861134e098cafbf939d325cd8;
only training/worker/campaign/submission/kd_banks changes and the five explicitly
listed recovery runtime additions are permitted. Existing scientific recipes,
model and training-forward code remain unchanged. Original-source replay is not
run with a patched old worktree.

Dry review uses sbatch --test-only. Live execution requires the exact saved plan
hash and AUTHORIZE_FULLSTAT_SPLITHEAD_RECOVERY_V1. An exclusive recovery_claim_v1
directory in the old campaign binds the new root and prevents duplicate recoveries
across roots. This is the only new entry in the old campaign; no old files change.
New live claim/journal likewise reject ambiguous resubmission. Preserve failed
claims/partial exports; no automatic retry, deletion or overwriting. Never weaken
checks to turn corrupt science into a reusable fit.

The interpretation remains exploratory single-seed development validation. Finite
poor performance must never fail recovery or block later registered science.
