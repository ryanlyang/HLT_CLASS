# CMS direct Strategy B comparison on SPORC/debug

## Authority and scope (2026-09-19)

User-authorized standalone comparison alongside the running coarse campaign.
Do not cancel, modify, requeue or wait for that campaign. Reuse the original
accepted dense campaign's common references and preparation read-only, exactly
as the coarse campaign does. This plan extends the existing native CMS plans;
it does not change their scientific or execution identities.

## Question and registered experiment

Does a single Strategy B acquisition/withdrawal transition from persistent
U000 to genuine HLT D000 retain more performance than ordinary direct logit KD?

The existing `DIRECT_D000` is a single-view model trained with U000 logit KD.
It is a comparator, not the new fusion fit. The new experiment is:

1. Import M0HLT CE, pure-OFFLINE CE, persistent U000 CE, the U000 probability
   bank and DIRECT_D000, with their authenticated reports and checkpoints.
2. Cold-start `ACQUIRE_D000`: primary D000, privileged context U000, ordinary
   Strategy B learned residual cross-attention fusion. CE/KD = 25/75, T=2,
   teacher = the imported U000 bank. Primary initialization and sampling use
   the same D000 seed alias as DIRECT_D000. Context has its registered seed.
3. Reduce the selected acquisition model into its frozen probability bank.
4. `WITHDRAW_D000` starts from that selected acquisition checkpoint with a
   fresh optimizer. Use the existing withdrawal loss: zero-route CE/KD
   0.25/0.30, privileged-route CE/KD 0.15/0.20, directed logit consistency
   0.05 and representation consistency 0.05. Teacher = frozen acquisition.
   Context alpha holds at one through pass 10, falls by cosine to zero at
   pass 60, then stays zero. Select checkpoints on the alpha-zero route.
5. Extract `CARRIER_D000`: remove all context parameters, require byte-exact
   prediction parity with alpha-zero fusion. Deployment is one HLT-only ParT,
   not a two-branch D000/D000 ensemble.
6. Aggregate the four imported references/control and the extracted carrier.

Both new fits retain the registered 100-pass maximum, minimum 60, patience 15,
warmup 1–3, LR 3e-4 through 45, cosine to 1.5e-5 at 60, floor thereafter,
batch 256, BF16 forward/FP32 losses, and restore-best selection. No scientific
kernel or model architecture is changed. This is two fresh optimization
phases versus one in direct KD: it is not a compute-matched comparison.

## Data and evaluation

Original CMS/Scouting only, not JetClass2. Reuse exactly the authenticated
500k training / 250k validation / 250k sealed final-test commitment, linear
salience full-cardinality matching, persistent-HLT support and train-only
scales. U000 can include unmatched HLT particles; OFFLINE is the separate
pure-offline reference. Training/selection/report partitions are unchanged.
Report accuracy, AUC, geometric macro R50 and recovery with M0HLT=0% and
OFFLINE=100%, using the unchanged validation REPORT subset. Also print
carrier-minus-direct-KD deltas and the two-phase compute caveat. No ordinary
task reads final-test inputs or evaluates them. Weak results never abort.

## Isolation, evidence and submission

Use a fresh source-pinned root, CAMPAIGN_SPEC/v6, GRAPH/v3, `ladder=direct_fusion`,
and `cmsdf_` job names. Eleven science tasks: five CPU imports, two fresh GPU
fits, one GPU acquisition reducer, one GPU extraction, aggregate and complete.
No U050/U100/intermediate-rung training is included. Existing v1–v5 campaign
specifications and dense/coarse graph hashes remain unchanged.

The original dense-v3 source must have valid accepted SPORC/debug evidence.
Reuse its original preparation and measured longest-U000/U000 batch-256
envelope (job 21720795 in the current execution), under the existing strict
runtime/code, data, resource, partition and 85% CPU / 90% GPU checks. The
direct D000/U000 pair is bounded by that same measured pair. This explicitly
extends the already authorized accepted-evidence reuse to the direct study;
no new GPU gate is claimed. Unchanged installed conda software remains an
assumption. Any incompatible source/kernel/resources fails closed.

New PREPARATION_IMPORT/v3, SHARED_SOURCE/v2, ACCEPTANCE_IMPORT/v2 and
ACCEPTANCE_REUSE/v2 bind this direct consumer. Publish CPU foundation and
acceptance-import receipts before science submission. Completed source jobs
are reused through authenticated outputs, not stale Slurm dependency IDs.
Pending source imports, if any, retain exact-ID dependencies. The coarse
campaign is never a dependency or a cancellation target.

`create-direct-fusion` creates and dry-materializes only. The helper has
separate `prepare` (create/import/audit, no sbatch) and `submit` modes.
Live science needs `AUTHORIZE CMS DIRECT FUSION 500K EXACT SPEC`, a clean
pushed checkout, complete dry plan and accepted imported gate. Its CLI must
reject `retire-dense` for this campaign even if a retirement phrase is supplied.

## Verification and handoff

Test old graph identities, exact direct DAG/teachers/seeds, debug-only plans,
read-only preparation/reference reuse, CPU-only evidence import, strict
schema/source rejection, no cancellation access, exact dependency handling,
and a tiny production-path acquisition/reduction/withdrawal/extraction chain.
Fake-Weaver local tests are plumbing tests, not new GPU acceptance. Confirm
unchanged real-Git preparation/runtime fingerprints on Python 3.10 and 3.13.
Give scoped commit/push and fresh-root queue commands; do not claim that local
tests submitted or measured anything on SPORC.
