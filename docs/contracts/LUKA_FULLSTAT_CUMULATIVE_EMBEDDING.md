# FullStat cumulative embedding contracts v1

Authority: [active scientific plan](../plans/LUKA_FULLSTAT_CUMULATIVE_EMBEDDING_PLAN.md).
This new namespace does not modify or relabel the
[four-teacher contract](LUKA_FULLSTAT_MULTITEACHER.md),
[U100 contract](LUKA_FULLSTAT_U100_EMBEDDING.md) or
[FullStat20 contract](LUKA_FULLSTAT_OSCAR20.md).

All records are canonical content-hashed `LUKA_FULLSTAT_CUMULATIVE_EMBEDDING_KIND/v1`,
schema1, explicit parent hashes, original FullStat sealed-test flags. Source,
specification, recipe and exact file SHA256 identities are mandatory. Paths,
mtime, observed accuracy and job names alone are not authentication.

## Scientific identities

| Stage | Registered input | Frozen native teachers | Weights |
| --- | --- | --- | --- |
| UE100 | U100 | original OFFLINE | 5 |
| DE066 | D066 | OFFLINE, new UE100 | 2.5 each |
| DE033 | D033 | OFFLINE, new UE100, new DE066 | 5/3 each |
| DE000 | D000 | OFFLINE, new UE100, new DE066, new DE033 | 1.25 each |

`RECIPE` binds stage, teacher order, separate head initialization seeds, 20 epochs,
batch128, CE1, KD0, total auxiliary5, optimizer/schedule and removable heads.
`TARGET_RECIPE` binds teacher/coordinate, native 128-D pre-classifier tap,
FP32/eval TRAIN export, FP64 population moments, scale floor1e-3. Auxiliary
predictions never become targets or classifier inputs. No old logit U100/D066/
D033 checkpoint may replace a newly trained ancestor. All fits start fresh.

`SPEC` binds the clean pushed source snapshot, `REUSE`, exactly four nodes and
tasks, 3M TRAIN / 500k validation, protected output root, quota declaration and
resource policy. Final-test1.5M and unused105325 remain sealed. Intermediate
inputs are oracle diagnostics; only DE000 is deployable native HLT.

## Bounded acceptance reuse, not a new preflight

`REUSE` is obtained by subprocess replay in the completed b21c2d4e four-teacher
worktree. Its original source authenticates setup, selected fit, banks and
upstream U100/OFFLINE/FullStat chain. The importer additionally authenticates
the exact successful four-teacher training Slurm ID, owner and account. All
pre-existing donor Python/shell runtime bytes must match; new imports must be
the additive cumulative package in a clean pushed execution worktree.

The original native seven-coordinate preflight and completed four-head setup
must have the same L40S/environment as this registration. Reuse their evidence
without claiming that the new multi-head/richer-view combination was timed.
Each request is max(original accepted train minutes, four-teacher train minutes)
plus120 minutes; values above48h fail, never silently clamp. CPU6, memory180000MiB,
batch128, original no-truncation particle policy and exact backend are unchanged.
Reserve12GiB personal durable capacity. Two cache copies plus conservative
four-target/export workspaces and16GiB reserve must fit85% host memory.

`STARTUP` binds spec/bundle/stage, actual GPU/environment, source commit, native
tap equality, three disposable max-TRAIN-support updates, a max-validation-
support forward, finite head/backbone gradients and host/GPU peaks. No benchmark,
technical epoch or full-validation pass. The scientific model is constructed
afresh afterward, resetting its original coordinate-paired seeds. Resource
failure stops the job; it does not change its scientific definition.

## Banks, training and completion

OFFLINE bank bytes retain their original sweep contract. The adapter explicitly
allows the four registered TRAIN consumers by exact identities/labels/prepared
hash; it never changes a cache's coordinate to bypass the original loader.

Each new `BANK` binds the selected native scientific checkpoint/report, prepared
identity and target recipe. `targets.npz` contains unique ordered uint8 Nx32
identities, int64 labels, FP32 Nx128 targets, FP64 128-D means/scales. All are
finite; scales obey the floor. No validation/test target bank is exported.
The export verifies that loaded native weights equal the recorded selected
checkpoint, not an auxiliary head or last-epoch state.

`BUNDLE` binds the stage, exact teacher order, each bank file/hash and every
new ancestor's completed `TASK` hash. Targets stay separate; there is no
averaging of unaligned embedding axes. The dataset's paired row order must
match exactly across teacher and student views.

`TRAINING_REPORT` contains all20 validation epochs, full population and update
counts, original learning rates, CE/per-head/weighted losses and wall times.
The existing AUC-first selector and tie-breaks select one native checkpoint.
The loss is CE + sum((5/N) * mean SmoothL1(head, target)); targets are detached.
Low AUC, low recovery, poor auxiliary fit or censored rejection are valid results.

`FIT` publishes native selected weights, report and full-validation predictions
after native checkpoint/probability roundtrip. No predictor weights in selected
artifacts. `TASK` is published last, only after this fit, startup validation and
(for the first three rungs) outgoing bank export. It binds every ancestor, exact
accepted Slurm ID, source commit, report/bundle and the complete output list.
Downstream jobs recursively authenticate these receipts and require exact
ancestor `COMPLETED/0:0` accounting. A fit receipt alone cannot release a rung.

## Submission and reporting

`PLAN` contains exactly four scientific jobs. Every rung depends afterok on all
prior rungs. No setup/preflight/export/aggregate job or detached controller.
All jobs use no-requeue, kill-on-invalid-dep, export=NONE and absolute-project
environment helpers. A saved exact dry review, hash, explicit authorization and
exclusive live claim are required. Uncertain submission never triggers retries.

`RESULTS` authenticates completed tasks and saved reference probabilities only;
no new inference or test access. It reports pending stages, all20 history in the
underlying reports, selected validation AUC/accuracy and linear recovery against
historical M0HLT/OFFLINE. R30 and R50 use the same AUC-selected checkpoints:
score p(signal)/(p(signal)+p(QCD)), zero denominator score0, descending signal
ceil(efficiency * count) rank with ties included. Zero QCD passing is null
(censored), not infinity. Per-class recovery is computed separately at each
efficiency; unresolved/zero reference gaps are null. The comparisons remain
single-seed, adaptively inspected development evidence, not independent test
claims or compute-matched comparisons.
