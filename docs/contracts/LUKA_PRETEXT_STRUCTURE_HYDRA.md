# OFFLINE pretext and HLT structure campaign contract v1

Authority: [scientific implementation plan](../plans/LUKA_OFFLINE_PRETEXT_STRUCTURE_HYDRA_PLAN.md).
Namespace: `hlt_classification.luka_pretext_structure_hydra`. Donor baseline:
`a6da7a02b773d2397ff0d301bb4b5b400fc7528c`. Existing campaigns are unchanged.

New artifacts use `LUKA_PRETEXT_STRUCTURE_HYDRA_<KIND>/v1`, schema1, with
content hashes, exact parents and the existing exploratory-population flags.
Kinds are RECIPE, SPEC, TRANSFER, INVENTORY, NORMALIZATION, STRUCTURE_BANK,
STATISTICS, PREPARATION, TEACHER_EPOCH, TEACHER_REPORT, TEACHER_FIT,
EMBEDDING_BANK, EPOCH, TRAINING_REPORT, DEPLOYMENT, FIT, STARTUP, REPLAY,
RESOURCES, OUTPUT_INVENTORY, ACCEPTANCE, PLAN, SUBMIT_CLAIM, RUN_CLAIM, TASK, RESULTS and PROGRESS.
EPOCH/FIT refer to students; TEACHER_EPOCH/TEACHER_FIT have distinct selection
semantics. Exact submission journals retain HCWDL v2.

## Registered science

Exactly8 new OFFLINE teachers,12 embedding students,8 structure students and
2 fresh controls. All22 HLT fits use the original native128 model, full30,
batch128 including partial batches, optimizer, learning-rate schedule and
U100 embedding weight5. Additional embedding weight5 is divided equally among
registered targets and ramped over3 epochs. Structure weights are1 except
S06=0.3, with the same ramp. S08 uses HLT structure; other structure rows use
OFFLINE structure. All prediction heads are removed from native deployment.
The fresh C_CE control predicts the already completed teacher-sweep T00 bank.

Teacher losses never use jet-class labels, logits or class-based corruption.
Every decoder consumes only the pooled128 embedding and fixed queries.
Every teacher includes0.25 times the clean structural-signature objective.
Selection minimizes full validation pretext loss over two fixed corruptions,
earliest exact tie. Students retain the original native validation selection.
Poor finite loss, accuracy or embedding rank never fails or skips a fit.

## Data and target capabilities

Use the original299234 TRAIN/74895 validation identities and native features.
Historical test has already been repurposed for training; original50k overlaps
validation. There is no independent test and no final-test operation.
Historical teacher-sweep artifacts are authenticated in their pinned worktree.
Existing physical TRAIN shards are reused through the original Group A parent.
A new minimal physical validation export binds the same prepared population,
source inventory and explicit length conventions. No fabricated particle match.

Physical arrays contain p4, charge, six measured categories, tracking in mm and
four applicability bits. Role-specific memmaps preserve every real particle.
Fixed decoder K is the maximum OFFLINE count from TRAIN/validation inventory;
individual target counts never enter the encoder or decoder. Native clean
features must exactly replay the old OFFLINE cache in setup.

Corrupt physical particles first, then rebuild the visible native features.
Masks are keyed by identity/task/epoch/draw; no labels or source indices enter
the model. Canonical physical sorting makes masks permutation-stable. N<2 has
zero primary loss. If a removal would leave no positive-pT particle, retain
the highest-pT particle and exchange a visible zero-pT particle. This explicit
rare-case rule keeps the visible frame defined without dropping the jet.

Scalar targets and mixed-set continuous fields use TRAIN FP64 population
moments, std floor0.001 and frozen constant-channel masks. Two fixed TRAIN
corruption draws determine pretext moments. New epoch masks never refit them.
Validation uses those same moments. Native U100 normalization is unchanged.
Each selected teacher exports a strictly reloaded native clean OFFLINE TRAIN
128D bank with its own mean/std and `PRETEXT_Txx` coordinate. Technical setup
banks are explicitly marked and cannot be accepted as scientific targets.

## Numerical definitions

The plan pins all bins, weights and transformations. Correlations use exact
FP64 pair sums and `trace(A^3)/6`; cumulative triples require all three edges
within the radius. Exclusive C/A uses rapidity/phi, E-scheme merges and stable
physical-key ties. Zero-pT input particles are counted but excluded from the
clustering tree. R0=1 is a normalization convention, not an inferred radius.
Rapidity/eta denominators use a fixed1e-12 numerical floor. Mass-squared may
be clipped only within the declared relative1e-6 rounding tolerance; larger
spacelike inputs fail with diagnostics. Under/overflow bins are explicit.
Missing tracking measurements have masks, not invented significance values.

Set assignment is exact rectangular Hungarian on detached FP32 costs, with
bounded chunks and at most6 CPU solver threads. Assignment groups are mean
kinematic SmoothL1, applicable tracking SmoothL1, category CE, charge CE and
validity BCE; omit tracking regression if inapplicable. The matched loss is
half uniform and half pT-weighted, with uniform fallback for zero target pT,
plus0.25 balanced presence BCE and0.25 count CE. Every jet contributes equally.
Scalar reductions average valid channels within blocks, blocks within families,
then families and jets. Shared multiplicity/dispersion have one owner.

## Execution and publication

Hydra `gpus/default/normal`, one typed RTX A6000,6 CPUs,32000MiB, setup12h,
fit cap48h. CPU release/report request1 CPU and4000MiB for1h and accept the
already observed one-to-two CPU allocation rounding. Workers pin exact clean
pushed code and installed environment, absolute project paths, no user site,
and environment lib first in LD_LIBRARY_PATH.

The bounded dry review covers33 jobs: setup, durable CPU release,30 science
fits and report. Release has afterok setup. Embedding students wait only for
their own teachers. Other science fits are independent; report uses afterany.
The exact interleaved order is in RECIPE. No accuracy-based dependencies,
array concurrency cap, priority mutation or login-host background dispatcher.

Setup authenticates inputs, prepares complete targets, runs actual miniature
teacher fits and clean bank exports, all22 student fit/reload paths, original
native numerical replay, full/partial backward and separate longest-support
stress. Twelve warmed representative updates measure each path. Estimates
include validation, export, startup and probes,2x safety and fixed allowances.
Admission requires <85% host RAM, <88% GPU RAM and the fixed time cap. A real
student also runs a disposable miniature using its actual selected banks.
Local tests do not certify the remote GPU or the real data.

Output allowance is32GiB plus2GiB personal headroom after transfer. Setup must
inventory exact arrays/checkpoints and publication temporary space. Files are
published atomically; manifests appear only after completion. Existence alone
is never reuse evidence. Exclusive claims and append-only exact-ID journals
precede submission; completed ledgers are idempotent. Partial/ambiguous
submission stops for inspection. Never erase claims, retry blindly or cancel
jobs by broad names.

Completed/provisional reports are read-only and show missing controls as
missing. Teacher task loss is never presented as classifier recovery. Student
tables use fresh C_U/C_CE deltas, historical anchors, and explicitly overlapping
original50k. Negative results remain visible.
