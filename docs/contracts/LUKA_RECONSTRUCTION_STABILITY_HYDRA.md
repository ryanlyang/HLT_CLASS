# Reconstruction stability contracts v1

Authority: [approved 42-fit design](../plans/LUKA_RECONSTRUCTION_STABILITY_IDEAS.md).
New artifacts use `LUKA_RECONSTRUCTION_STABILITY_<KIND>/v1`, content hashes,
explicit parents and inherited exploratory-development scope. Earlier
reconstruction/curriculum/diagnostic/repeat artifacts retain their own identities.

`RECIPE` freezes14 configurations, seeds7/8/9, exact42-fit order,30 data passes,
global128 SyncBatchNorm microbatches, FP32/TF32-off, complete OFFLINE TRAIN
support, native CE1/U100-w5 and the four readouts. The optimizer clock uses the
accumulation group's ending example count. The shared encoder is `mod.*`
excluding `mod.fc.*`; named ownership intersects original weight-decay groups.
No inference head, validation target or new data split is introduced.

G30 caps globally SUM-reduced sample-weighted reconstruction gradients with
rho0.3 and an example-clock norm EMA; PROJ projects only the auxiliary component
against the combined base gradient. Decoder gradients keep the original weight.
DEC2 updates decoder-owned AdamW states twice before the joint step during
3<t<=9; native/predictor gradients are absent during those steps. An optimizer
object may own all groups, but AdamW state advances only for present gradients.
C256/R256 and C512/R512 hold parameters fixed over2/4 microbatches, weight both
losses by actual group population and flush partial groups. The original
three-epoch warmup/cosine schedule has no optimizer reset or batch LR scaling.

`BN_MEMBERSHIP` freezes8192 identity-hash-selected TRAIN rows and ordering.
`SOFT_CALIBRATION` freezes512 other TRAIN-selected rows, the discarded technical
native seed2026111501, derived isolated decoder seed, initialization digest,
per-jet cost spreads, median/floor scale and numerical qualification. Calibration
uses no validation outcome. SOFT uses full real/null column masses, log-domain
FP64 transport, at most500 iterations, marginal/gap tolerances1e-4 and detached
optimal-plan envelope gradients. Empty targets retain the original presence/
count loss. No unconverged-plan acceptance, target cropping or hard fallback
inside the soft interval is permitted. At position9 the declared hard phase
begins. Nonfinite or unconverged execution fails with diagnostics; poor finite
classification metrics never constitute a gate.

The solver uses ordinary log scaling for50 iterations, then log-potential
relaxation1.8 if still unconverged. This numerical acceleration is pinned in
the protocol; the final objective, marginals, temperature and tolerances are
unchanged. The null-mass numerical qualification exercises this path.

`SPEC` pins clean pushed implementation, original58d0c0ed, qualified benchmark
1094dea4, environment, populations, bank and full-support source inventory.
`STORAGE_INVENTORY` measures serialized native/full model sizes at registration
and projects42 scientific outputs,42 startup/42 setup miniatures, four selected
readouts and overhead with1.25 safety. The existing298-slot local estimate is
29GiB plus1GiB reserve, subject to installed-serializer inventory and real setup.
Registration requires a fresh separate root and a sufficient personal quota.

`QUALIFICATION` and70 `PARITY_STEP` records cover14 configurations at
positions3,3.01,4.5,8.99,9 with complete common-start model/optimizer/runtime
states. Serial and four-rank references preserve individual microbatch BN
populations. Dropout is disabled only in these comparisons. Full reference
tensors are broadcast transiently in memory; their SHA256 digests, named-buffer
differences, loss/gradient/parameter/moment errors and counter equality are
retained. Tolerances: loss1e-3, gradient/parameter/AdamW relative L2 5e-4,
buffer absolute1e-4. This avoids retaining redundant large parity tensor files.

`INITIALIZATION` binds component hashes, seeded full TRAIN permutation and
rank dropout seeds. All42 setup miniatures must prove same-seed pairing and
fresh-seed independence. Decoder, evaluation and probe work preserve the native
training RNG stream. Technical trained weights are discarded before science.

`RESOURCES` measures every configuration's expensive phases, longest HLT,
largest OFFLINE, partial11 and final accumulated groups, four full-validation
readouts and TRAIN BN calibration. All42 miniatures reload native exports.
`ACCEPTANCE` recomputes requests with2x timing safety and30min fixed overhead,
48h fit cap,12 CPUs/64000MiB, per-card memory and aggregate host gates. The
measured storage projection must fit the registration allowance. Setup6h and
CPU release/report1h are technical ceilings. No accuracy acceptance threshold.

`EPOCH`, `TRAINING_REPORT`, `DEPLOYMENT`, `FIT` bind all readouts and their
separate original-rule checkpoint selections. EMA05/EMA10 average native
parameters after position3 with example-count half-lives; RAW_BN and both EMAs
recalibrate only copies' BN on fixed TRAIN rows with dropout disabled. Live
buffers/RNG are checked unchanged. Every selected export is one HLT-only ParT;
prediction identities, metrics, epoch files and reloads are authenticated.
RAW is primary; no max-of-four selection is reported as the main result.

`PLAN`, `SUBMIT_CLAIM`, `RUN_CLAIM`, `TASK` and exact-DAG journals bind exact
commands/IDs. Bootstrap submits `setup_pascal` then `release_pascal` afterok;
release qualifies setup and submits42 independent fits in registered6+18+18
order, followed by `report_pascal` afterany on all42. There are45 jobs total.
Full-envelope scheduler review precedes authorization. Accepted complete
submissions are idempotent; partial/ambiguous submissions stop for inspection.

`RESULTS`/`PROGRESS` preserve same-readout matched-control and ordinary-R24
deltas, explicit three-seed denominators, mean/min/sample SD, positive pairs,
50/52 thresholds and fixed epochs21-30 mean/min/last/count/streak. Missing
controls yield absent deltas. Readers authenticate artifacts without requiring
future-write space. Original50k overlaps development validation; final test is
sealed and no independent-test claim is made.
