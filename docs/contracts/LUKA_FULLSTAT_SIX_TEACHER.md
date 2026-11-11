# FullStat six-teacher contract v1

Authority: [active plan](../plans/LUKA_FULLSTAT_SIX_TEACHER_PLAN.md).
New artifacts use `LUKA_FULLSTAT_SIX_TEACHER_<KIND>/v1` with canonical content
hash, explicit parent hashes, development-validation scope, sealed final test,
and false event-independence/old-trial-overlap claims. Old artifacts keep their
original contract and original-source validators.

`REUSE` binds successful subclass job8761518 at16546c0e and completed fine-tail
04d1f0f5, exact teacher references, runtime bytes, original environment/GPU,
prepared population and frozen subclass files. `SPEC` binds that evidence to
the new pushed source, recipe, node, resources and optional same-source resume.
The registered order is OFFLINE, CONCAT, FUSION, UE100, DE066, DE033. Here UE100
is the hybrid model. Original COARSE_U100 is only historical reference lineage.

`RECIPE` registers CE1, three10/3 embedding objectives, three5/3 ramped embedding
objectives, six1/6 ramped T²-KL objectives (T2), contrastive0.3 and subclass0.5
with the original three-epoch ramps. Losses reduce over all examples, including
the final partial batch. Teacher targets are detached. Head initialization is
isolated from native encoder/classifier initialization and sampler RNG.

`KD_EXPORT_RECIPE` and `KD_BANK` bind each new CONCAT/FUSION TRAIN probability
bank to native selected weights, report, original view owner, prepared hash,
and new source commit. Role, temperature, class count, batch, ordered identities,
labels, dtype, finite normalized probabilities and payload hash are validated.
No inversion of embedding normalization is used for these KD logits.
`EXPORT_TIMING` is diagnostic; it cannot substitute for a complete bank.
Payload is atomically published before its immutable manifest.

`STARTUP` records real backend, installed-Weaver parity, exact candidate sampler,
connected gradients, CE-only native classifier, deployment reload equivalence,
full-objective resource/timing evidence and discarded probe. It is not a
scientific fit. `EPOCH` binds full model/optimizer/RNG state and loss/validation
record to all bank hashes and recipe. `TRAINING_REPORT` requires20 contiguous
epochs and the deterministic best-AUC selection rule. `TASK` binds all saved
exports, subclass files, epoch states and selected native output files to the
accepted scheduler job. Completion also requires COMPLETED/0:0 accounting.

`RESUME` binds unsuccessful terminal accounting, complete export files and any
contiguous completed epochs to the old spec. Partial exports are not reusable;
complete exports can be reused even if no epoch finished. New root, same source,
same recipe/teachers and exclusive claim are required. Completed epochs are
never repeated; all20 allows publication-only recovery. No implicit retry.

`PLAN` has exactly one `train_six` job. Exact plan hash, explicit bounded launch
authorization, canonical dry ledger and append-only submission journal guard
submission. `LAUNCH_*` receipts distinguish detached launch from successful
scheduler submission. Runtime is one L40S/6CPU/192000MiB/48h,12GiB quota minimum.
No separate preflight/setup/teacher-training jobs. Finite poor metrics never
invalidate a job. Final test remains sealed throughout.

## Reviewed memory repair and export recovery (2026-11-11)

New specs bind `LUKA_FULLSTAT_SIX_TEACHER_WORKER_MEMORY_POLICY/v1`. Each worker
reports its current-address-space peak through Linux `VmHWM`, with PID/unit/
positive-value checks. Startup records all three peaks and the old lifetime
`ru_maxrss` values separately. Admission sums the actual worker peaks plus the
main process peak, keeping the same limits. Old startup proofs cannot satisfy
this new policy. Existing donor samplers and scientific `RECIPE/v1` are unchanged.

`LUKA_FULLSTAT_SIX_TEACHER_EXPORT_RECOVERY/v1` permits only the reviewed
FAILED/1:0 job8821154 at commit15808976126fb87cdf69ffae7a7a406cb865afb1, before
any student epochs or accepted startup. It contains the original-source RESUME
evidence, both complete export inventories, diagnostic file hashes and terminal
accounting. It rejects altered science, partial exports, student state, different
failures and unreviewed runtime changes. A fresh root and exclusive original-run
claim remain mandatory. Original KD manifests/payloads keep their hashes and
exporter source; later same-source continuation follows this provenance chain.
