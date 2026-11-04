# Cross-attention message artifact contract v1

Authority: [scientific plan](../plans/LUKA_CROSS_MESSAGE_HYDRA_PLAN.md).
All new artifacts use `LUKA_CROSS_MESSAGE_HYDRA_<KIND>/v1`, schema1,
canonical content hashes, parent hashes and the existing ALLJETS exploratory
scope flags. No previous campaign's scientific artifacts are relabelled.

`ENCODER_TRANSFER` binds the original pair-campaign identity, original-source
authentication proof, imported COARSE_U100 teacher, original checkpoint and
validation prediction bytes. Its only members are `U100.pt` and
`validation_predictions.npz`; payload is bounded by128MiB. The existing Brown
bundle and validation-U100 delta retain their original contracts. Export needs
no inference or GPU. A directory or filename alone never admits reuse.

`SPEC` binds the two-teacher/twelve-student grid, exact source snapshot,
environment, original input bundle, validation delta, new encoder delta,
completed precision evidence, fixed8GiB output budget and reviewed resource
envelope. Source changes are additive relative to
`a40ae47ba87ee23020b482c17d467e2a42282e8f`; old runtime remains unchanged.

`RECIPE` declares frozen block7 U100 features, block3 HLT injection, 32/16-wide
messages, teacher CE/KD weights, all student weights/seeds and target scaling.
The process-local frozen-token cache preserves authenticated jet identities,
labels and real-particle support. Its FP32 arrays contain the frozen encoder's
BF16-evaluation outputs on CUDA. It has no persistent reusable artifact identity;
it is rebuilt from authenticated inputs inside each relevant allocation.

`TRAINING_REPORT` binds spec, recipe, prepared population and original embedding
bank for students; teachers and supervised students additionally bind the encoder
delta. Supervised students bind their selected message-teacher report. Fresh
initialization and sampler seeds, TRAIN-only normalization, every epoch's losses,
schedule, validation, selected pass and deployment reload are validated.
Teachers are privileged oracles; students accept HLT inputs only. Student
checkpoints strip embedding heads, and auxiliary-only models strip message heads.
Architecture controls never read privileged features/messages/teacher weights.

Teacher `message_disabled_predictions.npz` uses the same selected weights with
injection disabled. It is a distribution-shift diagnostic, not another selected
fit or matched training control. `FIT` and exact-job `TASK` receipts bind all
published files. `EPOCH` supports provisional live progress; final results require
FIT/TASK authentication and recomputation from saved predictions. Original50k is
an overlapping report-only subset; final-test inference is not admitted.

`ENCODER_REPLAY` records original and Hydra U100 validation metrics, probability
drift, cache preparation time/size and resource peaks. AUC portability tolerance
is5e-5; probability differences are diagnostic. This is numerical validation of
the same selected encoder, not a scientific performance gate for new teachers.
`ACCEPTANCE` binds real GPU/environment identity and all14 production-kernel
miniatures, including checkpoint publication/reload, worst-multiplicity full
batches and partial batches. Measured RAM/GPU/time must fit the reviewed envelope.
Weak finite accuracy, recovery or messages never reject a registered experiment.

`REVIEW` binds the complete setup plan and science envelope. A real exact dry
plan precedes each live submission. Exclusive claims and immutable journals
protect against duplicate submission after ambiguous scheduler responses.
The two teachers and three architecture controls are independent. Each of nine
supervised students has only its width-specific teacher dependency. The report
uses `afterany` on all14 fits. No artificial lanes or concurrency cap.

`finish` reuses the original review/authorization and successful exact setup.
A fully submitted science ledger is idempotent; an active or partial science
claim requires inspection. Do not delete claims, edit immutable specs or reuse
failed output directories. Use a new source-bound campaign after a code fix.
