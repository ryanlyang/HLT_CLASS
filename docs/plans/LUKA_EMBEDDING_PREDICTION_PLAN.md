# Immediate-predecessor native embedding prediction ladder

Authorized experiment: U000 -> U100 -> D066 -> D033 -> D000. U000 is the
existing selected fixed30 regular U000, not pure OFFLINE and not a new fit.
Four fresh students learn CE + 5 * mean SmoothL1 of auxiliary predictions
against their immediate predecessor's frozen native 128-D pre-classifier
embedding. There is NO student logit KD, attention KD, fusion, warm start,
embedding-only pretraining, U050 stage, or fixed-OFFLINE target bank.

| New student | Frozen TRAIN target model |
| --- | --- |
| PRED_U100 | Original fixed30 U000 |
| PRED_D066 | This campaign's selected PRED_U100 |
| PRED_D033 | This campaign's selected PRED_D066 |
| PRED_D000 | This campaign's selected PRED_D033 |

Export actual native features before the classification Linear, NOT the
auxiliary head output, classifier logits, probabilities or earlier predictions.
Every selected checkpoint contains only native weights; the next teacher is a
fresh native model strictly loaded from that selected checkpoint. Freeze it in
eval mode; export FP32 TRAIN embeddings once, fit FP64 TRAIN mean/population SD
with scale floor 1e-3, and save standardized FP32 targets. Refit normalization
for each new teacher. Targets detach, exact ordered jet identities/labels join,
and TRAIN-only target roles are mandatory. No particle correspondence is needed
for these whole-jet vectors. No validation targets or target-quality admission.

Use trial ALLJETS299234 TRAIN / 74895 validation (original50k report-only), not
Oscar FullStat. Historical test was consumed/retired; NO independent test exists.
Intermediate U100/D views contain privileged input and are ORACLE diagnostics;
only PRED_D000 is deployable native HLT-only inference, with predictor removed.
Do not claim all intermediate inputs are HLT-only.

Keep the existing fresh fixed30/batch128 training schedule, original per-
coordinate initialization/sampler seeds, AdamW, warmup3/cosine peak3e-4 and
floor1.5e-5, BF16 native forward and FP32 CE/auxiliary loss. Each new student has
a fresh small Linear128-GELU-Linear128 predictor with isolated initialization
RNG. Select full-validation AUC with existing CE/rejection/update tie-breaking.
Complete all30 epochs and all4 stages even if scores regress. Do not carry
weights, optimizer moments, predictor weights or RNG state across stages.

Implementation is additive relative to118aa375eea423e5e99338cd9a6eda198225ef21.
The reviewed integration baseline is b21c2d4e01211a31e57d879c5a86ca90aab2ca0b:
it adds only an unrelated FullStat multiteacher package, with every existing
scientific donor unchanged. The source guard remains strict after that pin.
New LUKA_EMBEDDING_PREDICTION_*/v1 records bind source, prepared input, original
anchor, exact predecessor checkpoint/report, ordered bank and stage receipt.
Original U000/M0HLT/OFFLINE authenticate in their pinned74aa58c5 source using
the unchanged original-anchor replay. No existing experiment is modified.

## Debug execution and compact storage

This plan explicitly uses SPORC debug A100 for its genuine production-worker
miniature and scientific runs, superseding generic Tigris site defaults only
for this experiment. One reviewed exact DAG queues setup then U100 -> D066 ->
D033 -> D000 via afterok dependencies. Every science job also revalidates setup
admission, immutable dependency receipts and exact successful Slurm accounting.
No login-node controller, manual follow-up or automatic retry.

Setup authenticates original U000 native validation readback, builds the shared
initial TRAIN bank, verifies installed-Weaver and predictor parity, stresses
maximum TRAIN backward and validation inference support at every coordinate,
and runs a disposable one-full-epoch four-rung chain with selected-native
readback/target export. These technical weights/banks are isolated from science;
science U100 uses ONLY the original U000 bank, later science stages use ONLY
their completed scientific predecessor. Technical descriptors cannot seed them.

Request6CPU,90000MiB,oneA100; setup8h and each science job6h. Project every stage
from its measured full-population epoch, loading, readback and export with1.75x
margin and300s allowance; measured host peak<85%, GPU peak<90%. Resource failure
blocks science; poor accuracy/AUC/embedding loss never blocks or skips a stage.
Require clean exact pushed source, dry review and explicit plan-hash authority.

Reuse read-only prepared U100/D066/D033/D000 caches. Fully deleted U000 blocks
may be rebuilt in RAM using saved assignments and exact original fingerprints;
partial/corrupt caches fail closed. No rematching, durable particle cache or
data reupload. No sealed/unused roles. Reserve2GiB for four scientific and three
technical compact TRAIN banks (~1.1GiB uncompressed including identity arrays),
selected native models, histories, predictions and provenance. No per-epoch
models or attention banks; no deletion of shared artifacts, old campaigns,
jobs or Fresh_check. Keep these small lineage-bound outputs for reporting.

Report each completed stage and pending stages with accuracy, AUC, recovery,
QCD rejection@50%, full validation and original50k. Baselines M0HLT/OFFLINE/U000
are historical references, not a new matched causal control. This is a
single-seed exploratory ladder, not independent evidence of a test-set gain.
