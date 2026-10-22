# Direct U100 embedding prediction, weight 5

Status: implementation plan; a separate tier3 experiment, not the debug
embedding-only20 followed by joint30 experiment.

## Scientific registration

Train one fresh HLT-only Particle Transformer for exactly 30 epochs, using
CE + 5 * mean SmoothL1(predicted embedding, frozen COARSE_U100 embedding).
There is no student logit KD, fusion, warm start, embedding-only phase, or
teacher retraining. The frozen teacher is the regular single-encoder U100
selected checkpoint from fixed30 parent commit
74aa58c582ff6f497a2c6d91c503f5489fac3e9a. Do not substitute another U100.

Use the existing authenticated 299,234 TRAIN / 74,895 validation ALLJETS split,
HLT D000 caches, batch128, fixed30 learning-rate schedule, AdamW settings,
initialization and sampler seeds from the matched M0HLT/OFFLINE-w5 experiment.
The existing small 128-to-128-to-128 predictor reads the final normalized jet
representation before the classification head. It does not feed the classifier.
Export teacher embeddings in FP32 once on TRAIN, standardize using TRAIN-only
mean and population standard deviation (floor 1e-3), and join targets by exact
ordered identities and labels. Validation U100 inference authenticates the
teacher checkpoint only; no validation embedding targets are fitted or used.

Select the checkpoint using the existing full-validation AUC/tie-break rule.
Save the native HLT backbone only and verify probability readback without the
auxiliary head. Report M0HLT, OFFLINE, OFFLINE-w5, COARSE_U100 oracle and the new
student on full validation and original50k (report-only). This is single-seed
exploratory development: the former test was retired and there is NO independent
test. More favorable metrics are not an admission requirement.

## Explicit no-new-preflight execution policy

The user authorizes one direct tier3 job with zero new preflight jobs and zero
technical training epochs. This narrowly replaces the generic requirement for
a new remote miniature for this experiment; it does not waive source, lineage,
input, numerical, allocation, checkpoint or exact-plan checks. Authenticate
the completed OFFLINE-w5 and its accepted resource report in their original
pinned source. Require the same A100 GPU/environment, 6 CPUs, 90,000 MiB,
population and student recipe. Request max(360, donor w5 request + 120) minutes,
at most 24 hours. These are inherited estimates plus an explicit setup allowance,
not newly measured U100 timing. Failure preserves outputs and requires review.

The single allocation authenticates U100 readback, builds compact TRAIN targets,
then executes the real fixed30 fit. No stress/parity training or automatic
retry is hidden in setup. Require an exact pushed clean commit, saved dry plan,
explicit plan-hash authorization and one exact submission journal.

Reserve 2 GiB persistent storage. Reuse read-only particle caches; retain only
compact targets, one selected model, predictions, results and provenance.
Do not delete shared artifacts, change holds, cancel jobs, modify Fresh_check,
or alter the existing D066 or debug20+30 campaign. The historical canceled U100
job is not a requirement: authenticate the completed replacement artifacts.

## Implementation / verification

Additive package `luka_u100_embedding`, thin CLI/worker and direct queue helper.
Frozen donor baseline is dd51e8e25770482134809dbb34d4766f8dac74b9. Reuse the
D066 direct execution and matched training semantics with a distinct U100
artifact family. Test target identity/role, teacher lineage, exact objective,
fixed30 selection, HLT-only deployment, inherited admission, single-job dry/live
submission, duplicate protection, saved results and a local installed-Weaver
CUDA miniature. Local miniature evidence does not claim new SPORC validation.
