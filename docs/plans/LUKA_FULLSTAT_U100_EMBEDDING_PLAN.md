# FullStat U100 embedding prediction, weight 5, Oscar

Authorized 2026-10-21: one fresh HLT-only student on the existing 3M TRAIN /
500k validation split, batch128, fixed20 FullStat schedule, original M0HLT
initialization/sampling and predictor seeds. Frozen teacher is the original
validation-selected **COARSE_U100** (coordinate U100), not OFFLINE or D066.
Loss: CE + 5 * mean SmoothL1_beta1(predicted128, standardized_U100128).
No logit KD in the student, attention loss, warm start, fusion, or privileged
classifier input. The prediction head is removed for native HLT deployment.

Export only TRAIN U100 vectors immediately before native mod.fc, FP32/eval,
frozen weights. Compute fresh TRAIN population moments in FP64, floor scales
at1e-3, retain FP32 normalized targets. Never relabel the old OFFLINE bank or
reuse its moments. Preserve all existing jobs, cancellations, artifacts and
the running OFFLINE-target w50. Final test and unused rows remain sealed.

## Explicit reuse instead of another full preflight

The user's request supersedes a new per-campaign full-epoch preflight: reuse
the completed fd687181 Oscar embedding-sweep acceptance, authenticated in
its original clean worktree, exact submission journal and successful accounting.
Also authenticate the completed original FullStat parent and selected U100
checkpoint/report/predictions/receipt. Require unchanged installed environment,
GPU identity, native/predictor/training kernel semantics, prepared population,
HLT supports, batch128, fixed20 optimizer/schedule and6CPU/180000MiB allocation.
The old acceptance is evidence for shapes/implementation, NOT evidence that the
new target/weight has been measured or scientifically validated.

Queue exactly two GPU jobs in one reviewed submission: export_U100 -> train_w5
(afterok, kill-on-invalid-dep). No detached login controller or second submission.
Export must complete mandatory new-target checks before its receipt commits:
selected teacher identity, native logit/tap parity, maximum-support FP32 export,
TRAIN bank and row joins, fresh normalization and finite values, a few HLT
forward/backward updates using real weight5 targets, validation inference shape,
memory limits. No technical full epoch is run. Discard probe weights; science
starts from the original seed. Failures prevent scientific training. Ordinary
poor metrics never fail either job. These integrated startup checks are not
misrepresented as zero validation, nor as a repeated full preflight.

Reuse the worst accepted sweep training request, not a new unmeasured speed
claim. Export uses the accepted8h gate ceiling; its U100 runtime is unmeasured.
Export batch is128, and U100 support is measured in-job before full extraction.
Runtime memory guards fail closed if reuse is inapplicable. Reserve6GiB personal
durable headroom and verify cache memory metadata; no new particle caches.
The training job validates the export's exact journal-bound ID and receipt,
then runs20 epochs and publishes native selected weights, predictions, histories,
and a comparison report. Permanent claims forbid ambiguous retry/submission.

M0HLT/OFFLINE/COARSE_U100 are authenticated historical validation references.
There is no new matched CE control: user cancelled w0 and has not authorized
retraining it. Recovery is explicitly historical-M0HLT-relative. Weight and
teacher were adaptively chosen; no independent conclusion about either factor
alone, no test-selected winner, no new test access.

Implementation is additive under luka_fullstat_u100_embedding. Donor runtime:
fd687181a60c1557f98d7498f577e26d6dbb35a7 embedding sweep and its unchanged FullStat
cache/model/runner. Integration baseline5bf9a00d722c867f652588572cb489451bb1a12e.
Require focused baseline, exact same-target/same-weight kernel parity, real
local installed-Weaver CUDA two-job miniature and negative reuse/lineage/DAG
tests. Genuine Oscar acceptance is reused, not newly executed locally; new
target checks run on Oscar as part of export before the queued fit is admitted.
