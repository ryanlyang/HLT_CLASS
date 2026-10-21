# Luka extended no-KD embedding sweep, SPORC tier3

User authorization (2026-10-20): implement eight new strengths **3, 5, 20,
50, 100, 200, 500, 1000** on tier3. Existing 0.75/1.5/10 runs and the original
0.3 run remain immutable references, not new jobs. This plan authorizes a
separate registration; it does not alter LUKA_EMBEDDING_WEIGHT/v1 or old source.

Every model is a fresh HLT-only native ParT, minimizing
`CE + lambda * mean SmoothL1(predicted_embedding, frozen_OFFLINE_embedding)`.
Reuse the original 0.3 run's authenticated TRAIN-only standardized 128-vector
bank and unchanged training-only predictor. Its outputs never feed the class
head; remove the predictor from deployment. No logit KD, attention loss,
ladder, warm start, changed normalization or new input features.

Hold fixed the original M0HLT initialization and sampler seeds, split, batch128,
AdamW, learning-rate schedule and 30 complete epochs. Trial ALLJETS has
299234 TRAIN / 74895 validation rows. Select by full-validation AUC first
with the existing deterministic tie rule. Original50k is overlapping report-only
validation, never a selection set. There is NO independent test: the historical
holdout was retired. This adaptive one-seed validation sweep identifies candidate
weights, not an unbiased final result or proof of monotonic improvement.

One new tier3/A100 preflight loads HLT and the shared target bank once, checks
installed-Weaver and zero-auxiliary parity, each role's longest-support batch128,
and a full technical epoch plus native checkpoint roundtrip for EACH coefficient.
Technical states never initialize science. CE, unweighted auxiliary loss,
weighted auxiliary loss, AUC and running best AUC/epoch are logged. Even1000
is tested exactly; poor scores must not fail admission or cancel other arms.
Nonfinite tensors/gradients, corrupt bytes, source drift or insufficient
resources fail closed. No coefficient clipping or loss rescaling is introduced.

After exact successful gate accounting and measured admission, review/authorize
eight independent scientific jobs. All use SPORC tier3,
reu-aisocial/qos_tier3, one A100,6 CPUs,90000MiB; measured per-arm walltimes
must fit24h. The gate ceiling is8h. The default gate/science commands remain
manual. Explicit opt-in automatic follow-up is now authorized by this plan:
review the exact eight-command graph with 24h ceilings and separately approve
`AUTHORIZE_LUKA_EMBEDDING_SWEEP_FOLLOWUP_V1`. A detached submit-host controller
queues or binds this exact gate, waits for COMPLETED/0:0 with matching job name,
owner and account, then authenticates the unchanged full technical receipt,
source, input/target lineage and measured resource admission. It saves and
checks the measured exact-plan dry review against the authorized graph before
submitting all eight jobs, with no further prompt. No scientific or preflight
checks are skipped. This separate authorization does not change the base
GATE's `automatic_followup=False` (no implicit continuation).

The controller polls every60s for at most14days, uses permanent atomic claims
and exact submission journals, and never takes over an existing science root.
An accepted gate is not resubmitted. Missing accounting waits, never passes;
terminal failure, missing/corrupt evidence, source drift or ambiguous scheduler
responses stop continuation. No automatic retry, cancellation, overwrite or
deletion. Inspect logs/journals before recovery. No compute node is held while
waiting; nohup survives an ordinary SSH disconnect, not a submit-host reboot.

Reserve2GiB persistent headroom for the new campaign, checked before/after
execution; share the original bank and prepared HLT cache read-only. No new
particle caches, target-bank copies, epoch checkpoints or optimizer snapshots.
Retain compact preflight evidence, one selected native checkpoint, validation
predictions, complete epoch history and receipts per fit. There is no automatic
deletion, including of older models or Fresh_check. Failed/partial directories
remain for inspection.

The results command authenticates completed rows and prints pending arms,
full-validation and original50k accuracy/AUC, AUC delta and recovery against
the matched CE/OFFLINE anchors, with original0.3 included. The historical
0.75/1.5/10 campaign retains its own original-source results command; it need
not finish or be rerun to launch the new sweep. No log-derived provisional
score is silently promoted to an authenticated comparison row.

Required local tests cover all eight losses/gradients, unchanged original
recipes, exact matched schedules, full30 native checkpoint selection, input
alignment, corruption and numerical failure, actual installed-Weaver/CUDA
miniature workers, exact tier3 dry/live plans, uncertain-submit refusal,
storage/source guards and partial results. Follow-up tests must exercise exact
eight-job success, already accepted gates, duplicate claims, source/receipt
failures, malformed/absent/terminal accounting, bounds, concurrent manual
science, partial submissions and unchanged scientific registration.
Local tiny tests do not replace
the genuine SPORC full-population preflight.
