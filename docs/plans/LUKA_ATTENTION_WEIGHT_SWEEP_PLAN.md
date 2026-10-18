# Luka attention weight sweep: 0.3 and 1.0, bounded storage

Authorized 2026-10-18. This separate campaign supersedes neither the completed
lambda 0/0.1 attention experiment nor the privileged-gate experiment. The old
attention contract's no-automatic-deletion rule remains valid for old artifacts;
the explicitly authorized retention policy below applies ONLY to new artifacts.

## Science

Two fresh three-rung chains from the same authenticated fixed30 COARSE_U100:
U100 -> D066 -> D033 -> D000. Constant attention coefficient 0.3 runs on SPORC
debug; coefficient 1.0 runs on SPORC tier3. Both require the same A100 family,
installed environment, 6 CPUs and 128000 MiB. Each partition has its own measured
preflight; do not infer timing portability. One sequential job per ladder keeps
ephemeral teacher targets on the allocated node. A measured whole-ladder request
must fit 24h, even on tier3; never shorten epochs or alter particles to fit.

Reuse the old head-mean block7 pre-dropout FP32 attention surface and per-jet
forward KL. Objective is0.25CE+0.75T²KL(logits,T=2)+lambda*KL(attention).
True batch 128, exactly 30 epochs, original seeds and optimizer, fresh model at
each rung, same AUC-first checkpoint selector. No warm starts or loss schedules.
Each arm uses its own selected immediate teacher for BOTH target kinds.
ALLJETS 299,234/74,895; original 50k report-only. Historical test was consumed and
retired into TRAIN; there is no independent test. All rows survive poor metrics.
Import the completed lambda 0/0.1 campaign as authenticated references, not new
fits or independent confirmation. Primary endpoint: final D000 AUC; also retain
accuracy, class rejection, histories and selected validation probabilities.

## Retention and execution

No raw-data, prepared-view, existing model, old bank, old experiment, source
worktree or Fresh_check deletion is authorized. No changes to old files.

Generate byte-schema-identical FP32 attention banks on explicitly configured
job-local scratch, NEVER silently falling back to home/persistent storage. Use a
fresh private directory per job. Only one consumed teacher bank is needed at a
time: verify/save the child's selected checkpoint, predictions and report; then
delete that bank's exact authenticated payloads. Export/validate the child's
bank before deleting its checkpoint. D000 has no descendants; remove its
checkpoint after selected predictions and metrics authenticate. Keep permanent
hash-bound bank manifests, checkpoint descriptors, consumer receipts, deletion
intent/completion records and results. No symlinks, path escapes or unknown
files may be deleted. On failure, preserve diagnostic metadata; never declare
an incomplete step reusable or automatically restart it. A process/job loss can
lose scratch; recovery requires a newly reviewed fresh run, not silent reuse.

Retain per-jet validation probabilities, labels, identities and original50k
membership so future R30/statistical diagnostics need no models. State explicitly
that retired checkpoints cannot perform new inference without retraining.
Results and preflight validators understand the new retirement contract; they
never treat missing old-format payloads as valid. Disk budget counts one bank,
bounded publication/read buffers and reserve on scratch, and separately counts
compact persistent results and metadata. Temporary checkpoint publication stays
on scratch, not persistent storage.

Preflight exercises the production chain with one full pass per rung, real
checkpoint/native readback and retirement. It also checks installed-Weaver
FP32/BF16 parity, longest-support TRAIN/validation batch128 stress with the
actual coefficient, and measures memory, storage, loading, exports and complete
epoch timing. Technical probes never become scientific teachers. Exact pushed
source, saved dry plan, successful scheduler accounting and accepted preflight
are required before separate science authorization. No submissions by coding.

## Verification

Focused tests cover both coefficients, unchanged original recipe, matched seeds,
three-rung production execution, scratch-only banks, immutable result readback
after retirement, dependency/corruption/path-escape/symlink/partial-delete guards,
partition routing, measured whole-ladder admission and exact-plan authorization.
Local installed-Weaver tests do not replace real SPORC/A100 acceptance.
