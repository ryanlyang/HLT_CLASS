# FullStat retained U100 predictor on Oscar

Active scientific plan, 2026-10-30. User authorized two fresh fits: retained
predictor first, then the identical expanded architecture with CE only. The
completed U100-w5 run is the historical reference. This port supersedes the
small-data population/schedule in the [strategy handoff](../LUKA_U100_RETAINED_PREDICTOR_HANDOFF.md)
for this campaign only; it does not change historical registrations.

## Population and architecture

Reuse the authenticated genuine FullSim FullStat 3,000,000 TRAIN and 500,000
development-validation rows. Keep 1,500,000 final-test and 105,325 unused rows
sealed. Pair identity, row order and the approved stratified jet-row split are
unchanged. Event-level independence and overlap with the old trial remain
unverified. All inference inputs are native D000 HLT features/vectors/masks.

Let h be the fresh ParT's 128-dimensional jet representation, g be
Linear(128,128), GELU, Linear(128,128), and z_hat=g(h). The deployed classifier
is Linear(256,11) on concatenate(h,z_hat). It learns from labels; no actual
teacher value enters this classifier. Keep g and this classifier at inference.
This expands deployment capacity; it does not add measured information to h.

Initialize the h-half and bias from the matched fresh native classifier and
the prediction-half to zero. Use isolated predictor seed 20261019. Freeze the
unused native classifier. Both arms have identical starting state, sampling,
optimizer and architecture. Predictor and combined classifier use FP32;
the native encoder follows the inherited BF16 CUDA path.

## Objectives and optimization

| Task | Loss | Teacher targets during its fit |
| --- | --- | --- |
| train_retained | CE(logits,y) + 5 SmoothL1(z_hat,z) | Original COARSE_U100 TRAIN embedding bank |
| train_control | CE(logits,y) | None |

SmoothL1 uses beta=1 and mean over examples and 128 coordinates. z is the
frozen teacher's native representation standardized with TRAIN-only moments
and the original 0.001 scale floor. Neither z_hat nor h is detached. The
combined classifier receives only CE gradients. At initialization its zero
prediction-half blocks CE gradients into g for the first backward, but CE
updates that half and the embedding loss already updates g. The control
learns this same branch using CE alone. No KD, readout loss, ramped auxiliary
loss, relational objective, warm start or additional teacher is registered.

Use 20 full epochs, batch128 including each final partial batch, the original
FullStat20 AdamW/learning-rate recipe, initialization/sampler seeds and AUC
checkpoint tie-breaks. Preserve optimizer grouping and all losses in reports.
Do not stop for poor finite metrics. Save complete model, optimizer and RNG
state plus a hashed receipt after every epoch before publishing final outputs.
No automatic resume is promised; a later publication failure retains these
states for explicit recovery.

## Parent and resource reuse

Authenticate the completed corrected readout campaign at source
db9afe8b87cdef6f3489f9b85603f52cf7d2084f in its original worktree. It must prove
the original U100-w5/COARSE_U100 lineage, Oscar acceptance, bank and reference
receipts. Do not use hybrid UE100 or small-data targets. The new namespace is
additive: existing donor Python/shell files remain byte-identical.

No standalone preflight or target-export job. Each fit performs bounded
installed-Weaver parity, real graph/gradient, retained deployment readback,
maximum-support batch128, timing and memory checks before starting fresh.
The retained arm additionally replays the original teacher bank with the
corrected readout export batching/backend policy. The CE control loads no
teacher bank and performs no teacher forward. Registration/submission may
authenticate historical teacher artifacts for the experiment as a whole.

Two Oscar L40S GPU jobs, each 6 CPUs / 180000 MiB / 24 hours, no requeue.
The control has afterany on the retained job: order only, not a scientific
dependency. A failure in retained does not suppress the registered control.
Inline timing must fit the full 20 epochs plus validation/publication margin;
resource or nonfinite failures stop that job. Reserve 8 GiB results plus
1 GiB free headroom. Native/deployed parameter counts, state bytes and bounded
forward timing and incremental CUDA allocation peaks are saved; these are not
production HLT latency certification. Preserve the scientific fit's peak
memory evidence before resetting counters for the inference measurement.

## Outputs and operational boundary

New immutable LUKA_FULLSTAT_RETAINED_* /v1 identities distinguish the retained
graph from native U100-w5 and frozen readout. selected.pt contains the entire
retained graph in both arms. Validate strict state readback, ordered prediction
identities, selected metrics and all epoch receipt hashes before publishing
task success. Results are available for retained before control completes;
show the missing control explicitly, never substitute the historical baseline.
Report AUC/accuracy and R50/R30 with the same AUC-selected checkpoints and
M0HLT/OFFLINE recovery anchors. Single-seed adaptive development evidence only.

The detached launcher authenticates source/parents, writes a full dry plan,
then submits exactly the authorized two jobs with a crash-safe exact-ID journal.
It has no later prompt, retry, cancellation or cleanup. A partial submission
must be inspected rather than automatically resubmitted. See the
[contract](../contracts/LUKA_FULLSTAT_RETAINED.md) and
[runbook](../LUKA_FULLSTAT_RETAINED_RUNBOOK.md).
