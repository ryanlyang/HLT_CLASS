# Luka privileged gate contracts (scientific v1, preflight v2)

Authority: [plan](../plans/LUKA_PRIVILEGED_GATE_PLAN.md).

`LUKA_PRIVILEGED_GATE_{RECIPE,ANCHOR,GATE,SPEC,PLAN,TASK,BANK,
TRAINING_REPORT,RESULTS,SUBMISSION_CLAIM}/v1` use canonical SHA256 content hashes,
exact parent hashes, immutable publication and execution-source lineage.
Old attention/fixed30 artifacts keep their original contracts and sources.

The corrected execution additionally registers ATTENTION_EXECUTION/v1 inside
GATE/v1, authenticated by the gate/source hashes and exact registration check.
It specifies scoped math SDPA for CUDA bias-only gradients at registered gated
attention calls, native dispatch otherwise, and exception-safe restoration.
The scientific RECIPE/v1 is unchanged. Old gates are not rewritten or silently
loaded under the new registration.

`LUKA_PRIVILEGED_GATE_PREFLIGHT/v2` supersedes v1 acceptance under this source.
It binds the execution policy, FP32/BF16 frozen-mode evidence and27 stresses:
13 joint TRAIN,13 validation inference, and one early maximum-support frozen
REPLACE_STAGED TRAIN stress. The frozen witness requires classification-only
gate gradients, a gate optimizer update, unchanged backbone weights/buffers,
one scoped math call, successful unfreeze and no override during inference.
FP32 math-reference logits/gradients must pass; their BF16 fields are explicitly
null, with finite gradients/updates still mandatory. No quality threshold is
introduced. Version1 reports and absent/altered frozen evidence are rejected.

The registered graph, seeds, view roles, gate placement/rank/bound, warm-start
policy, losses, fixed30 recipe and historical-test flags are checked at reuse.
Warm-start state uses native `mod.*` plus optional `gate.*`; only first-rung
gate parameters are new. Replacement copies all privileged teacher weights,
but input second half becomes own-view features. Strict loading rejects other
keys/shapes. Optimizers are never inherited. HLT models reject context args.

Only TRAIN target banks are permitted. Each bank binds producer checkpoint,
report, coordinate, context coordinate, population IDs/labels/counts, prepared
hash and temperature2. Optional ragged factors have float32(P,9), with offsets
derived from authenticated int64 particle counts; reconstruct bounded log gates
exactly in FP32. No dense Nparticle^2
bank, no target labels/source indices as model features. Population and payload
checksums are verified before use. Partial output directories cannot be reused.

Centered log-gate MSE uses teacher minus mean over real keys, likewise student;
per-jet equal weighting and no diagonal exclusion. Saved factors are not the
teacher's native attention. Joint student classification always uses its own
predicted gate. Staged replacement freezes backbone parameters AND buffers
during the first3 epochs. Only full-validation classification selects weights.

Every task receipt binds exact dependency receipts and all output payloads.
Reports retain all scientific rows even when performance is poor. Checkpoints,
predictions, original50k secondary metrics, oracle/deployable labels and timing
are authenticated. Test commands do not exist. Raw files are not reprocessed.

Preflight receipt is accepted only after exact job accounting COMPLETED/0:0,
full-payload authentication, source/environment checks, all-node probe coverage,
native parity, role-specific stress and measured CPU/GPU/disk/time headroom.
Science is a separate immutable specification and reviewed submission plan.
No retry, automatic cleanup, job cancellation, scheduler change or submission
of unrelated workloads is implemented.

## Opt-in follow-up contracts v1

Additional `LUKA_PRIVILEGED_GATE_FOLLOWUP_{PLAN,AUTHORIZATION,ARMED,REVIEW,
COMPLETE,FAILURE}/v1` artifacts are operational and never scientific teachers.
They retain canonical hashes and immutable publication. PLAN binds the gate
hash, source commit, exact gate command, full science command ceiling template,
13 fits/23 science jobs, 1440-minute train/export ceilings, 60-second polling,
14-day wait bound, and a distinct authorization phrase. The template contains
`${MEASURED_SCIENCE_SPEC_HASH}` and must NEVER be submitted as a Slurm plan.
It is replaced only by a fully authenticated measurement-bound SPEC and exact
PLAN after preflight admission; actual train/export times must be positive
integers not exceeding the ceilings. Normalizing those two times and the spec
hash must reproduce the reviewed ceiling plan exactly, including all commands,
dependencies, root paths, resources, and job count.

AUTHORIZATION binds the saved reviewed plan and source/gate identity. ARMED
also binds the authenticated gate submission ledger and exact preflight ID.
Scheduler polling checks job ID/name/owner/account/state/exit; missing accounting
means wait, unknown/failed terminal states fail closed. REVIEW binds the live
authorization, admitted science specification and saved measured exact plan.
COMPLETE binds the fully authenticated live science ledger and all23 unique job
IDs; `training_complete=False`. FAILURE preserves the exception and explicitly
forbids automatic retry, without claiming no jobs were accepted during a
possibly partial submission. The permanent exclusive claim is never removed
automatically. Existing manual gate/science commands are unchanged.
