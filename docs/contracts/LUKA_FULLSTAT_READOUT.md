# LUKA_FULLSTAT_READOUT contracts

Authority: [scientific plan](../plans/LUKA_FULLSTAT_READOUT_PLAN.md).

SPEC and STARTUP now use v2 / schema_version 2. TEACHER_REPLAY is new at v1;
REUSE, RECIPE, READOUT, EPOCH, TRAINING_REPORT, TASK, RESULTS, PLAN and LAUNCH
retain v1 / schema_version 1. All use canonical content hashes, explicit parent
hashes and inherited FullStat sealed-test flags. Historical records are unchanged.
SPEC binds exact source, original U100-w5 parent, bank/teacher references, fresh
node, fixed counts/recipe, runtime and resources. Only `train_readout` is permitted.

READOUT binds selected teacher checkpoint/report and original bank manifest;
payload contains only FP32 mean, scale, weight, bias. Reconstruct those bytes
from original artifacts before accepting reuse. Verify native teacher logits
against unstandardized bank targets on deterministic TRAIN witnesses.

SPEC/v2 binds TEACHER_REPLAY/v1. Select at most five evenly spaced original
export batch indices, including the first and last, plus the batch containing
the maximum-support row. Replay each entire contiguous batch of 128, with the
original partial tail and dynamic padding. This checks at most six batches /
768 TRAIN rows. No validation targets are read. Use the ordinary production
backend and disabled autocast, exactly as the original exporter does; do not
wrap historical bank replay in the strict fresh-model `parity_backend` context.
Elementwise tolerances remain rtol=2e-4, atol=2e-5 for both reconstructed native
embeddings and logits. Nonfinite outputs and parity failures still stop the job.

STARTUP/v2 records replay policy hash, exact batch ranges/rows, maximum-support
row, current backend flags and maximum absolute embedding/logit errors. These
flags describe the replay; the old bank manifest did not record historical
flags. Batches/coverage, policy, finite errors and backend record are validated
on reuse. Installed-Weaver and fresh native-model comparisons still use strict
parity mode on both sides. v1 SPEC/STARTUP records are rejected by new execution;
failed runs keep their immutable source/spec and a retry uses a fresh root.
This versions technical acceptance only; the scientific RECIPE is unchanged.

EPOCH binds spec, prepared cache, bank, readout and recipe; its immutable state
contains student (including predictor), optimizer, CPU/CUDA RNG and update.
TRAINING_REPORT covers all 20 epochs and the native selection rule. TASK binds
every published output, source and exact accepted job ID. Results require
matching ordered validation identities/labels and recomputed saved metrics.

No inference on final_test/unused, no validation-derived target normalization,
no teacher gradients or readout parameters, no automatic retry/resume. Resource
or corrupt/nonfinite/lineage failures stop; poor finite science does not.
