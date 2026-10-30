# LUKA_FULLSTAT_READOUT_* / v1

Authority: [scientific plan](../plans/LUKA_FULLSTAT_READOUT_PLAN.md).

New SPEC, REUSE, RECIPE, READOUT, STARTUP, EPOCH, TRAINING_REPORT, TASK, RESULTS,
PLAN and LAUNCH records use canonical content hashes, schema_version 1, explicit
parent hashes and inherited FullStat sealed-test flags. Old schemas are unchanged.
SPEC binds exact source, original U100-w5 parent, bank/teacher references, fresh
node, fixed counts/recipe, runtime and resources. Only `train_readout` is permitted.

READOUT binds selected teacher checkpoint/report and original bank manifest;
payload contains only FP32 mean, scale, weight, bias. Reconstruct those bytes
from original artifacts before accepting reuse. Verify native teacher logits
against unstandardized bank targets on deterministic TRAIN witnesses.

EPOCH binds spec, prepared cache, bank, readout and recipe; its immutable state
contains student (including predictor), optimizer, CPU/CUDA RNG and update.
TRAINING_REPORT covers all 20 epochs and the native selection rule. TASK binds
every published output, source and exact accepted job ID. Results require
matching ordered validation identities/labels and recomputed saved metrics.

No inference on final_test/unused, no validation-derived target normalization,
no teacher gradients or readout parameters, no automatic retry/resume. Resource
or corrupt/nonfinite/lineage failures stop; poor finite science does not.
