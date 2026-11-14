# New FullStat Hydra v1 contracts

Authority: [baseline plan](../plans/LUKA_NEW_FULLSTAT_HYDRA_OFFLINE_PLAN.md).

`LUKA_NEW_FULLSTAT_HYDRA_{FOUNDATION,RECIPE,PREPARATION_INTENT,PREPARED_FILE,
CACHE,SPEC,PLAN,SUBMISSION_INTENT,SUBMITTED,QUALIFICATION,EPOCH_COMMIT,
TRAINING_REPORT,RESULT}/v1` use canonical content hashes and explicit parent
bindings. All retain sealed final test and unverified event independence/old
dataset overlap. They are not aliases for historical Oscar artifacts.

FOUNDATION binds every source checksum, tree, schema, source stratum, scalar
cutflow, exact split quotas, membership payload checksum and particle budgets.
Labels 206 remain -1 with role 255. The old global label map is unchanged.
All other unknowns fail. Train/validation/test/unused role codes are 0/1/2/3.
Membership replay must match the exact seeded block/stratum allocation.
Dataset-relative identity depends on source path/hash/tree/entry, not locator.

The transfer manifest binds the foundation and explicit ordered inventory.
The standard-library receiver checks every member's regular-file type, safe
relative path, byte count and SHA256 before publication. A complete receipt
appears only after the exact inventory ends. Retries require the same manifest
and verify existing files; corruption is never overwritten. The SSH binary
pipe avoids Windows PowerShell text encoding and a second remote archive copy.
Raw snapshot bytes are opaque during transfer, including sealed/unknown rows.

Prepared caches contain only native OFFLINE TRAIN and validation ragged
features, vectors, labels and identities. Each block is identity/cardinality
joined to authenticated membership at load. Unknown rows are ignored; requested
particle spans cannot cross final-test/unused rows. No particle truncation or
automatic disk spilling occurs. Preparation is bounded by file workers.
No teacher, HLT model, balanced draw stream or test evaluator exists here.

SPEC binds clean pushed source, installed runtime, frozen foundation, original
20-epoch recipe, resource envelope and one explicit GPU model. The full two-job
plan is printed and tested against Hydra before any submission. Each sbatch
call has a durable intent before invocation and an exact-ID record afterward.
An uncertain response blocks automatic retries. Train depends afterok on
preparation. Live workers authenticate the controller allocation and source.

QUALIFICATION is a technical probe on real TRAIN/validation rows, not scientific
initialization: installed-Weaver FP32 gradients/outputs, two-pass miniature,
selected reload, longest-row batch-128 stresses, representative timings and
headroom checks. Evidence is written before resource admission. Scientific
training starts fresh or restores a complete authenticated epoch state.

EPOCH_COMMIT is published after full model, optimizer, CPU/CUDA RNG state.
Interrupted partial epoch directories are preserved; only committed contiguous
epochs are reusable. A `queue --resume` request requires all prior exact jobs
terminal and preparation complete; it submits one GPU continuation. No model
quality threshold gates success. RESULT binds the selected native checkpoint,
ordered validation predictions, all-epoch metrics and R30/R50 from the same
AUC-selected epoch. New-dataset recovery awaits its own HLT baseline.
