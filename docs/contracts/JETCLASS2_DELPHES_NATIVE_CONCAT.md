# Native concatenation oracle contracts

Family: `JETCLASS2_DELPHES_NATIVE_CONCAT_*/v1`.
Authority: [500k plan](../plans/JETCLASS2_DELPHES_NATIVE_CONCAT_500K_PLAN.md).

SPEC binds source commit/project, exact reference campaign and completed task
hashes, foundation/membership, validation partition, native input/model recipe,
fixed training/node seeds, execution site and resource ceiling. PLAN contains
exactly one job and no dependencies on mutable queues. ACCEPTANCE authenticates
installed Weaver, actual native input training, parity, capacity and allocation
before the single scientific fit. TRAINING_REPORT, VALIDATION_REPORT and
COMPLETE bind the spec and each other's hashes and exact output bytes.

This model is an offline-enabled oracle, not deployable. The source embedding
is an explicit exception local to this family; ordinary JetClass2 input/model
contracts remain unchanged. Combined support is native offline + native HLT,
not a matched union or persistent-HLT U000 plus HLT. Source metadata has two
codes only. No matching/index/identity or label features are permitted.

All reusable artifacts are content hashed and atomically published. Reuse
requires parent and output validation. Selected weights alone do not establish
successful completion. Final-test access and rolling resume are absent.
Scientific quality cannot fail the job. Source drift, unauthorized inputs,
corruption, nonfinite execution, capacity/resource or identity mismatch fail
closed. Creation/submission/worker writes are confined to a fresh study root;
all reference campaigns, matching assignments and raw data are read-only.
