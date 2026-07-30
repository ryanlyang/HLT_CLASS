# Testing and Acceptance

## Six-layer ladder

1. Pure-Python contract and unit tests.
2. NumPy/PyTorch tensor and model tests.
3. Shell, DAG, and static contract tests.
4. Synthetic end-to-end and failure-injection tests.
5. Authoritative installed-Weaver FP32 parity.
6. Real JetClass Tigris miniature through production workers.

Passing an earlier layer does not imply that a later layer passed.

## Local commands

```powershell
python -m pip install -e . --no-deps
python -m pytest -q
git diff --check
```

To avoid source drift from tracked or visible bytecode:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q -p no:cacheprovider
```

## Required coverage by transfer block

Block 1:

- package imports;
- all repository-relative Markdown links resolve;
- pytest discovers and runs the scaffold tests.

Block 2:

- class and filename mapping;
- required ROOT branches;
- stable identity and manifest hashing;
- balanced, globally mixed, disjoint deterministic splits;
- chunked ROOT reads and retry/failure behavior.

Local Block-2 coverage uses real synthetic Uproot/Awkward ROOT files, including
all 14 channels, exact one-hot labels, padding/truncation, jagged-length
failure injection, chunk-boundary reads, row-order restoration, CLI capacity
failure, manifest tampering, and immutable publication.

Block 3:

- exact strength-zero identity without RNG construction;
- hand-calculated HLT-v3 scaling and clipping;
- merge, mass, ordering, PID, charge, track validity, and replica semantics;
- batch/shard-independent deterministic bytes.

Local Block-3 coverage includes exact strength-zero short-circuiting,
hand-calculated type/strength/replica scaling, true neutral four-vector merges,
mass reconstruction, stable ties, explicit validity states, field-isolation
profiles, exact correlated track response and tails, replica cycling,
batch/order/shard invariance, malformed-input failures, and frozen donor-v1
array and diagnostics hashes.

Block 4:

- immutable shard hashes and manifests;
- corruption and cross-profile rejection;
- interrupted atomic resume;
- streaming order and identity coverage;
- bitwise HLT array invariance across source-shard, output-shard, and
  processing-batch layouts;
- exact offline-parent role, label, identity, and validity-state checks;
- absence of degradation construction indices from deployable artifacts.

Local Block-4 coverage injects interruption after durable publication,
compares resumed and clean trees byte for byte, corrupts shard bytes, attempts
cross-profile reuse, reads ranges across physical shard boundaries, verifies
parent and source lineage, audits absence of construction indices, and proves
semantic HLT bytes invariant to processing-batch and shard layouts.

Block 5:

- exact ParT feature transformations and masking;
- padding, single-particle, and all-empty rows;
- real-Weaver FP32 logits, gradients, masks, and state-dictionary parity;
- BF16 finiteness as a separate non-authoritative path test.

Block 6:

- exact resume equivalence;
- schedule and checkpoint tie rules;
- nonfinite failure;
- poor-performance continuation;
- all frozen metric edge cases.

Local Block-6 coverage compares uninterrupted and interrupted/resumed model,
optimizer, scheduler, scaler, sampler, replica-cycle, history, and selection
state; exercises a deliberately poor model through successful evaluation;
injects nonfinite logits; verifies label-free ordered prediction shards; and
covers ECE endpoints/empty bins, AUC ties, zero-background rejection, schedule
short-run behavior, and seeded class-balanced paired bootstrap quantiles.

Block 7:

- source drift and cross-campaign parent rejection;
- all-reused restart validation;
- shell paths, account, environment, dependency ordering, and dry-run
  nonmutation;
- interrupted/reusable task and cancellation lineage.

Local Block-7 coverage includes a temporary clean Git repository, tracked
source drift after an all-reused graph, cross-campaign parent rejection,
exact numeric `sbatch --parsable` parsing, dependency lineage, insufficient
storage, task-byte corruption, dry-run nonmutation, failed-descendant recovery,
stale exact-job cancellation lineage, worker environment/static contracts,
and both final-test locks.

Block 8:

- real data and real HLT-v3;
- real GPU forward/backward;
- production-worker training, resume, inference, metrics, and report;
- no manually injected artifact;
- complete dry-run graph after the real miniature.

## Tigris without pytest

The `atlas_kd_tigris` environment may lack pytest. Maintain standalone
parity/runtime commands whose successful exit is independently useful. Local
pytest tests should invoke the same underlying validation functions rather
than contain a separate implementation.
