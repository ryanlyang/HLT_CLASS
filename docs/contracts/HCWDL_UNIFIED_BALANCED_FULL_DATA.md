# HCWDL Unified-Balanced Full-Data Contract

Version: **HCWDL-UB-FULL3/v1**.

This contract is the reusable execution boundary for the all-mapped,
three-arm factorized scale-up authorized by
[`HCWDL_UNIFIED_BALANCED_FULL_DATA_THREE_ARM_PLAN.md`](../plans/HCWDL_UNIFIED_BALANCED_FULL_DATA_THREE_ARM_PLAN.md).
It is additive: it does not change any 300k HCWDL-UB artifact or claim.

## Population and access

The foundation includes every authenticated mapped train and validation row
from its bound split manifest. Exact integer counts are stored in the spec and
must equal the split inventory. The final-test count is recorded, but ordinary
access is exactly zero. A final-test path, assignment, coupling, cache, or
metric is forbidden in this contract family.

## Scientific registry

The registry contains exactly 38 fresh fits:

- shared CE roots `U000` and `M0paired`;
- for each of `C25P75`, `C10P90`, and `C10P75G15`, eleven factorized path
  fits `U020` through `M1F` and one `D100direct` control.

There are no joint nodes. The only deployable path endpoints are exact-HLT
`D0F` and `M1F`; `M1F` is the designated arm result. Every fit uses 20
natural-population passes, validation every pass, unweighted CE, and the
locked macro-AUC-first checkpoint policy. Homotopy KD uses temperature 2;
`M1F` uses 0.25 CE plus 0.75 parent KD at temperature 1.

## Artifact closure

The immutable foundation closure is:

```text
source-pinned template + split
  -> all-row selection + v5 recipe
  -> train/validation assignment manifests + audits + full assignment lock
  -> scale calibration + residual bases + legacy switch manifests
  -> exhaustive coupling audit + coupling lock
  -> balanced sidecars/manifests
  -> endpoint/resource lock
  -> U000 + M0paired + U000 target manifest
  -> foundation lock
```

Arm specs may be created only from that foundation lock. Each arm has an
independent root, command plan, ledger, monitor, exact-ID cancellation surface,
and failed/downstream recovery closure. Durable reconstructed particle views
are forbidden; compact U000 FP32 logits are the only shared durable targets.

## Contract identities

The family uses the versioned identities declared in
`hcwdl_unified_balanced_full_contracts.py`, including foundation/arm specs and
command plans, graph, recipe overlay and arm recipes, assignment/endpoint/
foundation locks, reports, aggregate/completion, monitor, automatic arm-launch
receipt/event, campaign submission, and recovery.
`HCWDL_RECIPE/v5` is the only recipe contract authorized for these 20-pass
fits. Existing HIGHCOV assignment, residual-coupling, balanced-sidecar, target,
PMARD resume, task-attestation, and submission-ledger contracts are reused
only after their exact parent hashes validate.

Changing the row population, graph, arm weights, rung coordinates, pass count,
checkpoint policy, endpoint construction, target routing, seeds, or final-test
boundary is a scientific version change. Resource recovery may only increase
CPU, RAM, or walltime while preserving GPU class and all scientific hashes.
