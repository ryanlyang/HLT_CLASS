# HCWDL tagged offline+HLT concatenation pilot contracts

## Scope

This document is the reusable-contract companion to Study A of
[`HCWDL_OFFLINE_HLT_CONCATENATION_FUSION_WITHDRAWAL_IMPLEMENTATION_PLAN.md`](../plans/HCWDL_OFFLINE_HLT_CONCATENATION_FUSION_WITHDRAWAL_IMPLEMENTATION_PLAN.md).
It registers one isolated validation-only feasibility fit named
`CONCAT_TAGGED`. It does not implement fusion or privilege withdrawal and it
does not create a deployable model.

## Scientific identity

For every authenticated train or validation jet, the model input is exactly
all native-order offline particles followed by all native-order canonical HLT
particles. Matched reconstructions are deliberately retained twice. Matching,
assignment, construction, file-order, and degradation indices are not model
inputs. No token may be truncated: the all-row audit must prove that the
fixed 400-slot capacity covers every selected row before training.

The numerical particle representation remains the established 21-channel
HLT-compatible schema. A separate two-value learned content-source embedding
identifies offline and HLT tokens after numerical embedding and before the
Particle Transformer blocks. Padding has source code `-1`, is never embedded,
and remains masked. The source code is not a physics feature.

The pilot is CE-only, uses the source campaign's matched `U000` initialization
domain, 60 passes, global batch 256, the established warmup/cosine schedule,
one GH200, best-validation checkpoint restoration, and no early stopping,
rolling resume, partial checkpoint reuse, KD, representation target, or final
test access. Particle views live only in process memory; durable output is
limited to locks, reports, task evidence, and selected/final model states.

## Artifact family

`hcwdl_offline_hlt_concat_contracts.py` exports the versioned
`HCWDL_OFFLINE_HLT_TAGGED_CONCAT_PILOT_*/v1` graph, recipe, source-lock,
capacity-audit, execution-acceptance, campaign-spec, command-plan, node,
training-report, selected-checkpoint, final-checkpoint, aggregate,
campaign-complete, monitor, and recovery contracts. Every artifact has a
content hash and exact parent lineage and is atomically published.

The aggregate reports `M0CE60`, pure-offline `U000`, persistent-support
`SP4P_U000`, and `CONCAT_TAGGED`. Recovery percentages use `M0CE60 = 0%` and
pure-offline `U000 = 100%`. Scientific metric values never control task
success or graph continuation.

## Execution gates and isolation

The exact DAG is:

```text
authenticate -> capacity_audit -> preflight -> train_CONCAT_TAGGED
             -> aggregate -> campaign_complete
```

Submission is deliberately split. The gate ledger contains the first three
tasks. Only after their real Tigris artifacts and the preflight task
attestation validate may the separate science ledger submit the final three
tasks. The immutable six-task plan records the complete dependency graph;
stage plans are exact projections of it, and the science projection records
`preflight` as an already satisfied prerequisite.

The capacity audit scans every selected train and validation row and fails on
any hidden truncation. The genuine-Tigris preflight exercises installed Weaver
with a complete real 256-row production batch containing the all-row audit's
maximum-length identity, the full 400-slot tensor shape, BF16 forward and
backward, and a finite nonzero source-embedding gradient. The science fit is
dependency-gated on that acceptance. Final test is inaccessible.

The campaign has its own root, source-pinned worktree, `hcwcat1_` job names,
ledger, output inventories, exact-ID monitor, and restart-from-zero recovery.
It has no scheduler dependency on, write path into, or authority to cancel,
hold, or reprioritize an existing campaign.
