# HCWDL Direct Offline-to-HLT KD Contract

Version: `v1`

This validation-only campaign isolates the value of direct supervision from a
fresh native-offline Particle Transformer into an exact-HLT student. It does
not use the structural/feature homotopy ladder and it never opens final-test
data.

## Registered fits

All fits use the authenticated unweighted 300k-train/100k-validation HCWDL
population, 60 passes, validation after every pass, and macro-AUC-first
checkpoint selection.

| Node | Input | Objective | Deployable |
|---|---|---|---|
| `HLT_CE` | exact HLT | CE | yes |
| `TOFF_CE` | native offline 19/7 two-stream | CE | no |
| `HLT_LOGIT` | exact HLT | 0.25 CE + 0.75 TOFF logit KD, T=2 | yes |
| `HLT_RSET` | exact HLT | paired logit objective + RSET, rho=0.10 | yes |
| `HLT_RREL` | exact HLT | paired logit objective + RREL, rho=0.10 | yes |

`HLT_CE`, `HLT_LOGIT`, `HLT_RSET`, and `HLT_RREL` are cold-started with the
same backbone initialization, train sampler, validation order, dropout,
augmentation, optimizer schedule, and seed. Representation projection heads
use strategy-specific seeds because those parameters do not exist in the CE
and logit-only controls. `TOFF_CE` is a separate fresh teacher root.

The campaign generates the TOFF logits and representation summaries exactly
once over the selected training population after selecting `TOFF_CE`. One
compact authenticated target bank is read by all three distilled students.
No validation targets, repaired particle datasets, native-offline tensors, or
final-test rows are persisted in that bank. Compact target arrays are removed
only after all three consumers publish authenticated reports; their manifests
and cleanup proof remain.

## Execution graph

```text
authenticate
  +-- HLT_CE
  `-- TOFF_CE -> TOFF target bank
                    +-- HLT_LOGIT
                    +-- HLT_RSET
                    `-- HLT_RREL

all five reports -> aggregate
all three consumers -> target cleanup
aggregate + cleanup -> campaign completion
```

Finite but poor scientific outcomes never block a registered descendant. A
nonfinite value, identity mismatch, forbidden role access, source drift,
artifact corruption, or lineage mismatch fails closed.

Every GPU task requests eight CPUs, 96 GiB RAM, six hours, and one GH200 on
the `reu-aisocial`/`tigris` account and partition. Workers use an absolute
source-pinned `PROJECT_DIR`, `python -s`, `PYTHONNOUSERSITE=1`, the active
Conda library path, and `USR1` checkpoint signaling. Submission is journaled;
monitoring, cancellation, and recovery operate only on exact campaign-bound
job IDs. Recovery preserves the exact source/spec/data/recipe/targets/seeds
and resumes rolling checkpoints.

Primary comparisons are paired `HLT_LOGIT - HLT_CE`, `HLT_RSET - HLT_LOGIT`,
and `HLT_RREL - HLT_LOGIT`. `TOFF_CE` is an oracle teacher reference, not a
deployable model. Validation findings are exploratory; sealed final-test
evaluation requires a separate future authority.

## Versioned identities

- `HCWDL_DIRECT_OFFLINE_KD_GRAPH/v1`
- `HCWDL_DIRECT_OFFLINE_KD_NODE/v1`
- `HCWDL_DIRECT_OFFLINE_KD_CAMPAIGN_SPEC/v1`
- `HCWDL_DIRECT_OFFLINE_KD_COMMAND_PLAN/v1`
- `HCWDL_DIRECT_OFFLINE_KD_RECIPE/v1`
- `HCWDL_DIRECT_OFFLINE_KD_TARGET_{SPEC,SHARD,GENERATION,MANIFEST,CLEANUP_AUTHORIZATION,CLEANUP}/v1`
- `HCWDL_DIRECT_OFFLINE_KD_{BASE_REPORT,REPRESENTATION_REPORT,AGGREGATE,COMPLETION}/v1`
- `HCWDL_DIRECT_OFFLINE_KD_{RUNTIME,AUTHENTICATION}/v1`
