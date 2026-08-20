# HCWDL-MHPE three-track 60-pass full-data contracts

## Scope

This document is the reusable contract companion to
[`HCWDL_MHPE_THREE_TRACK_60E_FULL_IMPLEMENTATION_PLAN.md`](../plans/HCWDL_MHPE_THREE_TRACK_60E_FULL_IMPLEMENTATION_PLAN.md).
The plan remains the scientific authority. This contract family registers one
validation-only all-mapped campaign and must not broaden or relabel any
completed HCWDL-UJ, HCWDL-UB, MHPE, or representation-KD artifact.

The ordinary campaign can read only authenticated train and validation rows.
It has no final-test task or final-test capability. Every deployable endpoint
uses the ordinary unified 21-channel, 200-token HLT Particle Transformer.

## Scientific identities

The new `/v1` family contains graph, node, recipe, integration, source-test,
foundation, endpoint/resource, probability-shard, probability-manifest,
probability-lock, ephemeral-representation-audit, training-report,
selected-checkpoint, final-checkpoint, deployable-checkpoint, stage-report,
campaign-specification, command-plan, runtime-profile, aggregate,
finalist-lock, execution-lock, campaign-completion, recovery, resource-
recovery, monitor, cancellation, and interruption-attestation contracts. The
exact strings are exported by `hcwdl_mhpe_tri60_contracts.py`; duplicates or
unversioned identities are forbidden.

The graph contains exactly 32 fresh 60-pass fits:

- one shared CE-only `U000`;
- 15 C25P75/T2 LOGIT specialists and `M1_LOGIT`;
- six C25P75/T2 LOGIT+RSET specialists and `M1_RSET`;
- six C25P75/T2 LOGIT+RREL specialists and `M1_RREL`; and
- one C10P90/T1 exact-HLT `M2` trained from `M1E`.

It has exactly 12 uniform ensemble reducers plus the one-member durable U000
probability publication. All model fits validate once per natural-population
pass and use macro-AUC, CE, log-R50, and earliest update in that order for
checkpoint selection. Finite poor performance never changes graph execution.

## Teacher and carrier separation

A representation specialist authenticates two independent parents joined by
the same 32-byte jet identity:

1. its declared uniform probability distribution supplies classification KD;
2. one fixed single-model carrier supplies RSET or RREL targets on the
   carrier's own input view.

The carrier is never selected by validation score and is never an averaged
representation. The complete carrier table is part of the graph hash. RSET
uses jet plus unordered token-set supervision. RREL additionally uses the
corrected raw block-2 relation state. Both use the exact v5 schedules,
calibration, kernels, masks, and weak-support continuation.

## Persistence boundary

Durable data may include only compact FP32 class probabilities with uint8
identity digests, selected and final model envelopes, small calibration and
lineage JSON, reports, locks, ledgers, and task attestations. Reconstructed
particle views, token states, jet states, set/relation sketches, and
representation targets are forbidden on persistent storage.

Each RSET/RREL worker streams its carrier once, constructs representation
targets in host RAM, releases the carrier, builds student train/validation
views in RAM, replays them for 60 passes, and releases all ephemeral arrays on
exit. The only durable representation artifact is a non-reconstructive audit
with logical hashes, shapes, dtypes, row coverage, carrier lineage, peak
memory, and `durable_payload: false`.

Rolling optimizer/RNG/model resume generations are forbidden. A successful
fit atomically publishes only selected and final checkpoints. An interrupted
fit publishes a small interruption attestation and recovery restarts the fit
from update zero with the same scientific seeds.

## Execution and recovery

The canonical source-pinned campaign spec owns one 50-task topological command
plan. LOGIT, RSET, and RREL branches may run concurrently after the shared
U000 publication. GPU resource classes are fixed at 256 GiB/three days for
LOGIT and 384 GiB/six days for RSET/RREL; reducers use 192 GiB/one day. A
bounded real GH200 production worker must authorize the RAM projection below
75 percent of the request before campaign creation.

Live submission requires the exact canonical dry-run ledger and a durable
per-job submission journal. Monitoring queries only exact ledger job IDs and
validates task attestations. Cancellation binds a monitor and records every
selected exact ID's pending, running, terminal, or unknown state. Recovery
binds the complete subject ledger/monitor and retries only the exact failed-
downstream closure. Completed artifacts remain immutable; incomplete fit
namespaces are removed only immediately before their source-pinned restart.

Resource recovery may increase CPU, RAM, or walltime while retaining the GPU
class. Source repair is separately phrase-authorized and restricted to an
exact reviewed execution-file allowlist. Neither path may alter graph, views,
losses, rows, passes, batch size, seeds, probability banks, or representation
semantics.

## Launch and claims

Local tests and an all-data dry run are nonmutating. Live Slurm submission,
cancellation, final-test access, and Git push remain separately authorized
operations. The bounded production acceptance is an operational worker test,
not a reduced scientific campaign.

The validation aggregate may compare all specialists, ensembles, three M1
models, M1E, and M2. It may not claim that an ensemble has an averaged latent
representation, that the carrier is uniquely correct, that missing offline
information was reconstructed, or that one exploratory seed establishes a
general performance result.
