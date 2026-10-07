# Frozen CORR_MID production and direct/coarse follow-up

## Decision and authority

On 2026-10-07 the user explicitly selected MID and requested the full 2.25M
dataset followed by a 100k-train/50k-validation SPORC direct/coarse experiment.
This plan supersedes the pilot plan's proposed CE-only strength screening.
There is no strength scan, fit to CMS, performance-based recipe selection, or
requirement that the offline/HLT accuracy gap be 1.5 points. That remains an
unmeasured hypothesis. No repeated-KD or dense branch is included in this request.

## Immutable endpoint

Freeze CORR_MID, amplitude 1, from the original correlated tracking pilot at
commit `0667355d8c4d72e48e32c03f2d1540595fe654b7`, job 228972. Its successful
20k diagnostic and serial/process replay were reported by the user. Production
authenticates the exact ledger/plan/spec, every output, and historical source.
Keep the original kernel and diagnostic recipe unchanged. A new production
recipe wrapper selects this endpoint and records the new authorization.

Read original dzfix OFFLINE with the same particle keys and canonical jet
identities used in the pilot. Call the frozen generate_all function and retain
only CORR_MID. No old NOMINAL, COUNT38, NOISE_V3 or CONTEXT transformation occurs.
Values/errors change only for eligible tracking components; p4, PID, charge,
particle order/count and validity masks remain exact. No latent shifts, labels,
RNG state, ancestry or construction keys become model inputs. One fixed draw
per jet, not per epoch. This is a synthetic per-jet reference perturbation,
not a verified CMS vertex model; the pilot's explicit convention still applies.

## Full dataset and execution

Reuse authenticated TRAIN_1M registry membership and the existing deterministic
250k validation subset: 1M train, 250k validation, 1M final test, disjoint and
unchanged. No new split or label-driven selection. Test construction is explicitly
authorized, but a TEST_BUILD_LOCK grants no test inference or distribution study.
Generation records structural counts only, including for test shards.

Create a fresh persistent sibling root under /home/ryreu/atlas/datasets, outside
all existing datasets/worktrees. Default 20 GiB output budget plus 2 GiB free
headroom, with current user quota/persistence attestation. Never delete old data.

One reviewed Tigris submission queues preflight -> generation array -> finalizer.
Preflight and generation request 36 CPUs, 64 GiB, 2h; at most 16 generation jobs
(576 CPUs), no GPU. Finalizer requests 1 CPU, 16 GiB, 2h. Account reu-aisocial.
The preflight uses the production raw reader, 36-process engine and physical
writer to reproduce 64 saved pilot training rows exactly. It records time/RAM/
storage projections and releases the array only after integrity/resource success.
This is part of production, not another scientific pilot or manual decision.
The completed pilot does not alone certify production-worker acceptance.

Bounded spawn processes (single numerical thread each), ordered chunks and
1000-jet lossless physical banks preserve identities independently of scheduling.
Validate raw source hash before opening; read only original OFFLINE branches.
Atomic outputs/receipts/manifests; quotas reserve concurrent shard output and
preserve failed attempts. Recovery requires terminal exact-ID accounting and
queues only missing shards in a new immutable attempt. No broad cancellation.
Train and validation releases are independent; full manifest requires all 2.25M.

## Requested SPORC comparison

After the dataset is finalized, use shared RIT storage (no transfer). Freeze a
label-blind 100k training / 50k validation subset before classifier results.
Keep final-test inference sealed. Use the same 17-input unclipped asinh/log1p
frontend, ParT architecture, paired seeds and CE/KD recipe for every model.

Topology is identical by construction: verify exact p4/PID/mask row alignment
before pairing. No Hungarian approximation or ascending U000/U050/U100 fits:
those views would duplicate OFFLINE here. Register OFFLINE and HLT CE controls,
OFFLINE -> HLT direct KD and OFFLINE -> D066 -> D033 -> D000 coarse KD, each
fresh initialized. Intermediate views are explicitly training-time oracle views;
their residual amplitude scales linearly and added error variance quadratically,
using the already frozen endpoint draw without resampling.
Only HLT D000 is deployable. Report accuracy, AUC, classwise rejection @50% and
recovery, including negative/poor results; do not tune the generator afterward.
CORR-only classification cannot establish superiority over matched INDEP noise
or isolate ladder benefit from additional training compute.

SPORC GPU training still requires its own source-pinned full-population preflight
and measured exact debug plan. Dataset production never silently submits GPU
training or changes existing campaigns. Implementation state is in HANDOFF;
the dataset and classifier source/receipts are distinct execution artifacts.
