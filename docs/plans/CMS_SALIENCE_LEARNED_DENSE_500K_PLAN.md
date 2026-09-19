# CMS/Scouting salience learned dense ladder (science v1, execution v3)

## Authority and scope

This is a new, isolated CMS/Scouting study requested on 2026-09-15. It does
not resume, rewrite, or submit jobs in any JetClass2 or historical CMS campaign.
It transfers the Strategy-B method, not a claim that its performance is proven.
It supersedes the older three-spine/54-fit panel **for this family only**.

Run on SPORC, account `reu-aisocial`, QOS `qos_tier3`, one A100 and
`atlas_kd_sporc`. New campaign specs explicitly select `tier3` (default) or
`debug` for all stages. The latter is an operator choice subject to RC usage
policy, not a claim of permission or faster scheduling. Both routes retain the
same resource limits and maximum 24-hour training request. The genuine gate
must run on the selected partition with the registered science resources.
Do not transfer GH200, JetClass2, or another campaign's execution acceptance.

Execution-spec v2 added this choice without changing scientific v1 artifacts.
Existing v1 specs remain tier3-only and immutable. A fresh v2/v3 root may import
completed native-CMS preparation read-only from an explicitly named source
campaign. Require identical scientific graph, population, split, raw-data
root, view configuration and preparation code lineage, plus authenticated
preparation receipts and payloads. Import only selection, matching, coupling,
scales and validation partitions; never import trained models or a GPU gate.
The new preparation stage verifies those artifacts and publishes its own
import receipt; gate and science then run normally. Do not cancel, edit, or
resubmit jobs in the source campaign as a side effect.

## Population and matching

Use original native CMS Scouting ROOT `tree` data (21 features, 15 classes),
not the 17-feature/11-class Delphes adapter. Require the authenticated original
file-disjoint split manifest. Select 500,000 train and 250,000 validation jets
by the existing proportional per-class smallest-identity-hash rule, seed 1337,
within their original roles. Reserve 250,000 final-test jets using the same
rule within the original final-test role; do not materialize that selection or
read its branches in this campaign. A separately authorized finalist/execution
lock and evaluation operation are required to open final test.

Freeze `SALIENCE_PT_LINEAR` as a transferred prior chosen on JetClass2, not a
CMS-tuned winner. Recompute assignments on this CMS subset. Use the existing
exact full-cardinality salience Hungarian solver and native-index orientation;
do not import JetClass2 assignment rows. Follow the native decoder's 200 HLT
cap, regular-offline matcher population (lost tracks excluded from matching),
and projected pure-offline reference's 90 charged / 60 neutral limits.
Keep the existing CMS balanced persistent-support homotopy: HLT slots persist,
matched slots carry offline content at U100, and D000 is byte-exact native HLT.
Residual coupling scales are fitted on at most 4096 selected training jets:
equal per-file allocations, evenly spaced in canonical selected-entry order.
No validation jets contribute. Reuse the
existing endpoint partition/coupling and mass-balanced switches, including its
bounded carrier policy; never silently truncate a carrier that exceeds 200.
Identity, pairing, switch indices and labels are never model input features.

Validation is deterministically class-stratified into disjoint 50% checkpoint,
25% diagnostic and 25% reporting subsets. Only checkpoint validation selects
weights/early stopping. Report recovery on the reporting subset, not mixed
with historical full-population numbers. Use all 250k only for non-selecting
parity checks. Final test stays sealed.

## Registered graph

`U000 -> U033 -> U066 -> U100 -> D080 -> D060 -> D040 -> D020 -> D000`

U033/U066 are exact thirds. D080/D060/D040/D020 are exact fifths of offline
content. Fresh references are native-HLT CE (`M0HLT`), persistent U000 CE
(`U000`) and projected pure-offline CE (`OFFLINE`). The one global ordinary
logit-KD control is `DIRECT_D000` taught by this U000. No imported trained
anchor, no per-rung direct-control panel, no random-seed ensemble panel.

Each of the eight arrows has:

1. A cold-start acquisition: lower view is primary, previous higher view is
   context. C25P75, T=2 KD comes from the previous extracted single carrier.
2. A reducer saving selected acquisition probabilities (T=2 train, T=1 validation).
3. Withdrawal initialized from the selected acquisition weights, fresh optimizer,
   with that frozen acquisition probability bank as teacher. Validate/select
   only the alpha-zero route, including before the training gate reaches zero.
4. Exact extraction of the ordinary primary ParT. This is the sole teacher for
   the next arrow. Never carry context weights to the next cold acquisition.

There are 20 fits, 8 extractions, 16 probability reducers, aggregate and
completion: 46 science tasks. Preparation and genuine A100 acceptance are
separate stages. The primary owns the sole head. Context injects one-way,
zero-initialized gated attention residuals after blocks 2/4/6/8, with native
pair geometry. At alpha zero no context encoder or cross-attention executes.

## Frozen optimization

Batch 256, one GPU, AdamW (3e-4, betas .9/.999, epsilon 1e-8, weight decay .01),
BF16 forward and FP32 loss. Warmup passes 1–3; hold through 45; cosine decay
through 60 to 1.5e-5; constant floor through 100. Minimum 60 passes, patience
15 after pass 60, significant AUC improvement 5e-5. Select lexically by AUC,
negative CE, macro log R50, then earliest update. Restore best weights.
All reference/control fits use this same schedule. Distinct deterministic
per-rung seed aliases; acquisition and withdrawal at a rung share the alias.

Withdrawal alpha: 1 through pass 10, cosine to zero at 60, exactly zero
thereafter. Loss: .25 zero CE + .30 zero KD + .15 privileged CE + .20 privileged
KD + .05 directed logit consistency + .05 masked normalized representation
consistency at blocks 2/4/6/8. The existing zero-gate objective is reused.

## Storage, authorization, and acceptance

Immutable content/parent hashes bind source commit, native split, selection,
matcher, scales, assignments, view identities and all selected artifacts.
Compact maps/probabilities and selected weights may be durable. Dense particle
views, hidden states, optimizer and rolling best state remain process RAM.
Preprocessing uses bounded spawned processes, not a Python thread-only pool.

Require a clean exact pushed source checkout, canonical dry-run plan, completed
preparation and genuine installed-Weaver/A100 production-worker acceptance
before science submission. Acceptance builds the full worst-case paired cache,
runs real CE/KD/acquisition/withdrawal updates, proves alpha-zero/extraction
parity, and records CPU/CUDA peaks and elapsed time. No scientific accuracy
threshold gates the graph. Poor performance is a result; invalid data, stale
source, mismatched lineage and nonfinite computations fail closed.

### Execution v3: explicit GPU headroom policy

Authorized on 2026-09-18 after debug preflight 21719837 measured
37,448,891,392 bytes peak live CUDA allocation on a 42,430,300,160-byte
A100 (88.26%). All four miniature routes and exact extraction completed;
the run failed the registered 85% CUDA threshold, not a reported CUDA OOM.
This is evidence for trying a new gate, not proof that production will fit.

New v3 campaign specs freeze `acceptance_policy`: CPU RSS must remain
strictly below 85% of requested RAM; peak live CUDA allocation must remain
strictly below 90% of the measured device capacity. CUDA reservation is logged
separately because it includes the allocator cache. Do not use current live
allocation after cleanup in place of the run's high-water mark.

In each paired miniature route, run five consecutive withdrawal-objective
optimizer updates at each of alpha=1, 0.5 and 0. Use the 256 longest U000
training jets from the population-backed cache, the same optimizer throughout
the probe, and no between-step cache clearing or peak reset. Record each
step's actual batch size and CPU/CUDA high-water measurements. Then verify
byte-exact alpha-zero extraction. Require all 30 step records across acquisition
and withdrawal in the new `EXECUTION_ACCEPTANCE/v2` artifact, bound to the
campaign policy. These are execution probes, not additional scientific fits.

The scientific batch remains 256. No microbatching, gradient accumulation,
activation checkpointing, model/loss change or schedule change is introduced.
Legacy campaign v1/v2 gates retain 85% CPU and 85% GPU limits with acceptance
v1; they cannot be silently relaxed or reused as v3 acceptance. A fresh root,
exact pushed source and genuine preflight are required before any science.
Completed original preparation may still be imported read-only as above.

### CMS-only temporary cross-attention memory reduction

After source `77c9d2b1` exceeded the v3 CUDA limit at 92.57%, the native CMS
adapter may premerge context padding into the rectangular cross-attention bias
once and share that tensor across injections 2/4/6/8. The shared fusion base
keeps its legacy allocation path for other adapters. Compute the same full
Weaver pair embedding before slicing: computing only cross pairs would change
its BatchNorm population. Materialize only the resulting rectangle, so the
slice need not retain the larger square output storage. Do not detach the
bias, omit either prediction route, reorder stochastic layers, or recompute
activations. State-dictionary keys and model parameters stay unchanged.

This is an execution optimization, not a new scientific arm. Require legacy-
versus-optimized tests of outputs, all withdrawal loss terms, gradients,
optimizer updates, buffers and RNG consumption, plus installed-Weaver parity
and a fresh measured A100 gate. Floating gradient accumulation order may
differ within numerical tolerances; whole-training bitwise identity is not
claimed. Keep batch 256, the registered losses/schedule and the strict 90%
CUDA / 85% CPU limits. An allocation-level saving does not prove that the
whole training peak is low enough; no previous failed gate is reusable.

Report validation accuracy, macro AUC, geometric macro R50 and each class's
QCD rejection. Recovery = (model − native-HLT CE)/(pure-offline CE − native-HLT
CE), with R50 in linear rejection space. Also report persistent-anchor
recovery separately; undefined denominators produce null, never fabricated 0.
No local test result is genuine SPORC acceptance.
