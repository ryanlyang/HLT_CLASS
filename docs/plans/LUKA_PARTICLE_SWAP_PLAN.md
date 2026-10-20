# Luka whole-particle replacement D ladder

Authorized 2026-10-19: implement a separate queue-ready SPORC debug experiment.
The recovery hypothesis is that coherent endpoint particle records provide a
better intermediate teaching path than feature interpolation. No improvement
is promised, and no construction parameter is selected using validation scores.

Two matched cold ladders reuse the authenticated fixed30 regular COARSE_U100:
U100 -> SWAP066 -> SWAP033 -> SWAP000. LOGIT uses C25/P75 T2; ATTN adds the
existing constant 0.1 block7 head-mean pre-dropout attention KL. Keep ALLJETS
299234/74895, original50k report-only, batch128, exactly30 epochs, paired
coordinate seeds, optimizer and full-validation AUC-first selection. No test,
warm start, fusion, added input feature, loss search or performance gate.

Keep the original full-cardinality assignment and HLT slot order. At a swap,
replace the entire physical record: p4, charge, category, d0/dz, errors and
validity. Unassigned slots remain HLT. Recompute ALL model features from each
new physical view with the existing feature kernel; never splice transformed
columns. U100 and SWAP000 must reproduce original U100 and native D000 input
fingerprints exactly. Intermediate jets are synthetic hybrids of real records;
algorithmic assignments are not truth-level matches.

Freeze a label-free nested schedule. Divide matched slots into six strata:
charged/neutral crossed with rank-tertiles of native HLT pT. Make three
replacement batches with counts differing by at most one. Process particles
by descending quantized native HLT pT share; assign each to a nonfull batch by
the least increase in the sum of squared normalized count, pT and within-stratum
count loads. Equal weights; scale1000000; nearest-integer quantization. Stable
hash ties bind only the view contract, jet identity, slot and batch. The hash
also distributes remainder capacities and rotates tied batch priority per jet.
The first batch swaps at066, second at033, third at000. No particle switches
back. Do not keep all high-pT particles until the last jump. Counts are balanced
exactly up to integer rounding, pT/type balance is approximate; record achieved
balance without using it as a scientific performance veto.

Build all four views once per role in process RAM from authenticated selected
ROOT ranges and saved assignments. No matching rerun or persistent particle
cache. Bind new input identity to old prepared/cache hashes plus the new view
contract. Internal D coordinate keys used by existing banks are scoped by this
new identity; old interpolated caches cannot masquerade as swap views.
Full-role fingerprints and schedule hashes measured in preflight must replay in
science. Resident authenticated FP32 targets eliminate per-minibatch file I/O;
fail on an insufficient declared RAM budget, with no unmeasured fallback.

One preflight then one sequential science job PER arm, all on debug/A100,
6CPU/128000MiB. Same coefficient/path in probe and science. Gate8h; measured
whole-ladder science request capped24h. Gate runs native installed-Weaver
FP32/BF16 parity, longest-support TRAIN backward and validation inference
stress, all three full-population one-pass probes, selected-checkpoint/native
readback, exact endpoint/view replay, targets and safe retirement. Poor metrics
never block a registered rung. Exact pushed clean source, successful matching
Slurm journal/accounting, accepted resources and explicit dry-plan authorization
are prerequisites for science; no implicit submission or retry.

Storage policy applies only to newly owned scratch. Keep one rolling teacher
bank, retire it after durable validated child outputs; retire a child checkpoint
only after its next export validates (D000 after selected predictions validate).
Keep per-jet predictions, IDs, labels, original50k mask, histories, compact
audits/manifests and retirement provenance. No raw data, old checkpoint/cache,
shared U100, other campaign or Fresh_check deletion. New inference after
checkpoint retirement requires retraining. Each arm gets its own local scratch
and at least1GiB persistent allowance. No home/shared-scratch fallback.

Import original LOGIT/ATTN interpolation results with the existing explicit
comparison-only archive proof. Compare all final endpoints against M0HLT,
DIRECT_D000 and both interpolation endpoints; also report all new rungs.
No independent test exists, and single-seed development comparisons do not
establish statistical significance or out-of-sample method selection.

Implementation: new `luka_particle_swap` namespace only. Reuse unchanged native
model/loss/bank math, exact scheduler journal and selected-row readers; adapt
weight-sweep lifecycle under distinct contracts. Record donor commit/files in
LEGACY_SOURCE_MAP. Test schedules, atomic records, input recomputation, real
ROOT-fixture rebuilds, endpoints, corrupt/alignment rejection, fixed30 chains,
retirement, resident-bank parity, resource admission and dry/live queue guards.
