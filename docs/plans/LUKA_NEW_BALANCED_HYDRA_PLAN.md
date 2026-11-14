# New FullSim balanced six-model Hydra campaign

Scientific authority, version 1, 2026-11-13. This campaign adapts the
[Oscar six-model design](LUKA_FULLSTAT_BALANCED_LADDER_PLAN.md) to the new
QCD Run-2 combined and Higgs2p samples. It does not modify the running natural
OFFLINE baseline or the registered Oscar campaign.

## Population and evaluation

The authenticated parent foundation is
`2f05adcca889288e705a69fc1e54670f1713d29ec16ab75c4f694ab7b5a23d02`.
Preserve every existing TRAIN, validation and test assignment. Extend only from
previously unused rows with `200 <= offline jet_pt < 2000 GeV`:

| Role | Parent in range | Add from unused | New in-range count |
| --- | ---: | ---: | ---: |
| TRAIN | 2,146,372 | 279,164 | 2,425,536 |
| Validation | 363,024 | 136,976 | 500,000 |
| Sealed final test | 1,086,639 | 113,361 | 1,200,000 |

There are 689 unused in-range rows left. Preserve outside-range parent TRAIN
(853,628, excluded from optimization), validation (136,976, diagnostic), test
(413,361, sealed diagnostic), and unused (205,760). Unknown raw label 206 remains
quarantined. No parent artifacts change. Row disjointness is enforced; event
independence and overlap with older datasets remain unverified.

Unused additions use sequential proportional largest-remainder quotas over
source stratum, class and 100 GeV bin, with deterministic SHA256 block ordering
and seed 2026111305. This preserves approximate natural proportions within the
eligible pool; the training sampler, not split construction, balances draws.
Freeze/replay every membership and audit all 198 new TRAIN class/bin cells.

**Select checkpoints on the new 500,000 in-range validation jets only.**
Outside-range validation and each pT-bin are diagnostics of that same selected
checkpoint. Validation is evaluated without replacement or balancing weights.
Final test, both primary and outside range, stays sealed until finalist and
execution locks; this implementation exposes no test-evaluation command.
This policy explicitly differs from the original Oscar full-validation policy.

## Six fresh fits

| Model | Input | Embedding and separate auxiliary-logit teachers |
| --- | --- | --- |
| OFFLINE | Pure offline | None, CE only |
| M0HLT | Native HLT | None, CE only |
| CONCAT | Full source-blind HLT + offline multiset | None, CE only |
| UE100 | HLT shell with matched offline measurements | OFFLINE, CONCAT, M0HLT |
| DE050 | Exact one-half offline contribution | OFFLINE, CONCAT, UE100, M0HLT |
| DE000 | Native HLT | OFFLINE, CONCAT, UE100, DE050 |

Each model starts fresh. All teachers are selected within this new balanced
campaign; no historical baseline is substituted. M0HLT and DE000 deploy as
unchanged HLT-only native ParT. Auxiliary predictions never enter their classifier.

Retain the Oscar objective exactly: CE 1, total embedding 10 split equally among
teachers, total auxiliary-logit KD 1 split equally, T = 2 with T-squared forward KL.
Native 128 embeddings have separate 128-128-128 GELU heads and unique-TRAIN
FP64 mean/population-std standardization (floor 0.001). Native T = 2 probabilities
use BF16 forwards and FP32 softmax; embeddings use FP32 native forwards.
DE000 alone adds pure-OFFLINE contrastive 0.3, ramped over three draw epochs,
tau 0.1, with 256 hard / 256 same-class / 512 global distinct negatives. Keep the
Oscar identity/epoch/occurrence RNG and negative distribution unchanged.
No subclass, FUSION, extra rung, weight sweep or warm start is introduced.

All six use uniform class, uniform conditional 100 GeV bin, uniform row in the
cell with replacement: 18 bins, 11 classes, 2,425,536 draws per epoch, 20 epochs,
batch 128, last batch 64, 18,950 updates/epoch. Use the identical PCG64 epoch draw
stream across fits. No additional class/pT loss weighting. Preserve native
initialization, auxiliary seeds, AdamW and two-epoch warmup/cosine 20 schedule.

New source particle pairing is constructed by the existing full-cardinality
matcher on authenticated TRAIN/validation rows; deterministic mapping hashes
are replayed between UE100 and DE050. Native HLT inputs never carry mapping,
source, role, or pT-selection metadata. CONCAT has no source tags and retains
all particles; per-jet capacity is chosen from full support, without truncation.

## Execution, storage and recovery

Use Brown CS Hydra, one explicitly selected BF16-capable L40S (L40 or A6000
available as explicit alternatives), six CPUs and a measured memory/walltime
envelope. The first three fits can run independently; later rungs require all
their teachers. No separate GPU preflight allocation. Each science job performs
installed-Weaver parity, connected-gradient, full-support and timing checks,
then discards the probe and begins its fresh fit or authenticated continuation.

Construct particle inputs in RAM in bounded workers inside each allocation;
do not create six persistent prepared caches. Retain selected native models,
all epoch optimizer/RNG checkpoints, five unique-TRAIN teacher banks and
selected validation predictions. Declare a 14 GiB new-results budget and reserve
the natural baseline's remaining footprint up to 17 GiB separately. Recheck
available filesystem space; quota remains an explicit operator declaration.
Preserve the original raw dataset, natural baseline cache and all other jobs.

An explicit resume checks exact prior jobs are terminal and reuses only
authenticated completed stages or contiguous committed epoch states. A completed
20-epoch fit needing export/publication performs no additional updates.
An ambiguous submission stops automatic retry. Partial files are preserved.

Report AUC, accuracy, CE, R50 and R30 with signal/background passing counts,
zero-background censoring, per-bin and outside-range diagnostics. Recovery uses
new balanced M0HLT/OFFLINE on identical populations. Historical natural OFFLINE
is a complete-recipe comparison, not an isolated flattening ablation, because
its in-range TRAIN size and checkpoint-selection policy differ.
