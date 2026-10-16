# Luka FullStat: Oscar regular DIRECT / COARSE, 20 epochs

Authorized 2026-10-16. This is a new dataset and campaign, not a recovery of
the 630k trial or its fusion experiments. This plan supersedes the old trial's
source paths, file split, additional pT cut and 60--100-pass recipe only for
the new `luka_fullstat` namespace. Historical artifacts remain unchanged.

## Population and limits of independence

Authenticate the downloaded CERN snapshot `output_20261015_fullsim_fullstat/merged`
against `source.txt` and `source.sha256`: 16 ROOT files, 7036584 raw rows,
5105325 eligible matched nonempty pairs. Signal comes from
`jetclass2/train_higgs2p/merged`; QCD comes only from `jetclass2sb/QCD_PT-*/merged`.
Discard QCD-labelled signal-file rows. Validate labels/schema/counts and finite
offline pT, but apply NO extra pT cut (the producer says it is already applied).

Freeze 3000000 train, 500000 validation, 1500000 sealed final-test and 105325
unused rows. Preserve natural eleven-class proportions and QCD source-bin
mixture using deterministic integer quotas. All coordinates/models use identical
row membership and each offline/HLT pair stays together. Never rebalance classes.

Merged files lack event identifiers. A split is ROW-disjoint, NOT proven
event-independent; overlap with the old trial is also unverified. Creating a
production split requires explicit operator acknowledgement of this limitation.
For bounded selected-only ROOT I/O, order 2048-entry source blocks by a fixed
seeded hash shared across strata, then apportion each stratum's rows in that
order to exact role quotas (boundary blocks may contain multiple roles).
This is not an event-group split or an iid per-row shuffle. Training independently
shuffles all selected rows each epoch. Split seed and algorithm are immutable.
No final-test or unused particle arrays may be requested. Pre-split scalar
metadata scanning and opaque-byte source hashing are permitted and recorded.

## Nine fits and five TRAIN-only banks

- M0HLT: native D000, CE only.
- OFFLINE: native offline oracle, CE only.
- U000: initial CE teacher.
- DIRECT_D000: U000 -> D000.
- COARSE_U050 -> COARSE_U100 -> COARSE_D066 -> COARSE_D033 -> COARSE_D000,
  starting from U000 and supervised by the immediately preceding selected model.

All fits start from scratch. No fusion, dense ladder, imported checkpoints,
warm starts or scientific use of technical probe teachers. The inherited
salience/persistent-shell view and 17-feature input kernels are unchanged;
512 capacity, no truncation. Same provisional mm/error-only zero-error masking
as the trial, explicitly operator assumptions, not producer confirmation.
Source identities and assignments never enter model inputs. Intermediate views
contain offline information and are oracles; compare the three single-HLT endpoints.

Exactly 20 complete epochs per fit, validation every epoch, no early stopping
or automatic extension. Warm up linearly over epochs 0--2 to 3e-4, then cosine
decay to 1.5e-5 at epoch20, evaluated at optimizer-update endpoints. AdamW,
betas(.9,.999), eps1e-8, decay.01, BF16 forward/FP32 loss. KD is .25 CE + .75
forward KL at T2, with one T-squared factor. CE controls remain CE-only.
The initial gate registers true batch128 (accumulation1, inference128); batch256
is an explicit alternative requiring a new spec/gate, never automatic fallback.
All compared models use the same batch and recipe. Preserve coordinate-paired
initialization/sampler seeds. Select by best full-validation macro AUC, ties by
CE/logR50/earliest update. Keep validation predictions and all-epoch metrics.
Single seed, unequal total compute across routes, no guaranteed ladder advantage.

## Execution and data preparation

Oscar only: existing `oscar_l40s` site, gpu/l40s, default/norm-gpu, six CPUs,
180000 MiB. CPU work uses Oscar batch/default (live scheduler shape validation
required). Raw snapshot and receipts/checkpoints are persistent; bulky ragged
prepared caches live in an explicitly supplied scratch location (not a durable
archive; no automatic deletion). Operator must confirm actual available quota;
filesystem-wide free space alone is insufficient.

Gate: authenticate/split, bounded file-parallel CPU preparation, cache assembly,
then one genuine installed-Weaver GPU miniature/resource gate. Preparation uses
bounded entry chunks instead of accumulating million-row merged files in RAM.
Persist authenticated ragged blocks once, ordered identities/labels, and maps;
GPU jobs never rebuild ROOT views. Completion receipts last, failed partial
directories preserved. No source data mutation, automatic retry or cleanup.

The GPU gate checks installed-Weaver parity, each coordinate's real longest
TRAIN backward and separate longest validation inference at the registered
batch, and representative fixed TRAIN/validation miniature passes with selected
checkpoint/TRAIN-bank readback. It scans/authenticates all cache blocks, but
does not train full 3M technical epochs. Record sampled population, timing,
headroom, actual allocation/environment, and conservative scaled 20-epoch
runtime including full cache I/O. Estimates are not measured full-fit runtimes.
Use 2x timing margin, 85% host/90% GPU headroom and an explicitly registered
48h maximum request. Failure preserves diagnostics; no scientific metric gates
execution. This Oscar acceptance is required; local tests do not substitute.

Science has 16 tasks (9 fits,5 reducers,aggregate,complete). Clean pushed source,
successful exact gate accounting, authenticated outputs, full dry review and
explicit exact-plan authorization are required. No jobs are launched by local
implementation. No implicit science follow-up. Final-test inference is NOT
implemented by this campaign and requires a separately authorized locked workflow.

## Required verification

Synthetic ROOT selection, quota/replay/identity/disjointness tests; no test
particle reads; serial/parallel chunk parity; cache corruption and role guards;
20-epoch schedule/gradient execution; unchanged historical recipe; all nine fits
and bank dependency checks; exact dry/live submission and interruption refusal;
real local-data TRAIN-only witness; installed-Weaver parity; Oscar gate pending.
