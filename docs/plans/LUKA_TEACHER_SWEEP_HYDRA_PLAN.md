# Twelve OFFLINE teachers and twelve HLT students on Hydra

Status: implemented and locally validated; ready to queue the real Hydra setup
after the pinned source and native-input transfer are available. This is the
active scientific plan, separate from the two-fit SPORC SupCon5 campaign. User approved
12 teachers + 12 students, including strong objectives and automatic release.

## Comparison

Each fresh OFFLINE teacher uses the original native 8-block, 128-dimensional
Particle Transformer, batch 128, original AdamW and 30-epoch learning-rate
schedule. All teachers share initialization and sampler seeds (the historical
OFFLINE seeds + 200000). Training heads are removable. Export the selected
teacher's **native pooled 128D preclassifier embedding**, never its projection.
Select by full OFFLINE validation AUC using the original tie rules. Selection is
not a quality gate. Poor teachers still export targets and release students.

Every fresh HLT student uses the original HLT architecture and same-seed U100-w5
initialization/sampler, with two separate prediction heads and loss
`CE + 5*SmoothL1(U100) + 5*SmoothL1(new teacher)`. SmoothL1 uses beta 1 and a mean
over rows and 128 channels. Both target banks are TRAIN-only; new targets use
TRAIN FP64 mean/population standard deviation with a 0.001 floor. No logits,
OFFLINE particles, teacher heads, identities or labels enter deployed inference.
The existing authenticated Hydra C0 U100-w5 control is reused, never retrained.
T00/S00 is the matched extra-teacher control, not a second U100 baseline.

## Fixed grid

| Teacher/student | OFFLINE training objective |
|---|---|
| T00/S00 | CE only |
| T01/S01 | CE + 5 SupCon; exact SPORC SupCon5 projection recipe |
| T02/S02 | CE + 20 SupCon |
| T03/S03 | CE + 5 SupCon directly on native h |
| T04/S04 | CE + 5 SupCon + 5 instance contrastive, separate projections |
| T05/S05 | CE + VICReg on native h: 25 invariance + 25 variance + 1 covariance |
| T06/S06 | CE + 5 Barlow loss on native h |
| T07/S07 | CE + 5 projected SupCon + 25 variance + 1 covariance on native h |
| T08/S08 | CE(clean) + 5 masked OFFLINE input-set reconstruction through pooled h |
| T09/S09 | CE + 5 physical-summary prediction through pooled h |
| T10/S10 | 0.5 CE(clean) + 0.5 CE(thinned) + 5 h-consistency |
| T11/S11 | CE with 0.5 per-channel standard-deviation noise on pooled h |

SupCon is temperature 0.1, same-class positives excluding self, mean positive
log-probability then mean over eligible anchors. Zero-positive batches produce
differentiable zero. Projectors are 128-GELU-128; T04 instance projection is
separate and uses symmetric 2B NT-Xent at 0.1 (paired views are positives).
Native h remains the export in every case. Labels only supervise losses.

For two-view objectives, independently remove 30% of valid tokens with local
deterministic RNG, retaining at least one per jet. T08 removes 40%, with at
least one visible and one target token when multiplicity permits. Removed
features/vectors are zero and masked. This is token masking of the authenticated
native representation: remaining features retain their original full-jet
normalization, not a claim to simulate another detector. No row identifiers
or original token indices are supplied to the network. Partial singleton
batches use differentiable-zero covariance/contrastive terms where undefined.

VICReg invariance is mean squared error of raw h. Variance is mean
`relu(1-sqrt(population_variance+1e-4))`; covariance is the sum of squared
off-diagonal sample-covariance entries divided by 128. Average each regularizer
over the two views. Barlow uses per-channel batch standardization with epsilon
1e-4; loss is the sum of squared diagonal errors plus 0.0051 times the sum of
squared off-diagonal correlations. T10 consistency is mean squared error of
L2-normalized h, multiplied by 128, with gradients to both branches. T11 noise
is Gaussian with detached batch population std (floor 0.001), active only
during training; validation and target export are clean.

T08's decoder sees only masked-view pooled h plus learned output queries. It
predicts the removed **native OFFLINE particle inputs**, with one query per
maximum TRAIN multiplicity, no truncation. Targets are 17 native features and
four Cartesian p4 components, standardized with TRAIN particle statistics.
Hungarian assignment uses mean SmoothL1 over 21 channels. Regression averages
over matched particles and then jets. Add balanced presence BCE and count CE;
the sum has weight 5. Unordered matching is between predicted and target
OFFLINE particles, not HLT/OFFLINE truth matching. A small query MLP decoder
(128-query + 128-h, two 256-wide GELU layers) deliberately forces all jet
information through pooled h; it cannot read visible tokens directly.

T09 predicts 36 summaries computed from OFFLINE native inputs/p4: log
multiplicity, log jet pT/energy/mass, absolute jet eta, pT dispersion, four pT-weighted radial
moments, five category number fractions, five category pT fractions, charge
number fraction and three pT-weighted charge moments, four tracking-feature
absolute means and four pT-weighted absolute means, and four leading-pT
fractions. The charge moments are pT-weighted q, q*r and q*r^2; radial moments
use powers 0.5,1,2,3. Each is TRAIN-standardized; SmoothL1 mean over jets/channels has
weight 5. These are measured-view summaries, not truth labels or new matching.
Tracking summaries use the four existing transformed native tracking channels,
including their existing zero convention for unavailable measurements.

Use isolated seeds for auxiliary heads and augmentation, preserving native
initial weights and data order. Log weighted encoder gradient norms and their
cosines with CE at epochs 1,3,10,20,30; record finite embedding diagnostics.
Large coefficients do not guarantee large gradients. No adaptive reweighting.

## Inputs, execution and safeguards

Reuse existing authenticated Group A bundle, U100 targets, C0 and software.
Export only native OFFLINE TRAIN/validation cache shards on SPORC, replaying
missing cache files against historical byte hashes. Transfer includes a sealed
manifest and original-source proof. Hydra compares every shard against the
already transferred original cache manifest and prepared fingerprint. This
small native-cache delta is necessary because Group A raw particles are
TRAIN-only and not the exact model-ready native OFFLINE caches. No ROOT dataset,
final-test data or validation target bank is transferred for student training.

Fresh pinned worktree and separate output directory. One A6000 (gpu2201 class),
6 CPUs, 32000 MiB per GPU job; compute/default/normal CPU release/report with
1 requested CPU and 4000 MiB (Hydra's 2-CPU allocation rounding accepted).
Setup cap 12h; measured fit cap 48h. Never loosen these after a failed probe.
Reserve at least 8 GiB outputs plus 1 GiB headroom after transfer/environment.

Setup authenticates inputs/environment, checks native parity and joins, runs
real two-epoch miniatures plus warmed representative and largest-support
backward updates for all 12 teachers and the common student path, and validates
native checkpoint reload and TRAIN target export. Setup's future-student target
is explicitly technical; each student also runs a disposable miniature using
its actual selected teacher bank before the scientific fit. Resource estimates
include validation, probes, target export, I/O overhead and a 2x safety margin.
Every diagnostic is saved even on failure. Setup tests technical validity,
never performance, and no miniature weights initialize science.

Bootstrap dry review includes setup, durable Slurm CPU release afterok:setup,
and the full 24-fit/report upper-bound plan. The user's single approval covers
automatic measured release. After setup passes, release authenticates its exact
job/receipt and submits a measured dry run then science. Review and exact-ID
append-only journals prevent duplicate/ambiguous retries. Logout is safe after
both bootstrap IDs are recorded. Failure does not silently resubmit.

Submit pairs in order 01,03,00,04,05,06,07,08,09,10,11,02. Each teacher is
independent; each student depends only on its own teacher. Report uses afterany
all 24 jobs and tolerates scientifically weak or missing rows while rejecting
corrupt completed artifacts. No chain between different recipes or artificial
concurrency cap. Slurm ultimately decides scheduling order.

Read-only `results` and `progress` compare HLT students to C0, S00 when present,
and historical M0HLT/OFFLINE recovery anchors. Teacher scores are explicitly
oracle results. Original50k overlaps validation. All results are single-seed
exploratory development validation; final test remains sealed. This campaign
cannot establish significance from the best of twelve recipes alone.

Method references: [SupCon](https://arxiv.org/abs/2004.11362),
[VICReg](https://arxiv.org/abs/2105.04906),
[Barlow Twins](https://arxiv.org/abs/2103.03230),
[masked particle modeling](https://arxiv.org/abs/2409.12589).
These inspire the specified adaptations; they do not establish an HLT gain.
