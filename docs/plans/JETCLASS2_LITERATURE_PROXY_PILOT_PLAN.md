# Literature-inspired JetClass2 response: frozen pilot v1

## Scope and scientific claim

This is a separate, controlled stochastic HLT-like benchmark, **not CMS HLT**,
not a detector simulation, and not a continuation of the CMS-fitted response.
It neither consumes nor modifies those response models, datasets, or campaigns.
The user authorized implementation and a quick Tigris pilot. Production
generation, classifier training, and any final-test access are outside this pilot.

The mechanisms below are motivated by:

- [CMS particle flow, 2017](https://arxiv.org/abs/1706.04965): tracks and calorimeter
  clusters provide complementary information; finite granularity and nearby
  showers complicate particle identification and separation.
- [CMS tracking, 2014](https://arxiv.org/abs/1405.6569): tracking performance depends
  on momentum, displacement, and environment. Its offline resolutions are not
  measurements of an additional HLT smearing.
- [CMS Run-3 HLT tracking](https://twiki.cern.ch/twiki/bin/view/CMSPublic/Run3TrackingHLT):
  online reconstruction makes different efficiency/computing tradeoffs.
- [Delphes 3](https://arxiv.org/abs/1307.6346): modular parametric detector response,
  tracking inefficiency, and calorimeter information surviving missing tracks.

**Every numerical constant here is a frozen benchmark choice, not a rate inferred
from these publications.** No fitting to CMS data, labels, KD performance, or
the earlier proxy diagnostics is performed. Neither physics realism nor useful
distillation is guaranteed. A disappointing pilot is a result, not a job failure.

## Inputs and population

Use the authenticated 20260918 dzfix **offline** particle arrays in GeV and mm.
Preserve the producer's impact-parameter sign and reference point. No native
HLT arrays, labels, raw CMS data, validation particles, or final-test particles
are read. Existing split metadata may be authenticated in full; hashing the
bytes of a selected training ROOT file does not evaluate its HLT branches.

The existing profile already contains label-stratified, HLT-matched eligibility.
We inherit that selection; we do not claim the parent population was selected
without labels or native HLT metadata. The *new* selection and generator are
label-blind. Select exactly 20,000 rows by smallest SHA-256 rank of domain, seed,
and immutable row identity, restricted to the profile's train membership. Restore
file/entry order. No new split, class balancing, event cuts, or particle truncation.
Smaller positive counts are allowed only for explicit implementation miniatures.

Default RC inputs:

```
/home/ryreu/atlas/datasets/jetclass2_10M_20260918_dzfix_partial_v1/jetclass2
/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_dzfix_partial_inventory_0b3ed523_r1/inventory.json
/home/ryreu/atlas/datasets/jetclass2_10M_20260918_dzfix_partial_v1/jetclass2_delphes_scaling_splits_20260918_v1/profiles/TRAIN_1M.json
```

## Exactly ordered response

The recipe artifact in `literature_proxy/kernel.py` is machine-readable authority
for constants; changing any scientific rule requires a new version, not editing
an old artifact. Freeze seed 20261002; evaluate strengths 0.5, 1.0, 1.5 on identical
jets. These are MILD, NOMINAL, STRONG, not three models selected by performance.

1. Canonicalize native particle keys. Compute nearest-other distance using
   wrapped phi, and `c=max(0,1-nearest_deltaR/0.05)`. Empty/singleton jets have
   zero crowding. Original coordinates determine all eligibility.
2. A charged hadron becomes a neutral hadron with probability
   `lambda*(0.03+0.05*exp(-pt/2)+0.04*c)`. An electron becomes a photon with
   probability `lambda*0.01*(1+c)`. Muon identity is unchanged. Unknown PID is
   retained, not guessed. Converted candidates retain their energy-bearing
   particle; charge becomes zero and tracking becomes unavailable.
3. For surviving charged particles with a valid value **and** positive valid
   uncertainty, independently smear d0 and dz. Let `k=1.5+0.5*c`,
   `v_add=lambda^2*((k^2-1)*sigma^2+b^2)` with b=0.020 mm for d0 and
   0.050 mm for dz. Set `d'=d+sqrt(v_add)*Z` and
   `sigma'=sqrt(sigma^2+v_add)`. Significance is always derived as d'/sigma'.
   With an unavailable error, preserve the value and mask rather than inventing
   a measurement. Nonfinite tracking source entries are unavailable; negative
   finite errors are invalid input. Neutral tracking is unavailable.
4. Smear pT using a mean-one positive lognormal with relative SD
   `lambda*(0.01 charged, 0.03 photon, 0.10 neutral hadron)`; unknowns use charged
   or neutral-hadron response according to their charge. Independently smear eta
   and phi with Gaussian SD `lambda*(0.001,0.003,0.010)` in the same categories.
   Rebuild physical four-vectors preserving individual invariant mass; tolerate
   only the existing source-rounding tolerance before rebuilding. Converted
   particles use their new category. No jet-pT rescaling or energy-sum constraint.
5. With probability `lambda*0.05` in eligible jets, drop exactly one uniformly
   selected *original* charged hadron, neutral hadron, or photon with pT<2 GeV.
   Electrons, muons, unknowns are ineligible. No resampling if none eligible.
6. Among surviving post-conversion same-category neutral pairs (NH/NH or
   photon/photon) with original deltaR<0.03, merge one uniformly selected pair
   with probability `lambda*0.05` in eligible jets. Sum smeared four-vectors,
   retain the common category, zero charge/tracking. This is not a massless
   projection. No second merge. Drop and merge can occur in the same jet.

Probabilities are below one at every registered strength. Empty input/output is
permitted and counted, not silently replaced. No splitting, extra particles,
PUPPI correction, pileup simulation, jet reclustering, or outlier component.
Consequently particle count changes by at most two, and neutral *fraction/count*
can grow by losing track identity without manufacturing particles. Charge and
energy need not equal the offline jet after losses/noise; merge alone conserves p4.

Independent PCG64 streams are seeded by SHA-256 of version, seed, jet identity,
and operation domain. Strength is deliberately absent: common random numbers
are shared between variants. Draw arrays use canonical native-key order before
any loss. Results are invariant to worker count, process scheduling, file chunks,
and input order after canonicalization. Exact replay is promised within the
same recorded numerical environment, not across CPU architectures/NumPy versions.
Strength zero is an exact identity diagnostic (not a fourth scientific variant).

## Pilot execution and artifacts

One CPU-only Tigris Slurm job, 16 allocated CPUs, up to 16 per-file workers,
64 GiB, 2 hours, no GPU. Numerical libraries each use one thread. These are
conservative initial requests, **not measured runtime promises**. Fewer source
files than workers naturally limits concurrency. Hash each used source file
before and after its bounded 512-entry offline-only reads, not once per jet.

Worker publishes immutable per-file physical NPZ blocks, diagnostics JSON,
and separate lineage JSON. Identity and offsets pair offline and each proxy
unambiguously. Physical columns contain p4, charge, category, tracking and masks;
ancestry is forbidden as classifier input. These are diagnostic pilot blocks,
not a production classifier-ready dataset or a replacement split registry.
Do not use generator ancestry as a shortcut for the future ladder's matcher;
that would change the supervision experiment and needs separate authorization.

Run a real serial/process replay on up to 64 selected jets before bulk work.
Save exact replay evidence, source commit, recipe/population/source hashes,
environment, per-file elapsed times, ROOT authentication/read time, and peak
per-process RSS where available. RSS is not claimed to be total simultaneous RAM.
Print per-file progress. A completion receipt hashes every output. Existing
outputs are verified before reuse. Exclusive run/submission locks prevent races;
failed locks are left for deliberate inspection, never automatically stolen.

Output includes counts per jet/PID, charged pT fraction, jet pT/energy/mass/eta/
width, particle pT/eta/radius, d0/dz/errors/significances and masks, absolute
tracking tails, response counts with eligible denominators, and conditional
tracking/PID diagnostics in original pT/eta/crowding bins. Fixed histogram bins
include under/overflow. Means/SD are exact aggregate moments; quantiles are
histogram approximations with overflow explicitly represented. SD is distribution
width, not uncertainty on a mean. Particle summaries are particle-weighted; jet
summaries are jet-weighted. No replicas inflate independent jet counts.

Also save CSV and overlay PDF plus small paired examples. Population-level
offline/proxy differences are expected: no automatic goodness threshold or
variant winner. Inspect physics plausibility before authorizing larger generation.

## Gates and handoff

Local tests must cover identity, source units/masks, PID coherence, physical p4,
coupled tracking, loss caps, merge conservation, replay, split isolation, corrupt
sources/artifacts, atomic writes, dry-only default, exact plan authorization and
source pinning. A clean pushed commit, authenticated metadata, reviewed plan,
`sbatch --test-only`, and explicit execution are required to queue the pilot.
The pilot itself is the genuine Tigris miniature; local tests do not substitute
for it. No automatic production, classification, or follow-up submission.

Post-pilot: inspect overlays and examples, document measured throughput and
memory, freeze or explicitly version changes. Do not tune to make KD succeed.

## Queue interface

After committing and pushing the scoped implementation, substitute that exact
40-character commit below on Tigris. Use a fresh suffix if the root exists; do
not delete or overwrite an existing study. The helper has the frozen dzfix
paths above; a different authenticated source requires explicit CLI arguments.

```bash
COMMIT=PUT_THE_PUSHED_40_CHARACTER_COMMIT_HERE
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT=/home/ryreu/atlas/HLT_Classification_litproxy_${COMMIT:0:8}
ROOT=${MAIN}/checkpoints/jc2_literature_pilot_${COMMIT:0:8}_r1
git -C "$MAIN" fetch origin main
git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
bash "$PROJECT/scripts/queue_jetclass2_literature_proxy_pilot.sh" "$COMMIT" "$ROOT"
```

Review the dry plan: one job, 16 CPUs, 64 GiB, two hours, no GPU, 20,000 train
jets. Then use the printed PLAN `content_hash` (not the SPEC hash):

```bash
PLAN_HASH=PUT_THE_REVIEWED_PLAN_HASH_HERE
bash "$PROJECT/scripts/queue_jetclass2_literature_proxy_pilot.sh" \
  "$COMMIT" "$ROOT" --execute "$PLAN_HASH"
```

The helper returns the one job ID. There are no deferred controllers or further
campaign stages to authorize. Read Slurm status with `sacct -j JOBID -X -P
-o JobID,JobName,State,ExitCode,Elapsed`; progress is in `$ROOT/slurm-JOBID.out`.
After completion, activate `atlas_kd_tigris`, set the worker's environment, and
run `python -s "$PROJECT/scripts/jetclass2_literature_proxy.py" results --spec
"$ROOT/study_spec.json"`. The report is printed in the job log too. Download
`statistics.csv`, `overlays.pdf`, `examples.json`, and `report.json` for review.
No file transfer, SSH connection or Slurm submission was performed locally.
