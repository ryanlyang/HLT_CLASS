# Lower noise and context coupled tracking pilot

## Purpose and scope

Implement the user's proposed mechanism test: reduce independent smearing and
introduce substantial, learnable context-dependent structure. This is deliberately
artificial, not a detector response, CMS fit, or evidence that ladder KD must win.
Freeze constants before classifier results. Poor distributions or negative KD
results are scientific outcomes, never reasons to discard jets or cancel rows.

This plan authorizes implementation of one training-only diagnostic pilot, not
2.25M production generation or classifier submission. Existing NOISE_V3, its
Oscar copy and running classifiers remain immutable. This new recipe has its
own `JC2_LITERATURE_CONTEXT_*/v1` namespace and sibling output root.

## Population and controls

Use the already completed 20,000-jet COUNT38_V2 pilot at
`/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_literature_count38_d4b5d5fd_r1/study_spec.json`.
Authenticate its full v1/v2 ancestry, saved physical blocks, calibration and
completion receipt. Read the saved OFFLINE and count38 control, not ROOT, labels,
CMS data, validation or final test. This is the same inherited training-only
population, not a new random prefix of the production dataset. Its selection
was label-stratified; the new transformation and calibration reuse are label-blind.

Four aligned endpoints are reported:

| Endpoint | Added tracking noise scale | Kinematic noise scale | Context map |
| --- | --- | --- | --- |
| OFFLINE | None | None | None |
| NOISE_V3 control | 4 | 2 | None |
| LOW_NOISE | 1 | 1 | None |
| CONTEXT | 1 | 1 | Six value couplings and two error-scale couplings |

Scales are relative to the original nominal literature generator, not fractions
of total measured displacement or error. Tracking added variance is 1/16 of V3;
added tracking amplitude is 1/4. Kinematic amplitudes are 1/2 of V3. Tracking
value noise and uncertainty inflation stay coupled in the LOW_NOISE base.
Do not simply divide previously stored noisy values; regenerate from the saved
OFFLINE with frozen random streams and full-precision count38 probabilities.

PID scale remains 3. Drop/merge eligibility, original-coordinate pair priorities,
draws, ancestry, charge, category and validity masks must equal the saved v2
control for every jet. Counts therefore equal that pilot's observed 38.0021
mean exactly on its original 20k population. A new population need not average
exactly 38. All low/context p4 arrays are byte-identical. This holds topology
fixed; it does not claim drops, conversions or noise are themselves invertible.

## Context and exact equations

Context is calculated after low-noise generation, drops and merges. Use only
physical p4, quantized to the model's float32 precision then evaluated in float64.
Canonical lexicographic p4 ordering makes sums independent of particle order;
no identity, source key, ancestry, label or offline geometry is an input.

For each constituent define `q=tanh(log1p(pT/2)/2)`,
`r=deltaR_to_visible_jet_axis/(0.1+deltaR_to_visible_jet_axis)` and
`rho=S/(1+S)`, where `S=sum_{j!=i} exp(-(deltaR_ij/0.1)^2)`.
Angles wrap; the jet axis is the sum of this same view's p4, with a numerical
pT floor of 1e-8. These bounded variables remain available at HLT inference.

Only particles with all four tracking-validity bits set are transformed.
Neutral and partially missing tracking is preserved byte-for-byte. Let
`x=asinh(d0/0.1 mm)` and `y=asinh(dz/0.2 mm)`. For k=0,1,2:

```text
theta = 2*pi*((k+1)*rho + q) + pi*r
a = 0.65*sin(theta)             b = 0.65*cos(1.3*theta)
u = 0.8*sin(theta+0.7)          v = 0.8*cos(theta-0.4)
x <- x + a*tanh(y+u)
y <- y + b*tanh(x+v)            # uses the updated x
```

Finally `d0'=0.1*sinh(x)`, `dz'=0.2*sinh(y)` in mm, and

```text
d0err' = d0err * exp(0.3*sin(2*pi*(rho+r)) + 0.2*tanh(y))
dzerr' = dzerr * exp(0.3*cos(2*pi*(q-r))   + 0.2*tanh(x))
```

All constants are deliberate synthetic stress-test choices, not measured or
literature-derived detector parameters. After this map, error fields are
encoded scales, not newly justified physical standard-deviation uncertainties;
their ratios to values are synthetic diagnostics, not calibrated pulls.

The inverse needs the CONTEXT endpoint only: compute context and final x/y,
undo error scales, then reverse the six triangular updates and reconstruct
tracking. No inversion model is fitted. Require raw inverse scaled error
`abs(recovered-original)/max(1,abs(original)) <= 1e-10` per jet. Invalid,
nonfinite, or non-replaying outputs fail as implementation errors.

## Model input issue and mandatory future comparison rule

The existing 17-input adapter applies tanh to displacement and clips errors at
1 mm. That is not an information-preserving representation in float32. Report
saturated displacement slots and clipped-error slots, with denominators.
Do not assert raw invertibility proves equivalence after that frontend.

Provide a separately versioned 17-input alternative, retaining all old p4,
kinematic and PID inputs but replacing four tracking features with
`asinh([d0,dz]/[0.1,0.2])/4` and `log1p([d0err,dzerr]/[0.02,0.05])/2`.
No tracking clipping; no added identities/ancestry/features. Measure its
float32 reconstruction error, not claim exact losslessness. The pilot does
not alter any old campaign or activate this frontend in an existing trainer.

Any future KD comparison must register the SAME new frontend for every view,
teacher and CE/direct/coarse arm. Also hold matching/assignment policy fixed
across LOW_NOISE and CONTEXT, with no oracle ancestry available to the models.
The key contrast is ladder-minus-direct improvement on CONTEXT versus LOW_NOISE,
not a comparison only against the noisier historical V3 scores. Use paired
seeds, matched data/optimization budgets and unchanged validation selection.
This future classifier campaign is not implemented or authorized here.

## Pilot diagnostics and execution

Save four-side moments and histograms, PID/count distributions, jet p4/width,
tracking values/scales/significances/validity, paired response and observed-density
conditional summaries. CSV retains tails and counts; SD is distribution width.
Save NOISE_V3/LOW_NOISE/CONTEXT physical banks, plus separate diagnostic ancestry.
Report raw inverse error, transformed coverage, old-input saturation and the new
frontend's reconstruction error. All selected jets remain in the report.

Exactly one Tigris CPU job, 16 CPUs, 64 GiB, 2h, account reu-aisocial. Up to
16 spawn processes, one numerical thread each, shared immutable calibration.
Run up to 64 jets serially and in spawned workers and require exact replay.
Record measured file runtimes/RSS. Requested resources are an envelope, not an
already measured prediction. No GPU or automatic followup. Clean pushed source,
full dry review, site test-only and exact plan-hash authorization are required.

```bash
bash scripts/queue_jetclass2_literature_context.sh COMMIT COUNT38_SPEC NEW_ROOT
# Review the printed one-job plan, then use that exact hash:
bash scripts/queue_jetclass2_literature_context.sh COMMIT COUNT38_SPEC NEW_ROOT \
  --execute REVIEWED_PLAN_HASH
```

The parent physical pilot blocks live at RIT; the Oscar production transfer did
not include this complete v1/v2 pilot ancestry. This initial CPU pilot targets
Tigris deliberately. It neither overwrites nor requires re-transferring the
existing full Oscar dataset. Inspect its genuine results before choosing any
new production dataset or classifier campaign.
