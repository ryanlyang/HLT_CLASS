# JetClass2 Delphes 500k Salience Learned-Fusion Handoff Plan

Date: 2026-09-13

Status: implementation-authoritative plan for an isolated SPORC study. This
document does not authorize live submission or final-test access.

## Objective

Reimplement Strategy B from
`HCWDL_ADJACENT_VIEW_FUSION_HANDOFF_LADDERS_PLAN.md` on the JetClass2 Delphes
benchmark using the selected full-cardinality salience matcher and the
persistent-HLT view. The study asks whether removable representation-level
context can carry more of the privileged-to-HLT performance gap than ordinary
single-view logit KD.

The campaign is independent of the historical CMS/Tigris campaign and of the
ordinary JetClass2 salience ladder. It trains fresh references and every
scientific student on the fixed `TRAIN_500K` population.

## Frozen data and execution boundary

- Exactly 500,000 registered training identities.
- Exactly the shared 1,000,000 validation identities.
- The shared 1,000,000 final-test identities remain sealed and unavailable.
- Seventeen particle inputs, eleven classes, capacity 240, and `trim=False`.
- The immutable selected salience matcher foundation and selection lock are
  mandatory parents. A bottleneck or nearest-neighbour foundation is invalid.
- SPORC `tier3`, account `reu-aisocial`, QoS `qos_tier3`, one A100 per fit.
- Batch size 256, BF16 forward, FP32 losses, no gradient accumulation.
- Particle tensors, paired views, context states, morph coordinates, optimizer
  state, and in-progress checkpoints are RAM/device-only.
- Only selected checkpoints, reports, attestations, compact probability banks,
  and small locks are durable. There is no rolling resume.

## Persistent-HLT coordinates

The complete HLT skeleton persists through the path. At `U000`, matched HLT
slots contain offline endpoint content, unmatched HLT slots remain native, and
unused offline particles form a source-only tail. The U path removes only that
tail. At `U100`, the tail is gone. The D path replaces matched offline content
with HLT content. `D000` is exactly native HLT and has no offline capability.

Pair identities, salience values, assignment indices, degradation coordinates,
and construction metadata are never model inputs.

## Strategy-B transition

For every registered parent-to-child arrow, including U-side arrows, the
campaign owns three matched fits:

1. `DIRECT`: a cold ordinary child-view model trained with C25/P75, T=2 logit
   KD from the incoming single-view parent carrier.
2. `ACQUIRE`: a cold asymmetric two-view model. The child view is the primary
   branch and owns the only classifier; the parent view is removable context.
   One-way context residuals enter after particle blocks 2, 4, 6, and 8. The
   model uses the same parent teacher and C25/P75, T=2 objective.
3. `WITHDRAW`: a fresh-optimizer fit initialized from the selected acquisition
   checkpoint. It holds context strength at one through pass 10, cosine-decays
   it to zero through pass 60, and uses exact zero from pass 61. Selection is
   exclusively the exact context-free route. The selected primary branch is
   physically extracted as an ordinary single-view child carrier.

The withdrawal objective is fixed at 0.25 lower CE, 0.30 lower teacher KD,
0.15 privileged CE, 0.20 privileged teacher KD, 0.05 directed logit
consistency, and 0.05 blockwise representation consistency. The frozen
acquisition checkpoint is its teacher. At exact zero, context is neither read
nor allocated.

Each next arrow is cold-started. It receives only the extracted parent's logits;
no fusion weights, optimizer state, or hidden representations cross a rung.

## Three main spines

The shared references are a fresh native-HLT CE model `M0HLT` and a fresh
persistent-HLT privileged CE anchor `U000`.

```text
DIRECT:
U000 -> D000

COARSE:
U000 -> U050 -> U100 -> D066 -> D033 -> D000

DENSE:
U000 -> U033 -> U066 -> U100 -> D080 -> D060 -> D040 -> D020 -> D000
```

There are 14 arrows and therefore 42 main Strategy-B fits. Repeated coordinates
in different branches are independent artifacts. Within one arrow, DIRECT and
ACQUIRE use the same primary initialization and sampler seed; context-only
parameters use a separate graph-bound seed domain.

## One shared ten-fit control panel

The old Strategy-B control panel is run once, not duplicated by spine.

The first-rung topology/capacity controls attach to the DENSE branch's
`U100 -> D080` transition:

- `FUSION_LOW_LOW_D080`: both branches receive identical D080, alpha one, CE.
- `LOW_WARM_CONTINUE_D080`: continue its DIRECT D080 with CE under a fresh
  optimizer for the withdrawal-sized budget.
- `LOW_PARAMETER_MATCHED_D080`: cold single-view D080 with approximately the
  fusion model's trainable parameter count, CE only.

The terminal controls attach to the DENSE branch's `D020 -> D000` transition:

- `FUSION_LOW_LOW_D000`;
- `LOW_WARM_CONTINUE_D000`;
- `LOW_PARAMETER_MATCHED_D000`;
- `CE_SINGLE_D000`, a cold ordinary HLT-only model.

The three global controls start from the campaign's true source `U000`, not
from `U100`:

- `STATIC_U000_D000`: fixed `(context=U000, primary=D000)`, alpha one, CE only.
- `DIRECT_VIEW_MORPH_U000_TO_D000`: fixed D000 primary and a context view that
  follows the exact two-leg path below, alpha one, CE only.
- `DIRECT_VIEW_MORPH_WITHDRAW_D000`: initialized from the selected morph
  checkpoint, fixed `(D000,D000)`, trained with the ordinary withdrawal loss,
  selected at alpha zero, and physically extracted.

The exact one-indexed morph schedule is:

```text
pass 1:       U000
passes 2-26:  U004, U008, ..., U100
passes 27-51: D096, D092, ..., D000
passes 52-100:D000
```

Every coordinate is integer-defined over denominator 25. Passes before 51 are
ineligible for checkpoint selection and do not advance patience. This preserves
the old 50-transition morph duration while traversing both persistent-HLT axes
and leaves 50 endpoint validations including pass 51.

The campaign therefore contains 54 fresh fits: two references, 42 main fits,
and ten controls. Reporting aliases never add fits.

## Optimization and validation

All fresh fits use AdamW (betas 0.9/0.999, epsilon `1e-8`, weight decay 0.01)
and the registered schedule: passes 1-3 warm up to `3e-4`, passes 4-45 hold,
passes 46-60 cosine-decay to `1.5e-5`, and passes 61-100 remain at the floor.
Minimum passes are 60; patience is 15 on macro-AUC improvements greater than
`5e-5`, with the patience clock beginning after pass 60 (so the earliest
ordinary early stop is pass 75); the best eligible checkpoint is restored using AUC, cross entropy,
macro log R50, then earliest update.

Validation identities are deterministically and class-stratifiedly divided
into disjoint checkpoint, diagnostic, and report roles. Model selection uses
only the checkpoint role. Every registered result is reported on the untouched
report role. Poor metrics complete normally.

The aggregate reports accuracy, macro OVR AUC, linear macro R50, every class's
R50, and M0HLT-to-U000 recovery. It also reports paired differences and fixed
alpha curves, correct/zero/permuted context diagnostics, extraction parity,
context gain, withdrawal recovery, and all control contrasts inherited from
Strategy B. The global-control names and comparisons explicitly say U000.

## Artifact graph and queue gates

Campaign creation authenticates the winner salience foundation, its selection
lock, fixed split membership, model/input contracts, runtime profile, source
bytes, and fresh output root. The immutable DAG contains all 54 fits and every
required reducer, extraction, aggregate, and completion task before execution.

Queue readiness requires:

1. focused local graph/model/data/submission tests;
2. exact pushed source and a clean detached SPORC worktree;
3. a complete nonmutating dry run;
4. metadata, partition, and storage gates;
5. a genuine installed-Weaver A100 miniature proving acquisition,
   withdrawal, alpha-zero dispatch, extraction, U-side pairing, low-low
   identity, and all morph boundaries;
6. measured host/GPU memory and walltime with sufficient headroom;
7. a separate exact science dry run and explicit live authorization.

The screen and gate waits may be automated without weakening this sequence.
The registered deferred pipeline is:

```text
exact salience-screen complete job
  --afterok--> CPU-only campaign creator + four-gate submission
  --afterok on all four gate jobs--> gate authentication + science dry run
                                  + 87-task science submission
```

Scheduling that pipeline requires its own exact authorization phrase and a
fresh launcher root. It binds the screen's live submission ledger rather than
a job name, so an older or parallel screen cannot satisfy it accidentally.
The launch workers are source-pinned, validate their exact Slurm receipts, do
not poll, and make no mutation to either screen. A screen or gate failure
therefore leaves the downstream launcher unsatisfied instead of starting any
science work.

Recovery is restart-from-zero under a new source-pinned attempt. It may reuse
only content-valid completed parents, refuses ambiguous active jobs, never
modifies source campaigns or raw data, and never changes the scientific graph.
