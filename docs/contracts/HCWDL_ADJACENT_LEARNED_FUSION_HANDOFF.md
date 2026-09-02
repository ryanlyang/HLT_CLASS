# HCWDL Adjacent Learned-Fusion Handoff Contract

Version: `HCWDL_ADJACENT_LEARNED_FUSION_HANDOFF_*/v1`

This contract implements Strategy B in
`docs/plans/HCWDL_ADJACENT_VIEW_FUSION_HANDOFF_LADDERS_PLAN.md`. It is an
isolated 25-fit, full-data, single-GH200 campaign. Its initial source is the
authenticated completed non-persistent full-cardinality `U100` artifact used
by Strategy A. Source outputs are read-only and no running campaign is a
scheduler dependency.

## Scientific graph

The carrier coordinates are:

```text
U100 -> D080 -> D060 -> D040 -> D020 -> D000
```

At every transition the campaign owns three matched fits:

1. a cold ordinary lower-view `C25P75`, `T=2` direct-KD control;
2. a cold asymmetric two-view acquisition model, with the lower view as the
   primary branch and the adjacent richer view as removable context;
3. a fresh-optimizer withdrawal fit initialized from the selected acquisition
   checkpoint and selected only through exact `alpha=0` validation.

The withdrawal checkpoint is physically projected into an ordinary
three-input `ScoutingParticleTransformer`. Only that extracted checkpoint and
its compact probability table can teach the next rung. No fusion weights are
warm-started into the following rung.

## Architecture and withdrawal

The primary branch owns the only classifier. One-way context residuals enter
after particle blocks 2, 4, 6, and 8. Their output projections are initialized
to exact zero. At `alpha=0`, dispatch returns before context encoding,
cross-pair construction, or cross-attention. Extraction retains only the
ordinary primary `mod.*` state.

Withdrawal holds `alpha=1` through pass 10, uses a cosine trajectory through
pass 60, and is exactly zero from pass 61. Its fixed losses are 0.25 lower CE,
0.30 lower teacher KD, 0.15 privileged CE, 0.20 privileged teacher KD, 0.05
logit consistency, and 0.05 representation consistency. Selection is lower-
only macro AUC. Every fit uses the registered 100-pass H45/floor schedule,
minimum 60 passes, patience 15, batch 256, best-checkpoint restoration, and no
rolling resume. The lower primary and direct control reuse Strategy A's exact
rung seed aliases (including terminal seed `S1`), while architecture-only
context parameters use a separate graph-bound seed.

## Mandatory controls

The campaign additionally owns first-rung and terminal lower/lower fusion,
warm-continuation, and parameter-matched single-view controls; a terminal cold
ordinary CE model; static `(U100,D000)` fusion; the exact pass-indexed
`(U100,D000) -> (D000,D000)` context-morph control; and a separate explicit
withdrawal/extraction fit initialized from the morph checkpoint. Reporting
aliases never create extra fits.

The morph coordinate is integer-defined: pass 1 is `U100`; passes 2--51 use
feature-degradation numerator `pass-1` over denominator 50; passes 52--100 use
`D000`. Only one context coordinate may be resident in RAM at once. The morph
checkpoint remains a two-encoder model even when both supplied views are
`D000`; only its companion withdrawal publishes a single-view endpoint.
Checkpoint selection and the early-stopping improvement clock are ineligible
before pass 51. Thus the selected morph checkpoint is necessarily evaluated
with `(context=D000, primary=D000)` and cannot silently restore a privileged
earlier point on the input trajectory.

## Data, reporting, and storage

All models share authenticated train and validation rows. Validation is
class-stratified into disjoint `V_checkpoint`, `V_blend`, and `V_report`
partitions. It reproduces Strategy A's exact assignment rule and seed but is
published under the distinct Strategy-B validation-partition contract.
Fusion diagnostics evaluate both views, graph-bound zero context,
a zero-primary richer-context-only route, identity-permuted context, and exact
alpha zero on identical `V_report` identities. Every selected two-view
checkpoint also records a fixed alpha validation curve at
`[0, 0.25, 0.5, 0.75, 1]`; only metrics survive the three additional interior
inferences. Compact alpha-zero `V_report`
probabilities are hash-bound to the
diagnostic report so that the morph control can participate in paired
comparisons. The aggregate uses 2,000 deterministic label-stratified paired
Gaussian-multiplier AUC replicates for every adjacent carrier transition, all
eight preregistered causal comparisons, and the five matched
learned-carrier-minus-direct-KD contrasts. At every rung it also reports the
both-view context gain, alpha-zero acquisition result, extracted-carrier gain,
and fraction of context gain recovered by withdrawal. Every fit row embeds its
selected pass, complete validation/training histories, elapsed and preparation
times, peak CPU/GPU memory, parameter counts, durable bytes, and macro/per-class
M0CE60-to-U000 recovery. Final test is unavailable.

Cross-strategy `LEARNED_T_i - OUTPUT_T_i` comparisons are deliberately not
part of this isolated artifact. A later joint report may authenticate and
compare the two completed campaign aggregates, but Strategy B neither waits
on nor mutates Strategy A and never substitutes an incomplete output row.

Particle tensors, dynamic morph views, context states, cross-attention states,
and optimizer state are RAM/device-only. Durable artifacts are limited to
models, reports, attestations, and compact 15-class probability banks. The 11
distributions that supervise a later fit retain train and all three validation
roles; the 15 nonteacher distributions retain only `V_report`. This avoids
full-population target files and inference for controls that have no consumer.
A failed task restarts from update zero in a new source-pinned worktree.
The production preflight visits the D100 parity endpoint and D098/D000 morph
boundaries sequentially; it never retains more than two full coordinate caches
at once and retains only compact hash evidence after releasing each temporary
coordinate.

Scientific metric quality never controls DAG completion. Invalid lineage,
nonfinite required quantities, corrupt artifacts, forbidden final-test access,
or failure of exact extraction parity fail closed.

## Queue gate

Live science submission requires, in order: focused local tests; exact pushed
source; a dry-run gate ledger; genuine Tigris execution of authentication,
partition/storage audit, and installed-Weaver GH200 preflight; inspection of
measured headroom; and an exact science dry run. The full campaign is not
authorized by this document alone. Live science submission revalidates the
immutable output inventory and task attestation for all four gate tasks, not
only the terminal preflight artifact.
