# JetClass2 Delphes salience MT20 500k implementation plan

## Objective

Re-run the multi-grandparent logit-distillation intervention on the frozen
JetClass2 Delphes `TRAIN_500K` population and the selected full-cardinality
persistent-HLT salience matcher.  This is an isolated SPORC campaign.  It does
not modify or import fits from an immediate-parent campaign.

## Frozen data and views

- Train: the exact registered 500,000-jet `TRAIN_500K` membership.
- Validation: the exact registered 1,000,000-jet validation membership.
- Final test: the registered 1,000,000 jets remain sealed and inaccessible.
- Inputs: the reduced 17-input, 11-class JetClass2 contract.
- Matching: the already-completed `SALIENCE_PT_LINEAR` winner foundation at
  `checkpoints/jc2_salience500k_linear_4f5e520e_r1/foundation`, authorized by
  `checkpoints/jc2_salience500k_screen_4f5e520e_r1`. Its assignments and
  selection evidence are authenticated read-only before use. Matching and
  matcher selection are never rerun by this campaign.
- U000: offline content on matched HLT slots, native content on unmatched HLT
  slots, and unused offline particles only as the removable tail.
- U progression removes only that offline tail. U100 retains the HLT skeleton
  and offline content in matched slots. D progression interpolates matched
  content to HLT. D000 is exact native HLT and deployable.

Particle views and multi-teacher mixtures exist only in process RAM. Assignment
metadata is never a model input. The only large durable intermediate is a
compact float32 T=2 probability bank for each nonterminal model. The 12 banks'
raw identity-plus-probability payload is projected below 1 GiB; the complete
campaign has a conservative 4 GiB durable upper bound and requires at least
16 GiB free before execution.

## Three path-density ablations

The common fresh references are `M0HLT` at D000 and `U000`. The campaign then
runs three independent spines:

```text
DIRECT:      U000 -> D000
COARSE:      U000 -> U050 -> U100 -> D066 -> D033 -> D000
DENSE:       U000 -> U033 -> U066 -> U100 -> D080 -> D060 -> D040 -> D020 -> D000
```

The experiment therefore starts from U000. U100 is an intermediate coordinate;
there is no D100 start.

## Multi-grandparent objective

References use CE only. Every downstream fit uses C20/P80 at T=2 and every
earlier selected teacher from the same spine. Teachers are ordered nearest
first. Cross-spine teachers, model ensembles, and weight continuation are
forbidden.

- With one teacher, its KD contribution is 0.80.
- With at least two teachers, the immediate teacher contributes 0.50.
- All older teachers share the remaining 0.30 with nearest-first geometric
  decay ratio 1/2.

The exact rational contributions sum to 0.80. Durable component banks are
identity-joined in the student's exact train order, accumulated in float64,
normalized once, converted to contiguous float32, and discarded after the fit.
One forward-KL term against that mixture has the same student gradient and
optimum as the weighted component forward-KL terms; their scalar values differ
only by a teacher-only constant.

## Optimization and pairing

The schedule and every non-loss setting match the registered JetClass2
path-density protocol: batch 256, AdamW, BF16 forward/FP32 loss, warmup passes
1--3, hold through pass 45 at 3e-4, cosine decay through pass 60 to 1.5e-5,
constant floor through pass 100, minimum 60 passes, patience 15 with 5e-5 AUC
delta, and restoration of the selected checkpoint. There is no rolling resume.
Initialization and sampling seeds remain paired by coordinate across spines.

## DAG and storage

There are 16 fresh fits: two references plus 1/5/8 downstream fits. There are
12 probability reducers, one aggregate, and one completion task: 30 science
tasks. Three separate gates bring the registered total to 33 tasks.

These are path-density ablations only. DIRECT, COARSE, and DENSE use the same
MT20 teacher policy, loss, optimizer, data and selected matching foundation.
`M0HLT` and `U000` are common reference endpoints, not alternative
teacher-policy ablations. This campaign does not register immediate-parent,
C25/P75, seed-ensemble, or warm-start controls.

The gate authenticates source/data/foundation, audits free storage and forbidden
durable state, then runs a genuine SPORC A100 miniature using the selected
salience U000 view, installed production model, exact C20/P80 loss, and a
two-teacher RAM mixture. Science submission cannot be materialized until those
three reports are complete and authenticated.

SPORC execution is one A100, eight CPUs and 72 GiB under account
`reu-aisocial`, QoS `qos_tier3`, and environment
`/home/ryreu/miniconda3/envs/atlas_kd_sporc`. Submission routes every task to
the `debug` partition by default. Any explicitly named task may instead be
routed to `tier3`; this changes scheduler placement only, not its scientific
identity, resources, environment, dependencies or output contract. The exact
per-task routing is frozen in each dry-run/live command plan. Debug's registered
24-hour ceiling exceeds every request here: full fits and reducers retain the
measured walltimes of 808 and 43 minutes respectively. A pending task is not
mutated between partitions; changing its route requires exact-ID cancellation
and restart-zero recovery.

Gate and science are separate exact dry/live stages. Recovery starts failed
tasks from zero and reuses only authenticated completed parents.

## Success criteria

Scientific quality never controls workflow completion. Completion means every
registered fit and reducer produced valid finite, source-pinned artifacts; the
aggregate reports all three endpoints against fresh M0HLT and U000; no final-test
access occurred; no rolling state or durable mixture was written; and the
immutable campaign-complete artifact exists.
