# JetClass2 Delphes 500k Full-Cardinality Salience Persistent-HLT Plan

Date: 2026-09-13

Status: implementation-authoritative plan for an isolated SPORC study. This
document does not authorize live submission or final-test access.

## Objective

Port the exact full-cardinality salience matching strategy to Luka's JetClass2
Delphes benchmark and measure immediate-parent logit distillation over three
persistent-HLT spines. Hold the dataset, selected rows, model interface,
optimizer, loss, schedule, and execution site fixed relative to the registered
JetClass2 500k study. Change matching/support semantics explicitly and rebuild
all assignment-dependent artifacts.

## Frozen population and execution

- Reuse the immutable `TRAIN_500K` membership: exactly 500,000 training rows.
- Reuse the same exact 1,000,000 validation identities.
- Keep the same exact 1,000,000 final-test identities sealed.
- Use the 17-input, 11-output, capacity-240, `trim=False` model interface.
- Execute on SPORC `tier3`, account `reu-aisocial`, QoS `qos_tier3`, one
  A100, eight CPUs, 72 GiB, and `atlas_kd_sporc`, subject to fresh acceptance.
- Particle views and caches are RAM-only. No rolling optimizer resume exists.

## Matching

Use the versioned `HCWDL_FULLCARD_SALIENCE_MATCHER_SPEC/v1` implementation.
Every jet has exactly `min(n_HLT,n_offline)` one-to-one pairs. The global
primary objective maximizes total bounded angular closeness weighted by
endpoint pT salience; exact secondary objectives and integer tie-breaking are
unchanged from the reusable matcher contract.

The three preregistered candidates are `SALIENCE_PT_LINEAR`,
`SALIENCE_PT_QUADRATIC`, and `SALIENCE_PT_QUADRATIC_CORE25`. Because JetClass2
is a new domain, build all three foundations and run a matched CE-only U100
screen. The historical bottleneck matcher is a contextual fourth row but is
not eligible to win. Selection uses a deterministic 75% checkpoint / 25%
one-shot selection partition of the fixed validation role, macro OVR AUC, a
`5e-5` AUC equivalence band, then macro log R50, accuracy, pT-weighted selected
delta R, and registry order. Poor metrics never fail completion.

## Persistent-HLT view

The complete HLT skeleton is present at every point:

- `U000`: matched HLT slots carry offline endpoint features; unmatched HLT
  slots stay native HLT; unused offline particles form the source-only tail.
- U progression removes only that source-only offline tail.
- `U100`: exact HLT cardinality, matched slots still offline, unmatched slots
  native HLT.
- D progression changes matched slots from offline toward HLT.
- `D000`: exact native HLT and requires no offline input.

Pair validity is structural provenance, never correspondence confidence and
never a deployable model input.

## Three-spine production graph

```text
fresh M0HLT: native-HLT CE baseline
fresh U000:  persistent-HLT privileged CE anchor

DIRECT:
U000 -> D000

COARSE:
U000 -> U050 -> U100 -> D066 -> D033 -> D000

DENSE:
U000 -> U033 -> U066 -> U100 -> D080 -> D060 -> D040 -> D020 -> D000
```

There is no ULTRADENSE branch in the graph. There are 16 fresh fits: M0HLT,
the shared U000 anchor, and 14 branch students. The 12 nonterminal teacher
nodes publish compact probability banks. Every student uses only its immediate
single parent. Repeated coordinates in different branches are independent
fits with paired coordinate seeds.

## Training recipe

- References: CE only.
- Students: `0.25 CE + 0.75 forward-KL`, temperature 2, exactly one T-squared
  factor.
- AdamW: betas 0.9/0.999, epsilon `1e-8`, weight decay 0.01.
- Batch 256, one GPU, accumulation 1, BF16 forward and FP32 loss.
- Passes 1-3 warm up to `3e-4`; 4-45 hold; 46-60 cosine decay to `1.5e-5`;
  61-100 constant floor.
- Minimum 60, maximum 100, patience 15 on improvements greater than `5e-5`.
- Validate each pass and restore the best checkpoint by AUC, CE, log R50, then
  earliest update.
- Every selected training row appears once per pass; no class rebalance.

## Artifact and gate sequence

1. Each candidate foundation publishes compact identity/offset/int32 mappings,
   diagnostics, exhaustive small-row checks, sampled recomputation, and a lock.
2. A screen preflight authenticates all four foundations, verifies installed
   Weaver and the measured A100 allocation, exercises each candidate on real
   ordinary-role data, and publishes a salience-family runtime profile.
3. Four matched CE U100 fits run independently; selection publishes a report
   and immutable selection lock.
4. Production creation authenticates the winner foundation, selection lock,
   runtime profile, source bytes, split hashes, and model contract.
5. Production dry-run must show 30 tasks: 16 train, 12 reduce, aggregate, and
   completion. Live submission is a separate explicit authorization.

No stage mutates the old bottleneck campaign, its foundation, active jobs, raw
data, or frozen split registry. No ordinary stage reads final-test particles.

## Required validation

- Exact production/exhaustive assignment agreement on eligible small jets.
- Smaller-side coverage and no index reuse for both cardinality orientations.
- Deterministic assignment and persistent-support nesting.
- Exact D000 native-HLT endpoint without offline access.
- Assignment byte/lineage validation and sampled recomputation.
- Candidate screen split firewall and recomputed winner.
- Source-pinned SPORC worker/environment/allocation checks.
- Synthetic campaign, task graph, output inventory, dry/live guard, monitor,
  and restart-zero recovery tests.
- Genuine installed-Weaver/A100 preflight before production is considered
  eligible for submission.

