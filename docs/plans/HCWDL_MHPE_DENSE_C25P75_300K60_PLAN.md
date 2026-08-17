# HCWDL-MHPE dense C25P75 anchor-50 300k/60-pass campaign

Status: scientific implementation authority for one additive, paired campaign profile.

## Scientific question

This campaign asks whether the dense anchor-50 MHPE ladder benefits from more
label supervision. It is paired to `C10P90_DENSE_ANCHOR50_300K60` and changes
exactly one scientific variable for every specialist: `0.10 CE + 0.90 KD`
becomes `0.25 CE + 0.75 KD`. M1 remains `0.10 CE + 0.90 KD` at temperature 1.

All other semantics are fixed: authenticated 300k train, 100k validation,
100k sealed-final-test population; unified 21-channel architecture; fresh
initialization; 60 complete passes; validation every pass; unweighted CE;
temperature-2 specialist KD; macro-AUC-first checkpoint selection; view
coordinates; teacher routes; seed aliases; probability construction; exact
anchor-50 weights; resources; and task dependencies.

## Exact graph

`U000` and `M0paired` are imported from the completed unified-balanced 300k
foundation. The campaign owns the same 29 fits and six reducers as the dense
C10 profile:

```text
U033:  U000                                                    (1)
U066:  U000, U033                                              (2) -> U066E
U100:  U000, U033, U066E                                       (3) -> U100E
D75:   U000, U033, U066E, U100E                                (4) -> D75E
D50:   U000, U033, U066E, U100E, D75E                          (5) -> D50E
D25:   U000, U033, U066E, U100E, D75E, D50E                    (6) -> D25E
D0:    U000, U033, U066E, U100E, D75E, D50E, D25E              (7) -> D0E
M1:    D0E                                                      (1)
```

The coordinates are `U033=V(1/3,0)`, `U066=V(2/3,0)`, `U100=V(1,0)`,
`D75=V(1,1/4)`, `D50=V(1,1/2)`, `D25=V(1,3/4)`, and
`D0=M1=V(1,1)`. `D0` specialists, `D0E`, and M1 are HLT-only.

## Loss and ensemble contracts

Every specialist uses unweighted `0.25 CE + 0.75 KD` with temperature 2.
M1 uses `0.10 CE + 0.90 KD` with temperature 1. Each ensemble gives exactly
half its probability mass to the immediate-predecessor specialist and divides
the other half equally among skip specialists. Component probabilities are
computed in FP32, combined in lexical order with exact rational weights and
an FP64 accumulator, then published as little-endian FP32. Validation metrics
never alter weights, edges, or execution.

The dense C10 and dense C25 graphs use the same seed aliases at corresponding
nodes. Their paired comparison is therefore about specialist CE/KD weighting,
not initialization, sampling, dropout, graph depth, ensemble composition, or
data selection.

## Contracts and execution

The profile name is `C25P75_DENSE_ANCHOR50_300K60`. Graph, node, recipe,
campaign, training-report, foundation-reuse-lock, and waiver identities are
version 6 (reuse lock version 4). Existing dense C10 v5/v3 artifacts remain
immutable. Anchor-50 target, stage, aggregate, finalist, completion, and final
evaluation v2 contracts are reused because their reducer semantics are
unchanged; every artifact still binds this profile and its v6 recipe lineage.

There are exactly 38 jobs: 29 fits, six reducers, aggregate, finalist lock,
and completion. GPU jobs request 8 CPUs, 96G, six hours, and one GH200. CPU
reporting requests 4 CPUs, 32G, and one hour. A separate `hcwmhpe25d` Slurm
namespace prevents confusion with the running dense C10 campaign.

No new standalone smoke is required for this loss-only paired profile. This
is an explicit operational waiver based on the authenticated 300k foundation,
executed dense graph machinery, focused numerical tests, and a complete
nonmutating 38-job dry run. Live submission still requires the exact pushed
commit, clean detached Tigris worktree, immutable spec, and explicit creation
and submission authorization phrases. Final test remains sealed.
