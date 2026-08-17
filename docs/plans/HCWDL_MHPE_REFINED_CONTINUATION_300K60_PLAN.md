# HCWDL-MHPE Refined Continuation 300k/60-Pass Plan

Status: **implementation-authoritative additive validation campaign**.

This campaign tests whether a same-view distilled ensemble is a cleaner teacher
for the next HCWDL projection. It imports the completed authenticated
`C25P75_300K60` MHPE campaign read-only. It does not mutate or reinterpret that
campaign, and it never accesses final-test rows.

## Scientific graph

```text
source U100E --C10P90,T1--> U100R
U100R       --C25P75,T2--> D066_from_U100R
source D066E + D066_from_U100R --uniform four-member mean--> D066Eplus
D066Eplus   --C10P90,T1--> D066R
D066R       --C25P75,T2--> D033_from_D066R
source D033E + D033_from_D066R --uniform five-member mean--> D033Eplus
D033Eplus   --C10P90,T1--> D033R
D033R       --C25P75,T2--> D000_from_D033R
source D000E + D000_from_D033R --uniform six-member mean--> D000Eplus
D000Eplus   --C10P90,T1--> M1R
```

Every arrow is KD plus unweighted population CE. Every fit is fresh. Refiners
use 10% CE, 90% probability KD, and temperature 1 on the same input view.
Projection children use 25% CE, 75% model-logit KD, and temperature 2. All
fits run 60 complete passes, validate every pass, and select by macro AUC, CE,
logR50, then earliest update.

Projection children and `M1R` reuse the corresponding source target-view seed
alias. Refiner seeds are new and immutable. No checkpoint or optimizer state
crosses an edge.

## Exact ensemble semantics

```text
D066Eplus = (3 * D066E + p(D066_from_U100R)) / 4
D033Eplus = (4 * D033E + p(D033_from_D066R)) / 5
D000Eplus = (5 * D000E + p(D000_from_D033R)) / 6
```

Because every source ensemble is a uniform mean, these are exactly uniform
means of four, five, and six individual specialists. The reducer reloads all
source components, reproduces the durable source ensemble byte-for-byte, then
performs one canonical uniform reduction over the original components plus the
new specialist. Component softmax is FP32 at temperature 1; canonical addition
and division are FP64; publication rounds once to little-endian FP32. Weights
never use labels, confidence, class, or validation performance.

`M1R` is not blended back into `D000Eplus`. The primary endpoint comparison is
`M1R - source M1`.

## Population, reuse, and access

The source must authenticate as completed `C25P75_300K60` with exact role
counts 300k train, 100k validation, and sealed 100k final test. Creation binds
the source completion, foundation reuse lock, executable recipe, selected
reports/checkpoints, U100E/D066E/D033E/D000E T1 bundles, and stage reports.
Ordinary access is 300k train, 100k validation, zero final test.

Source probability banks are read-only. Augmented ensembles persist compact
identity-ordered probabilities and authenticated metadata only. Particle views
remain process-local RAM and are built once per job. Existing specialists are
not rerun.

## Jobs and comparisons

The DAG has seven GPU fits, three GPU reducers, one CPU aggregate, and one CPU
completion task. It is sequential because each refiner causes the next edge.
The aggregate reports the new versus source projection specialist, ensemble,
and endpoint at every stage. Macro OVR AUC is primary; CE and mean log QCD
rejection at 50% signal are mandatory. Finite poor performance completes.

## Operations and acceptance

GPU tasks request 8 CPUs, 96G, 06:00:00, and one GH200. CPU tasks request 4
CPUs, 32G, and one hour. GPU workers receive `USR1` 120 seconds before walltime
and use the absolute source-pinned project path and canonical Tigris environment.

The campaign requires a canonical dry-run ledger and separate live submission
authorization. It carries the completed 300k foundation and MHPE worker
evidence; no additional standalone smoke is required. This is an operational
evidence carry-forward, not a claim that this graph completed a smoke. Exact
failed/downstream recovery preserves completed outputs.

Queue readiness requires focused and complete tests, CLI help, compilation,
contract probes, synthetic target/numerical tests, Markdown links, and
`git diff --check`. Local implementation performs no Slurm mutation.
