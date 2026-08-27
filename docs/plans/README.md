# Scientific Implementation Plans

The additive
[TRI60 D000 matched-seed diversity ablation](HCWDL_TRI60_D000_SD5_ABLATION_PLAN.md)
re-fits the five frozen LOGIT D000 teacher edges with the exact five distinct
CE5 seed domains, then evaluates one fixed uniform validation-only ensemble.
It changes only stochastic domains, launches all five fits in parallel, and
cannot mutate or depend on the running TRI60, dense, or CE5 jobs. Its contract
and launch procedure are in the
[contract companion](../contracts/HCWDL_TRI60_D000_SD5_ABLATION.md) and
[runbook](../HCWDL_TRI60_D000_SD5_ABLATION_RUNBOOK.md).

The additive
[TRI60 dense-extension plan](HCWDL_MHPE_TRI60_DENSE_EXTENSION_PLAN.md)
reuses exact completed specialists from the running full-data campaign while
adding LOGIT D083/D050/D017 and RSET/RREL D075/D025 ensemble stages. It owns a
separate immutable graph, root, job namespace, ledger, source-completion gate,
and recovery path; it cannot mutate or control the running source campaign.
Its contracts and queue procedure are in the
[contract companion](../contracts/HCWDL_MHPE_TRI60_DENSE_EXTENSION.md) and
[runbook](../HCWDL_MHPE_TRI60_DENSE_EXTENSION_RUNBOOK.md).

The implementation-authoritative
[full-data three-track 60-pass ensemble-compression plan](HCWDL_MHPE_THREE_TRACK_60E_FULL_IMPLEMENTATION_PLAN.md)
defines one shared fresh U000 root, a full triangular LOGIT ladder, shorter
triangular RSET and RREL ladders, fixed within-track probability ensembles,
and an `M1_LOGIT + M1_RSET + M1_RREL -> M1E -> M2` HLT-only compression.
It freezes 32 fresh fits, 12 reducers, batch size 256, 60 passes, RAM-only
representation targets, durable compact probability banks, disabled rolling
resumes, and parallel full-data execution under one immutable campaign DAG.
Its reusable identities and operational handoff are in the
[contract companion](../contracts/HCWDL_MHPE_THREE_TRACK_60E_FULL.md) and
[source-pinned runbook](../HCWDL_MHPE_THREE_TRACK_60E_FULL_RUNBOOK.md).

The additive
[TRI60 full-data CE60 control plan](HCWDL_MHPE_TRI60_CE60_CONTROL_PLAN.md)
registers the independent pass-matched exact-HLT CE-only comparator without
changing the immutable three-track graph.

The additive [HCWDL-MHPE refined continuation](HCWDL_MHPE_REFINED_CONTINUATION_300K60_PLAN.md)
tests an ensemble-refine-project route from the completed uniform 300k MHPE
campaign using seven new fits and three equal-component reducers.

Campaign-specific scientific plans live here. A plan must freeze scientific
meaning, controls, access rules, selection, metrics, artifacts, and execution
stages before implementation.

The original implemented campaign is the
[Scouting Particle-Matched Alpha-Repair Distillation plan](SCOUTING_ALPHA_REPAIR_DISTILLATION_OPTIMAL_CAMPAIGN.md).
The user explicitly invoked its implementation mandate on 2026-08-05. PMARD's
versioned contracts govern Scouting data, matching, repair, teachers, students,
selection, storage, and final-test access.

The completed precursor design is documented in the
[high-coverage HLT-to-offline matching research plan](HIGH_COVERAGE_HLT_OFFLINE_MATCHING_RESEARCH_PLAN.md).
It framed the correspondence, coverage, confidence, audit, endpoint, and
storage questions and remains the research record rather than a reinterpretation
of the fitted-strict pilot. The independently selected matcher and portable
handoff are now imported normatively by the successor ladder plan below.

The current active successor plan is the
[high-coverage cold/warm distillation ladder plan](HIGH_COVERAGE_COLD_WARM_DISTILLATION_LADDER.md)
on 2026-08-08. It fixes Shell Exact v1 as the primary endpoint, ports the
high-coverage completion-shell matcher, requires less than 10% overall token
dustbins on both selected train and validation roles, and defines complete cold
and warm ladders, a 60-pass/every-pass-validation protocol, AUC-first checkpoint
selection, a deferred immutable optimization recipe, one-time assignments, and
the final-test lock boundary. Its endpoint qualification is diagnostic rather
than a validation-driven repair selector. It does not reinterpret the original
PMARD pilot.

The implementation-authoritative supplemental
[HCWDL structural-feature homotopy plan](HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_IMPLEMENTATION_PLAN.md)
defines the next validation-only 300k study. It adds a deeply specified
`TOFF -> U -> D100` structural-support bridge and compares a depth-matched
factorized `U then D` path with a joint `U+D` path. It explicitly
distinguishes the native two-stream TOFF oracle from the unified 21-channel P0
carrier, requires U100 to be byte-identical to existing D100 and both terminal
paths to be byte-identical HLT, registers stationary-depth controls, and never
reinterprets completed HCWDL or dense artifacts. Its presence authorizes
the implemented validation-only contract and local verification, not a live
campaign or final-test access. The reusable artifact identities are frozen in
the [structural-feature homotopy contract](../contracts/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY.md),
and the source-pinned dry-run, smoke, monitoring, resource-capture, pilot, and
recovery procedure is in the
[structural-feature homotopy runbook](../HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_RUNBOOK.md).

The implementation-authoritative successor
[HCWDL unified-root balanced homotopy plan](HCWDL_UNIFIED_BALANCED_HOMOTOPY_IMPLEMENTATION_PLAN.md)
defines a separate future campaign without changing the running HCWDL-UJ
execution. It promotes a CE-trained unified P0 model to the `U000` root,
replaces the cost-ordered U switch schedule with a deterministic balanced
atomic coordinate, and replaces confidence-warped D motion with uniform
continuous strength plus deterministic nested population-level switches for
discrete groups. It also freezes paired legacy-U and legacy-D ablations, a
sealed larger-data test boundary, and a six-arm concurrent
CE/parent-KD/grandparent-KD study sharing one authenticated U000 root. The
successor also records the user's explicit decision to carry forward prior
production-worker evidence instead of running another standalone smoke
campaign.
Its implemented reusable semantics are frozen in the
[unified-balanced homotopy contract](../contracts/HCWDL_UNIFIED_BALANCED_HOMOTOPY.md),
and its source-pinned foundation, six-arm submission, monitoring, recovery,
aggregation, and sealed-test procedure is in the
[unified-balanced homotopy runbook](../HCWDL_UNIFIED_BALANCED_HOMOTOPY_RUNBOOK.md).

The additive
[HCWDL unified-balanced full-data three-arm plan](HCWDL_UNIFIED_BALANCED_FULL_DATA_THREE_ARM_PLAN.md)
scales the same corrected balanced/uniform factorized path to every mapped row
with 20 passes and exactly `C25P75`, `C10P90`, and `C10P75G15`. It rebuilds
full-population assignments, couplings, roots, and targets under new contracts
and does not modify the running 300k six-arm campaign.

The additive
[HCWDL unified-balanced full-data coarse three-arm plan](HCWDL_UNIFIED_BALANCED_FULL_DATA_COARSE_THREE_ARM_PLAN.md)
reuses the authenticated completed FULL3 foundation and compares six-edge
factorized and joint paths for `C25P75`, `C10P90`, and `C10P75G15`. Its exact
thirds/sixths coordinates, paired seeds, 20-pass schedule, 36-fit registry,
and no-final-test boundary are new immutable contracts; it does not relabel or
modify the fine-grained FULL3 runs.

The implementation-authoritative
[HCWDL multi-horizon projection-ensemble full-data plan](HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_IMPLEMENTATION_PLAN.md)
defines a triangular teacher lattice over `U000`, `U050`, `U100`, `D066`,
`D033`, and exact-HLT `D000`. Historical teachers are projected independently
into each current student domain before uniform same-domain probability
ensembling; five exact-HLT specialists form `D000E`, which is compressed into
one HLT-only `M1`. It reuses the authenticated full-data foundation, fixes
`C25P75/T=2` for all 15 specialists and `C10P90/T=1` for M1, runs 20 full-data
passes, and records the user's explicit decision to proceed directly to the
all-mapped campaign after implementation and dry-run acceptance without a new
Slurm smoke or 300k pilot.

The implementation-authoritative additive
[HCWDL-MHPE C10P90 parallel plan](HCWDL_MHPE_C10P90_PARALLEL_PLAN.md)
changes only the 15 non-M1 specialists from `C25P75/T=2` to `C10P90/T=2`.
M1 remains `C10P90/T=1`; the foundation, graph topology, coordinates,
teachers, seeds, passes, ensembles, resources, and final-test boundary remain
paired. It runs in a distinct worktree/root/ledger with `hcwmhpe90_` job names
and does not modify the running primary campaign.

The implementation-authoritative
[paired 300k/60-pass MHPE study](HCWDL_MHPE_300K_60E_PAIRED_PLAN.md)
runs the same complete 16-fit/four-ensemble graph twice on the authenticated
300k/100k/100k population. It compares `C25P75/T=2` with `C10P90/T=2` for
specialists, keeps M1 at `C10P90/T=1`, and pairs all seeds across the two
independent 60-pass roots and ledgers.

The additive
[dense anchor-50 300k/60-pass MHPE campaign](HCWDL_MHPE_DENSE_ANCHOR50_300K60_PLAN.md)
uses 29 cold C10P90 fits across seven factorized horizons and six
predecessor-anchored probability ensembles. It reuses the authenticated 300k
foundation, keeps final test sealed, and has its own v5/v2 artifact family.

The paired
[dense C25P75 anchor-50 campaign](HCWDL_MHPE_DENSE_C25P75_300K60_PLAN.md)
repeats the same 29-fit graph, six exact anchor-weighted ensembles, stochastic
seeds, 300k/100k/100k population, and 60-pass schedule. Its only scientific
change is specialist loss from C10P90 to C25P75; M1 remains C10P90/T1.

The additive
[TRI60 five-seed CE ensemble reviewer study](HCWDL_TRI60_CE5_SEED_ENSEMBLE_REVIEWER_PLAN.md)
trains five independent full-data, 60-pass, exact-HLT CE models, averages all
five class-probability outputs uniformly, and compares C10P90/T1 compression
against an exactly seed-paired CE-only student. It has an isolated graph,
root, job namespace, ledger, and restart-from-zero recovery path.

The additive
[TRI60 standalone M1 compression screen](HCWDL_TRI60_M1_COMPRESSION_SCREEN_PLAN.md)
compares 20 exact-HLT LOGIT endpoint-compression conditions over cold, warm,
and polish initialization; C10P90/C25P75/C50P50 plus one predeclared ramp;
T=1/T=2; and three peak learning rates. It imports the completed source M1 as
one control, runs 19 independent full-data/60-pass fits, requires only the
already-complete LOGIT endpoint subset rather than source campaign completion,
and uses a separate nice-5000 job namespace with no standalone smoke.

The additive
[D066 full-data optimization schedule screen](HCWDL_MHPE_D066_SCHEDULE_SCREEN_FULL_PLAN.md)
is the immutable 60-fit predecessor study. The additive
[D000 teacher-distance full-data screen](HCWDL_MHPE_D000_TEACHER_DISTANCE_SCHEDULE_SCREEN_FULL_PLAN.md)
instead trains 24 paired 80-pass exact-HLT students and preserves the best
checkpoint available by passes 20, 40, 60, and 80.
tests 20 pass/peak-LR schedules under three paired teacher horizons. It uses
all mapped full-data training rows, partitions the full validation role into
disjoint checkpoint-selection and schedule-scoring halves, queues all 60 fits
independently, and never accesses final test.

The additive
[HCWDL matching-free privileged representation-KD ascent plan](HCWDL_MATCHING_FREE_REPRESENTATION_KD_ASCENTS.md)
registers two nested representation strategies, each as a complete cold and
warm M1--M6 ascent. It preserves the implemented logit-only HCWDL graph and
recipe as controls, uses compact one-time privileged target sketches, compares
paired jet and unordered token-set structure without particle correspondence,
and adds a corrected latent-relation ablation whose optimized coordinates
remain differentiable with respect to the student encoder. It is an
implementation-grade plan only; no representation-ascent implementation or
live authorization currently follows from its presence.

The validation-only
[HCWDL common-schema architecture–input factorial](HCWDL_ARCHITECTURE_INPUT_FACTORIAL_PLAN.md)
separates HLT versus projected-offline particle information from unified
versus charged/noncharged split processing. Its four CE-only paired cells use
one common 21-channel schema; canonical native `TOFF` remains an explicitly
nonfactorial reference. Its reusable identities are frozen in the
[factorial contract](../contracts/HCWDL_ARCHITECTURE_INPUT_FACTORIAL.md).

The [PRAD implementation plan](PRIVILEGED_RELATIONAL_ATTENTION_DISTILLATION_IMPLEMENTATION_PLAN.md)
remains an implemented, separate JetClass campaign. It is not the active
scientific campaign in this conversation and does not define PMARD semantics.

The reviewed [HOSD implementation plan](HLT_OFFLINE_STRUCTURE_DISTILLATION_IMPLEMENTATION_PLAN.md)
remains a separate planned campaign. Its donor implementation-status
annotations are intentionally reset: the plan is authoritative for HOSD, but
its HOSD package is not implemented in this repository.

The repository-construction plan remains active through real miniature
acceptance:

```text
../../REPOSITORY_TRANSFER_PLAN.md
```
## HCWDL-MHPE endpoint teacher-mixture add-on

- [Implementation-authoritative 300k/60-pass endpoint mixture plan](HCWDL_MHPE_ENDPOINT_MIX_300K60_PLAN.md)
