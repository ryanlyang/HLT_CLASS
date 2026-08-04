# PRAD Requirements Traceability

This ledger maps `prad_implementation_agent_prompt.md` to executable repository
surfaces and distinguishes local implementation evidence from scientific
campaign evidence. “Implemented” never means a real-data experiment has run.

## Scientific and data requirements

| Prompt sections | Repository surface | Local evidence | Real-campaign state |
|---|---|---|---|
| 1, 5--7: HLT-only PRAD graph | `models/prad_particle_transformer.py`, `prad/relation.py` | symmetry, masked centering, bounded projection, zero-gate parity, gradient, public-signature tests | installed-Weaver CUDA PRAD attestation is a mandatory smoke task |
| 2: source/data audit | active implementation plan, `prad/audit.py`, `audit_prad_data.py` | schema-capability and synthetic audit tests | real `reports/data_audit.md` pending Tigris data |
| 3: fixed split | `prad/splits.py`, `build_prad_splits.py` | exact constants, capacity failure, proportional class allocation, disjoint identities, checksums | 500k/150k/500k manifests pending real inventory |
| 4: HLT baseline | `reference_engine.py`, E0 registry and worker | training/resume/inference mechanics tested | full E0 reproduction pending |
| 8: structural targets | `targets.py`, `artifacts.py`, `streaming.py`, `build_prad_targets.py` | exclusive C/A K=2/3/4 plus task-local and in-memory reconstruction tests | full targets are recomputed per consuming task; vertex coefficient is zero because source has no vertex field |
| 9: association | `matching.py`, paired/target cache builders | charged/neutral Hungarian gates, cost rejection, pair-mask tests | real coverage tables pending audit |
| 10: teacher | `teacher_engine.py`, `cache_prad_teacher_outputs.py` | attention-connected relation graph, semantic validation, selection/freeze and compact/dense output-cache tests | E1 checkpoint and val/test metrics pending |
| 11--12: losses and schedule | `losses.py`, `training.py`, `engine.py` | normalized masked losses, reliability weights, KD, staged freeze/LR behavior, exact resume | full fixed-budget runs pending |
| 13: oracle | explicit `forward_oracle`, E2 | oracle/offline path is separated from public inference and payload follows trimming | E2 headroom result pending |

The source capability audit found no physical event ID, direct particle link,
vertex assignment, or original jet radius. The registered split therefore uses
the strongest stable jet identity available, cannot claim event-overlap proof,
uses the specified Hungarian fallback, masks vertex supervision with coefficient
zero, and does not invent a jet radius. These adaptations are recorded in the
active implementation plan and real audit report template.

## Experiments, evaluation, and artifacts

| Prompt sections | Repository surface | Status |
|---|---|---|
| 14: E0--E10 | immutable `CORE_EXPERIMENTS`; one student/reference/teacher implementation | implemented and registry-tested |
| 15: V1--V10 | `experiment_variant` switches, including V7/V8/V9 grids | implemented and registry-tested |
| 16: order and five seeds | campaign DAG, validation-only selection, `{11,22,33,44,55}` confirmation | implemented; execution pending |
| 17: metrics | `evaluation.py`, `inference.py`, benchmarks, diagnostic histories | implemented and edge-case tested; numerical results pending |
| 18: statistics | `reporting.py`, paired identity bootstrap and seed pairing for every final five-seed graph | implemented and tested; sealed predictions pending |
| 20: reusable modules | `src/hlt_classification/prad/` with thin `scripts/` and Slurm workers | implemented |
| 21--22: tracking/report | run reports, keyed prediction shards, CSV/report/plot assembler | implemented; production outputs pending |
| 23: acceptance | fail-closed campaign locks and evidence gates | only code-level criteria are currently accepted |

Every run report records resolved graph configuration (including the versioned
in-batch E10 shuffle), seed, source/split/cache lineage, parameter count,
checkpoint selection, histories, and accumulated
training time. Evaluation adds keyed predictions, metrics, relation diagnostics,
gate values, matching coverage, throughput, latency, and peak memory. Dense
teacher relation and centered-bias outputs can be stored as validated float16
shards with `--dense-pairs`; the default campaign streams/recomputes them to
remain within its reviewed storage model. The minimum-storage profile also
keeps paired views, compact targets, and teacher jet outputs in Slurm job-local
temporary storage. Completed runs retain model-only best/final checkpoints;
only incomplete runs retain one rolling exact-resume checkpoint.

## Safeguard coverage

The 15 explicit safeguards in prompt section 19 are covered as follows:

1. exact split counts and capacity: `test_prad_foundation.py`;
2. identity and available-event disjointness: `test_prad_foundation.py`;
3. paired label agreement: cache/audit validation tests;
4. unmatched/self pair masking: `test_prad_foundation.py`;
5. relation symmetry, centering, padding, and bounds: foundation and model tests;
6. zero-gate logits below `1e-6`: model test and installed-runtime preflight;
7. class-gradient flow at nonzero gate: model/runtime tests;
8. frozen teacher gradients: training/runtime tests;
9. HLT-only inference: model and inference tests;
10. E10 relation targets are deterministically deranged within the realized
    batch without preserving class, while hard labels remain unchanged:
    training tests and the versioned training configuration;
11. sealed test entry and exact-lineage claim recovery after preemption:
    inference/campaign tests;
12. optimizer, scheduler, scaler, epoch/update, RNG, replica cycle, history, and
    configuration resume: checkpoint/training tests.

Items are grouped where a single test exercises multiple numbered safeguards.
Corruption, stale-source, lineage, nonfinite, storage, partial-submit, recovery,
exact array-attestation identity, and exact-ID cancellation failures are
additionally tested.

## Acceptance status

| Acceptance criterion | State | Evidence needed to close |
|---|---|---|
| Exact full split exists | Pending external execution | real JetClass inventory and immutable manifests |
| Baseline reproduced | Pending external execution | full E0 training/evaluation |
| Teacher trained/frozen | Pending external execution | selected E1 checkpoint and output artifacts |
| Oracle completed | Pending external execution | E2 validation result |
| Main architecture implemented | Locally passed | current tests; v3 Stage-A/Stage-B CUDA runtime attestation still required for production readiness |
| HLT-only inference | Locally passed | also rechecked in every real evaluation worker |
| Core causal ablations run | Pending external execution | E3--E10 full-data rows |
| Priority variants evaluated | Pending oracle/production execution | registered variant rows selected by declared policy |
| Five-seed finalists | Pending external execution | validation-selected confirmation rows |
| Untouched test evaluated after selection | Pending and sealed | finalist/execution locks and single final wave |
| Code/config/commands/artifacts/report | Code and commands implemented | checkpoints, metrics, plots, and final report require campaign results |
| Causal scientific conclusion | Pending external execution | completed controls and statistical report |

## Required next evidence

The next authorized step is not a full campaign. Commit and transfer the exact
clean source, run the installed-Weaver parity and CUDA PRAD-runtime checks, dry
run the complete smoke DAG, then submit and monitor only the genuine Tigris
miniature. Production requires reviewed `sacct` resource measurements, storage
headroom, a second dry run, and explicit authorization. The exact commands are
in `docs/PRAD_RUNBOOK.md`.
