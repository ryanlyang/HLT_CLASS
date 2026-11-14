# FullStat balanced ladder contracts v1

Authority: [scientific plan](../plans/LUKA_FULLSTAT_BALANCED_LADDER_PLAN.md).
Implementation: `src/hlt_classification/luka_fullstat_balanced_ladder/`.
This namespace does not change historical campaign schemas or sampling.

## Population and optimization

Keep original 3M TRAIN / 500k development validation / 1.5M sealed final test /
105,325 unused roles. Select TRAIN using saved offline reconstructed jet pT:
`200 <= jet_pt < 2000`, 18 half-open 100 GeV bins, all 11 classes. Freeze
2,425,536 eligible identities and the 198-cell occupancy from audit hash
`8383cbacb0aeb7664a534ab5e8f3287b8e67c4963dbe69e7c90146786c25522c`.
Do not expand sparse cells, merge bins, change roles, or infer counts from draws.

Each epoch draws 2,425,536 rows independently with replacement: uniform cell,
then uniform row in that cell. This is uniform class and uniform conditional
bin. It gives equal probabilities, not exact equal realized counts. A separate
PCG64 generator uses `SeedSequence([2026111301, epoch])`; all six fits share the
same draw order. Save draw SHA256, actual cell counts, unique exposure and
repeated draws each epoch. No loss weighting duplicates the sampling weights.

Twenty epochs, batch128, final batch64, 18,950 updates per epoch, 379,000 total.
Native FullStat AdamW, weight-decay exclusions and warmup2/cosine20 schedule:
peak3e-4, floor1.5e-5, betas0.9/0.999, eps1e-8, weight decay0.01. BF16 native
forwards; FP32 objectives. No early stopping, accumulation or warm starts.
Common fresh native initialization seed2026111302. Independently seeded heads
do not consume the native RNG stream; see `contracts.seed` for exact domains.

## Six fits

| Stage | Input | Embedding and auxiliary-logit teachers | Additional objective |
| --- | --- | --- | --- |
| OFFLINE | Pure offline | None | None |
| M0HLT | D000 HLT | None | None |
| CONCAT | Full source-blind HLT+offline union | None | None |
| UE100 | U100 | OFFLINE, CONCAT, M0HLT | None |
| DE050 | Exact D050, half offline contribution | OFFLINE, CONCAT, UE100, M0HLT | None |
| DE000 | D000 HLT | OFFLINE, CONCAT, UE100, DE050 | OFFLINE instance matching |

Each baseline uses CE1. Each ladder rung uses CE1 + total embedding10 + total
auxiliary KD1, split equally across its teachers. SmoothL1 beta1 means over
rows/dimensions. KD is `T² KL(q_teacher || p_auxiliary)` at T2, class sum/batch
mean. Independent 128→128→128 GELU embedding heads and 128→11 logit heads share
the encoder. The main classifier has direct CE supervision only. Teacher
outputs and predicted targets never enter that classifier. M0HLT and DE000
deploy as native HLT-only ParT; intermediate inputs remain explicitly privileged.

CONCAT preserves every constituent, including duplicates. Its transform/order
does not expose source labels. D050 reuses authenticated original assignments;
no rematching, support truncation or new degradation source inputs. Native D000
bytes are replayed when reconstructing D050.

Only this campaign's selected native teacher checkpoints supply targets. Export
native embeddings with FP32 forward, batch128; mean/std use every unique
eligible TRAIN row once, FP64 moments, std floor0.001. Standardized values are
stored FP32. Native T2 probabilities use BF16 forward/FP32 softmax. No auxiliary
teacher head, sampled repeat frequency or validation row enters these banks.

DE000 alone adds contrastive0.3, linearly ramped over three new epochs, tau0.1,
one positive and 1024 distinct negatives: hard256/same-class256/global512.
Keep donor class-balanced hard proposals4096 and ranks257–1024, canonical tie
ordering and positive/available-alias exclusion. The negative bank is unique
eligible TRAIN; it does not inherit the query pT flattening. Query occurrences
use a versioned identity/epoch/draw-position/component/stream RNG (seed2026111304),
so repeated identities can have distinct candidates. Three bounded CPU workers,
lookahead and eight-query FP32 GPU ranking preserve serial candidates.

## Artifacts and publication

All metadata names start `LUKA_FULLSTAT_BALANCED_LADDER_<KIND>/v1`, schema1,
canonical SHA256 and parent bindings. Common scope is development validation;
final test sealed, event independence and old-trial overlap not verified.

| Kind | Purpose / critical parents |
| --- | --- |
| SOURCE, INPUTS, RECIPE | Exact pushed source including JSON semantics; original population/cache provenance; explicit new science |
| POPULATION | Input/prepared hashes; original scalar TRAIN/validation rows, eligible mask and frozen occupancy |
| SPEC, PLAN | Source, population, six fresh nodes, resources and exact remaining DAG |
| VIEW_BUILD, FIT_BINDING | Population/coordinate and exact fresh teacher receipt parents |
| STARTUP_PHASE, STARTUP | Real allocation parity, gradients, support, timings and memory; never local acceptance substitutes |
| EPOCH, EPOCH_COMMIT | Contiguous epoch report plus full student/AdamW/CPU/CUDA RNG state, draw hash, update cursor |
| TRAINING_REPORT | All20 epochs, fixed selection rule, restored selected weights |
| BANK | Same population, selected native checkpoint/report, temperature and unique-TRAIN normalization, separately hashed arrays |
| METRICS | Same selected checkpoint over natural/in-range/outside/per-bin validation; R50 and R30 passing counts |
| TASK_RECEIPT | Job binding, complete result inventory, epoch references and teacher lineage; completion marker written last |
| LAUNCH_INTENT/APPROVAL/COMPLETE, CLAIM, RECOVERY_CLAIM, FAILURE | Explicit bounded launch, immutable review/claims and failure evidence |
| RESULTS | Fresh balanced M0HLT/OFFLINE anchors per metric and evaluation population |

Use immutable atomic files; no pickle in array banks, no memory-mapped writable
scratch exports. Validate payload bytes, shape/dtype/finiteness and ordered
identity/label joins before reuse. Raw ROOT and teacher stores remain read-only.
Epoch checkpoints retain all auxiliary heads and optimizer state; `selected.pt`
contains native weights only. Artifact existence alone never proves completion.
Task publication binds `fit_binding.json` in its file inventory. Reuse verifies
the exact teacher receipt hashes, integrated acceptance flags, selected epoch
and all twenty committed epoch references before accepting the selected result.

Primary selection uses original natural500k validation: maximum macro OVR AUC,
then minimum CE, maximum macro log R50, earliest update. No class/pT-specific
reselection. R30/R50 use `p(signal)/(p(signal)+p(QCD))`, descending signal rank
with ties included. Zero QCD passing is censored; missing classes are unresolved.
Recovery is `100*(model-M0HLT)/(OFFLINE-M0HLT)` using new balanced anchors on the
same population. A zero or negative anchor gap is unresolved.

## Execution and explicit recovery

Six science allocations, zero separate setup/preflight jobs. First three are
independent; each later rung has `afterok` dependencies on all its teacher fits.
Requests: one L40S, six CPUs, 192000 MiB RAM and48h per stage. These are declared
ceilings, not new measured runtimes. Before fresh fitting, integrated acceptance
measures full-support gradients,64 complete balanced updates,16 validation and16
dual-export forwards. Admission includes remaining epochs, selected evaluation,
outgoing export,1.3 timing factor,60min fixed margin and CPU/GPU memory headroom.
Save measurements before refusal. Discard probes and reset fresh initialization.

Allow32GiB result artifacts plus4GiB headroom; require36GiB available quota at
launch. No automatic quota cleanup. Save all epoch states. Source/runtime and
scheduler identity must match; use the existing verified Oscar repair environment.
The launcher checks actual module imports before claiming a campaign.

After a terminal failure, a new recovery root can reuse successful completed
stages, resume failed stages at the last committed epoch, and queue downstream
stages. All prior jobs must be terminal; cancel only explicitly identified pending
descendants if needed. Uncommitted partial epoch files are not trusted. No repeated
updates when all20 epochs committed: only selected validation/export/publication.
Runtime bytes, science, draw law and resource registration remain identical for
this recovery path. Changed scientific recipes or resource envelopes require an
explicit new implementation/registration. Preserve prior roots and claims.
