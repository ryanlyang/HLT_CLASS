# Embedding bridges, responses and regions: recipe v1, execution v2

Scientific authority: [implementation plan](../plans/LUKA_EMBEDDING_BRIDGE_RESPONSE_REGION_HYDRA_PLAN.md).
Execution and commands: [runbook](../LUKA_EMBEDDING_BRIDGE_HYDRA_RUNBOOK.md).
This namespace is distinct from the older embedding/logit ladders.

## Population and deployment

Exactly 299234 original TRAIN and 74895 development-validation jets; original
M0HLT/OFFLINE recovery anchors. Original50k overlaps development validation.
The retired historical test was repurposed upstream; no independent-test claim
or final-test reader is introduced. All 22 final classifiers use the same
authenticated D000 input and native 128-wide ParT. Auxiliary heads are removed
by exporting only `mod.*`; native reload reproduces selected probabilities.

There are 30 scientific fits: C0, six intermediate assistants, three D000
endpoints, TR, TP, nine R students and nine P students. R0/P0 alias C0.
`registry.py` fixes every row, coefficient, parent and submission order.
Every final student retains original U100 weight5. New losses ramp linearly
over three epochs. No target enters the classification forward path.

## Target meanings

- Assistant banks contain the **prediction** in original standardized U100
  coordinates. They are never independently standardized or substituted for
  native embeddings. Assistant selection minimizes full-validation original
  U100 prediction error, then favors native AUC and earlier updates.
- TR is a single native encoder/classifier across B/K/I/T/IT/KIT. Only KIT
  receives the weight1 original-U100 coordinate anchor. Six vectors share
  TRAIN population mean/std (floor0.001). Differences are formed against B;
  J=IT-I-T in difference space. Each scalar response RMS has floor0.1;
  J uses IT's scale. R9 predicts absolute vectors with the same scales.
- TP shares its encoder/classifier across HLT/OFFLINE and three regional
  scales. Full CE and CE on the mean of valid crop embeddings each have
  weight1. A crop is encoded independently from its physical particles;
  outside particles and full-jet contextual tokens are unavailable to it.
  TRAIN normalization weights sources/scales/nonempty jets/valid regions
  equally. Whole-OFFLINE P9 has separate TRAIN moments. Empty regions are
  masked targets, with compatible dummy forwards on both DDP ranks.
- The HLT axis and pT90 radius define all regions. Coarse/medium/fine have
  2/4/8 nested regions. No particle matching is used for regional targets.
  P8 changes the target source to HLT; descriptors do not encode source.

Full scientific defaults, mathematical reductions and exact row tables remain
in the implementation plan and hashed RECIPE. Finite poor quality is a result,
never a failed qualification or reason to prune a branch.

## Input provenance and storage

TRANSFER/v1 exports both roles' physical HLT/OFFLINE rows, original assignment
arrays, identities, labels and the original frozen U100 checkpoint. Rebuilding
the assignment NPZ must reproduce its original hash; rematching is forbidden.
The export binds original prepared/source proofs, conventions and summaries.

SPEC/v2 requires `native_input_policy`, `native_root`, `native_hash`, and the
native hash as an explicit parent. Source registration freezes donor runtime
at2fb09f38a3a1da85cdceff9df311963fdb3aa872 and allows only additions/modifications
inside the declared campaign runtime. The original scientific donor remains
45075fe6c2d8ccdcf7463cd3a99cbe2fe6976a08; imported numerical code is unchanged.

NATIVE_TRANSFER/v1 contains exactly TRAIN D033/D066 and validation
D033/D066/U100 historical NPZ files. It binds original preparation/cache
proofs, exact source snapshot, ordered file inventory, roles and byte count.
Absent source caches may be reconstructed with original assignments on SPORC;
existing corrupt files are fatal. Every emitted NPZ must match the original
byte hash, and all five populations must match their original input digest.
Publication is completion-last and never writes to the historical cache.

Setup copies D000 and TRAIN U100 from the original bundle, OFFLINE from the
parent teacher-sweep delta, and the remaining historical views from the new
native delta. Per-file hashes, identities, labels and particle offsets are
checked before copying; whole-population fingerprints remain exact.
Only D025/D050/D075 are constructed through the original frontend on Brown.
Cross-platform roundoff is never admitted by relaxing historical hash checks.
Field substitutions use the established K/I/T columns and HLT slot identity.
Indices, source keys and matching metadata never enter model tensors.

Original TRAIN U100 arrays and float64 moments are preserved exactly in new
memory-mapped containers. Validation U100 targets come from the original
checkpoint with those TRAIN moments. A fixed4096 TRAIN witness checks the new
extraction against original targets (absolute/relative tolerance1e-4).
Native D000 FP32 checkpoint replay separately requires AUC difference <=5e-5.

NPY payloads are written privately, flushed/fsynced, exclusively linked to
final names and completed by immutable manifests. Readers validate hashes,
parents, dtype, shape, capability, identities and representation meaning.
Validation target banks are evaluation-only, and technical target banks are
forbidden as scientific parents. They cannot become valid by directory name.
Raw and normalized arrays are processed in bounded chunks or memory maps.

## Artifact inventory

Artifacts use `LUKA_EMBEDDING_BRIDGE_HYDRA_<KIND>/v1`, schema_version1,
except SPEC/v2, schema_version2. All retain canonical content hashes, explicit
parents and inherited evaluation disclosures. DATA/VIEW/PHYSICAL/U100_BANK
parents include the native delta hash; the changed SPEC hash also separates
every downstream result from the failed v1 execution. Old physical transfers
and the scientific recipe retain their original v1 hashes.

| Kind | Required binding / evidence |
| --- | --- |
| RECIPE / SPEC | Complete task registry, original recipe, source, population, seeds, limits, input transfer and byte inventory |
| TRANSFER | Original source/prepared proofs, assignments, physical shards, U100 checkpoint and each payload hash |
| DATA / VIEW / PHYSICAL / U100_BANK | Spec/transfer/prepared/recipe, role, exact input representation, ordered rows and arrays |
| TARGET_BANK | Producer full-checkpoint hash, task, native/predicted meaning, role, technical flag, indices, TRAIN statistics, validity and file hashes |
| EPOCH / TRAINING_REPORT / FIT | Actual bank lineage, coefficients via recipe/update, selection, all metrics, throughput, memory, target coverage, diagnostics, native reload and output files |
| MINI_POPULATION / PREPARATION | Selected real rows, technical capability, CPU preparation time and peak memory |
| PARITY / REPLAY / U100_REPLAY | Factory/control/distributed/probe neutrality and checkpoint reproduction evidence |
| RESOURCES / ACCEPTANCE / CPU_RESOURCES / STARTUP | All GPU/CPU paths, actual timing samples, explicit report extrapolation, memory, per-task walltimes and actual-parent startup checks |
| PLAN / SUBMIT_CLAIM / RUN_CLAIM / TASK | Exact authorized commands, source, accepted job IDs, ownership, measured worker seconds/GPU allocation and complete output inventories |
| RESULTS / PROGRESS | Fresh-C0 comparisons, historical anchors, oracle labels, component metrics, own/ancestor GPU-hours and incomplete states |

Submission also reuses the existing versioned HCWDL exact-DAG ledger/journal
contracts unchanged. Old artifacts do not become this campaign's artifacts.

## Execution and release

GPU jobs: Hydra/gpus/default/normal, one node/task,6CPU,48000MiB,
`gpu:geforce_gtx_1080_ti:2`, exactly two torchrun ranks, FP32 and SyncBatchNorm.
Global batch128 is padded before splitting; local means receive
`world_size*local_rows/global_rows` weighting. Keep tail rows and gather unique
evaluation positions in exact original order. TRAIN samplers are seed-bound.

Setup requests12h and uses genuine production two-rank miniatures for every
row. It checks all six teacher conditions, source transitions, longest support,
partial batches, native reload, validation, diagnostics and serialized target
exports. Full-fit time is measured/extrapolated with 2x safety +30min, capped
at48h. Each card must use <88% physical memory and aggregate host peaks <85%
of requested RAM. No precision/batch/population/epoch adjustment is automatic.

Original-U100 AdamW parity is exact. One/two-rank comparison tolerances are
loss1e-3, relative-gradient5e-4 and buffers5e-5. Diagnostic neutrality uses
loss1e-6, gradients/parameters/moments5e-6 and buffers1e-6 with exact RNG equality.
These are numerical checks, not accuracy gates.

After setup succeeds, its afterok CPU release submits30 fits and an afterany
report. Release uses1CPU/4000MiB/1h; report2CPU/8000MiB/2h, allowing Slurm CPU
rounding. Only true producer dependencies are present. Producers finish only
after outgoing banks are published. Each scientific fit checks its actual
parents and runs a disposable startup before fresh initialization.

Setup additionally measures CPU authentication and full-size reference metrics
in a disposable process. It validates all 30 real miniature receipts and
extrapolates their history-check cost to 30 epochs. CPU estimates use 2x safety
plus ten minutes and <85% of the corresponding requested host memory. The
setup TASK receipt requires this CPU evidence; GPU acceptance alone cannot
authorize release. Optional S2IT/S6 historical comparisons require an
authenticated selective-campaign spec and frozen prediction/report references;
they never replace fresh C0 or become optimization targets.

Source must be clean, exact and pushed, with only additive runtime in this
namespace since45075fe6c2d8ccdcf7463cd3a99cbe2fe6976a08. Submission requires a
full scheduler dry review and one bounded authorization. Claims and journals
are exclusive; ambiguous partial submission never triggers automatic retry.
Failed dependencies become terminal so the report can complete.

Local tests establish readiness to queue setup. Real Hydra qualification is
still required before the release may submit scientific fits.
