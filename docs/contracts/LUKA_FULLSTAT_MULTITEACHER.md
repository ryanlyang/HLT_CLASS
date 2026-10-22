# LUKA_FULLSTAT_MULTITEACHER, v1

Authority: [scientific plan](../plans/LUKA_FULLSTAT_MULTITEACHER_PLAN.md).
All new artifacts use `LUKA_FULLSTAT_MULTITEACHER_<kind>/v1`, schema1,
canonical content hashes, explicit parents and the existing sealed FullStat flags.
No old bank, preflight, training report or source is relabelled.

| Kind | Required binding |
| --- | --- |
| REUSE | Exact completed original U100-w5 specification; replay in its clean pinned source; original four completed teacher fits, two target banks and reference predictions |
| RECIPE / TARGET_RECIPE | Teacher-coordinate mapping, four independent heads/seeds,1.25 each, TRAIN normalization, native deployment, FullStat20 optimizer/schedule |
| SPEC | Pushed source snapshot, REUSE hash, population, recipe, teachers, reference, two-job DAG, output root and resource/admission policy |
| BANK | Teacher identity, selected checkpoint/report, prepared cache, target recipe, exact TRAIN rows/width128, atomic targets payload |
| BUNDLE | SPEC plus all four bank hashes; original/new manifest file refs; fixed teacher order and prepared TRAIN population |
| PLAN / SUBMISSION_CLAIM | Exact SPEC, canonical dry-reviewed command plan and exclusive live claim |
| SETUP | SPEC/BUNDLE, exact accepted setup job, successful probes, finite measured resources/timings, derived training estimate and immutable outputs |
| TRAINING_REPORT / FIT | Recipe/prepared/BUNDLE, all20 passes and losses, selection, native-only weights and exact probability readback |
| RESULTS / TASK | SPEC/SETUP, authenticated reference predictions, selected fit, exact submitted job and all output file hashes |

BANK arrays are exactly identities uint8[N,32], labels int64[N], targets
float32[N,128], mean/scale float64[128]. Reject nonfinite values, nonunique or
misordered identities, changed labels, scale below1e-3, wrong role/coordinate,
teacher, checkpoint, recipe, width or payload. Reused OFFLINE/U100 banks retain
their original loaders/contracts. BUNDLE preserves their authenticated identities.

Teacher export is TRAIN-only. Student cache access is TRAIN/validation D000;
the only extra allowed TRAIN coordinates are the four registered teacher views.
Inference checkpoints contain no `predictors.*` keys and must strictly load into
native ParT. Target/prediction dictionaries must contain exactly four teacher keys.
Per-teacher SmoothL1 is mean-reduced independently; no hidden concatenation or
double division by four. Selected checkpoint excludes all auxiliary parameters.

Authentication uses original U100 commit
`7ea8b0c693734a26ced3820354e047f4af26c979`; byte-compare imported donor runtime.
The compatibility inventory explicitly excludes six unrelated SPORC trial D066
entrypoints/modules modified since that commit; none is imported by this path.
Every other pre-existing donor runtime file is compared, including transitive
FullStat kernels. New execution must add only its isolated runtime relative to baseline
`118aa375eea423e5e99338cd9a6eda198225ef21`. Replayed source/runtime or artifact
changes fail closed. Do not repair mismatches by editing receipts or old outputs.

Setup and science use immutable attempt directories. No fit can run without the
exact accepted setup job's COMPLETED/0:0 state, owner/account, receipt and measured
envelope. Recompute the estimate from saved measurements. Reject unmeasured,
nonfinite or excessive resource evidence. Training quality is never an admission
criterion. Submission failures preserve journals; do not blindly resubmit.
