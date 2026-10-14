# Luka regular campaign and durable inputs v1

Authority: `../plans/LUKA_FULLSIM_REGULAR_PLAN.md`.

New namespace `LUKA_FULLSIM_REGULAR_{GATE,CACHE,SPEC,PREFLIGHT,PLAN,TASK,RESULTS,
SUBMISSION_CLAIM}/v1`, canonical content hashes, parent hashes and schema 1.
Original stage 1/2 contracts and kernels are reused, not redefined. New durable
cache manifest binds prepared-v1, exact source and every ordinary-role block;
it cannot be substituted for a legacy RAM-only preparation by filename alone.

Each `(role, coordinate, file_index)` payload contains only `offsets` int64
shape(N+1), `features` float32(P,17), `vectors` float32(P,4), `identities` uint8
(N,32), `labels` int64(N). Offsets begin at 0, end at P, and increase by 1..512; features
and vectors are finite. Identities and labels equal the selected membership in
sorted-entry order. Hash the ordered identity + little-endian int64(label,Npart)
+ float32 features + vectors exactly as stage 2 ViewMeter. No mask is persisted:
unchanged RamCache derives the exact Boolean mask and minimum 16 padding.

Authentication checks contract, source, parent, canonical location and all
referenced payload sizes/SHA256 before using arrays. Roles are exactly TRAIN
and validation; final-test/unused access is rejected before opening payloads.
Unknown coordinates, duplicate/missing/reordered blocks, corrupt arrays, changed
fingerprints or changed source fail closed. Partial outputs have no completion
manifest and are not reusable. Output paths cannot overlap source/raw inputs.

Source-bound GPU evidence includes genuine Slurm identity, installed environment,
GPU identity, parity, worst-support stress, full-population technical reports,
checkpoint/bank readbacks, measured peaks and conservatively projected requests.
Admission requires successful exact gate accounting and complete payload hashes.
Scientific checkpoints and TRAIN-only teacher banks bind spec, selected report,
row order and dependencies. No technical checkpoint can satisfy a science edge.

Dry/live plans contain the complete afterok DAG. A live call must match the
saved reviewed plan and canonical dry ledger; sanitize inherited SBATCH/SLURM
options. Exclusively claim submission and journal each accepted exact job ID.
Ambiguous submission or failed workers require explicit inspection, not retry.

`historical_test_evaluated=true` records the consumed prior holdout;
`final_test_accessed=false` is the capability/evidence for this campaign only.
