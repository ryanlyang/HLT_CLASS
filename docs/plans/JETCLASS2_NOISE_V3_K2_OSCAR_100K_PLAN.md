# Frozen NOISE_V3 K2 concatenation on Oscar, 100k/50k

## 2026-10-04 operational recovery amendment

User authorized targeted recovery after GPU ECC failures on gpu3001 and a
separate missing-Python-files failure. This amendment permits reuse within
this exact OSCAR experiment, not the cross-dataset imports forbidden below.
The original root, failed outputs, receipts, ledgers and source checkout stay
read-only. A sibling recovery root owns replacement jobs/results. Authenticate
all original receipts and output bytes, exact journals, terminal scheduler
states, original native acceptance and the restored software fingerprint.
Only incomplete science tasks may restart, from zero, with unchanged resources,
population, models, seeds, matching, training, inference and reporting.

Preserve the seven completed science tasks reported through D050; retry the
two failed CE controls, D050 reducer, D025/D000 fits and reducers, x1 compression,
aggregation and completion (ten jobs if that state still authenticates).
Dependencies refer only to replacement jobs; completed original parents are
artifact dependencies, never aged-out Slurm dependencies. Exclude gpu3001
from all replacement GPU allocations. Check CUDA health before cache loading.
Use the original source's scientific modules byte-for-byte, with new explicit
I/O/lineage orchestration under the separate RECOVERY v1 contracts. A full
replacement dry run and exact explicit authorization precede live submission.
No cancellation, environment repair, source-root mutation, silent partial-task
resume, changed GPU family, or new final-test capability is authorized.
See [recovery contract](../contracts/JETCLASS2_NOISE_V3_K2_RECOVERY.md).

## Authority and scope

Authorized by the 2026-10-04 request. This is a new experiment, not a
continuation of SPORC training. The relocated NOISE_V3 consumer handoff is the
data authority. The original K2 plan supplies the capacity-two objective,
fixed-support construction and training recipe. This plan supersedes their
native-HLT dataset, population and execution assumptions only for this family.

Use the frozen manifest
`26c902d04988b3f32be98a8d2ccbed6d48dd8c3f49c71ddc2c660da0f2ec5deb`.
Never regenerate the proxy, access native `hlt_part_*`, rewrite producer JSON,
or treat NOISE_V3 as a fitted/certified CMS detector response. Tracking is mm.
The read-only relocated consumer remains independently owned and unchanged.

## Population and views

Register exactly 100,000 of the actual 1M training rows and 50,000 of the
actual 250k validation rows. Class-stratified SHA256 ranking uses only canonical
identity, role and manifest hash. Natural-proportion Hamilton quotas with a
minimum of one per class are frozen before matching. Scalar labels are selection
metadata/targets, never inputs. Restore shard/raw-entry order for caching; use
the original epoch sampler for fitting. Preserve sealed 1M final-test metadata;
no worker has a final-test particle capability.

Validation is per-class hash ordered 50/25/25 checkpoint/diagnostic/report.
Checkpoint rows alone select epochs; report rows alone compare recovery.

The user explicitly confirmed retaining the existing K2 formula. Supply the
original K2 campaign spec and its externally recorded raw SHA256 (formula only).
The copy's embedded content hashes and matcher definition must validate. There is no
implicit winner, new performance screen or imported assignment/checkpoint.
Freeze positive integer salience on complete native offline and proxy endpoints.
Keep the top min(Noffline, 2*Nproxy) offline entries before matching. Apply the
existing exact capacity-two objective, including ordered integer tie-breakers.
Unknown category remains unknown; explicit tracking validity is preserved.

Each proxy particle owns three active slots: original, rich1, rich2. Empty rich
slots contain active copies of their owner. D100 is retained offline + all
original proxy particles + fillers. D075/D050/D025 interpolate rich contents;
identity, charge applicability and differing tracking-validity groups switch
atomically with deterministic nested thresholds. D000 is exactly proxy x3.
HLT_X1 and D000 require neither offline nor maps nor identities in preprocessing.
Rebuild all 17 features from the resulting set. No source/slot/ancestry features.
The physical-schema adapter is separately versioned, not disguised as native
Delphes raw arrays. Derive padded capacity from selected ordinary-role maxima;
never silently truncate. Report overflow, cropped pT/salience, fillers, occupancy,
angular distances and per-class differences. Poor metrics never stop descendants.

## Fits

```
D100 CE -> D075 KD -> D050 KD -> D025 KD -> D000 KD -> HLT_X1_COMPRESSED KD
```

Also HLT_X1_CE, HLT_X3_CE, OFFLINE_CE, and DIRECT_HLT_X3_KD from D100.
Ten cold single-encoder fits, five training-bank reducers, aggregate, complete.
No learned fusion, seed ensemble, U ladder, K3, warm start or shortened pilot.
Reuse the original paired initialization/sampler seed domains, batch128,
inference128, AdamW, BF16 forward/FP32 loss, C25P75/T2 (including T-squared KL).
Warmup1-3, hold through45, cosine to1.5e-5 at60, floor through100. Minimum60,
patience15 starting at60, significant AUC delta5e-5; earliest stop75. Restore
best checkpoint with original AUC/CE/R50/update tie ordering.

## Execution and acceptance

New `NOISE_V3_K2_*/v1` artifacts and fresh output roots outside data/checkout.
Use Oscar batch CPU preparation and the repository's Oscar L40S gpu/norm-gpu
profile, not SPORC debug/tier3. Initial bounds: GPU6CPU/90000MiB, fit24h,
preflight12h, reducers6h. CPU metadata1CPU/8192MiB/4h, assignments1CPU/8192MiB/12h.
Submission uses actual-site `sbatch --test-only`; bounds are not measured claims.
If the full100-pass conservative projection exceeds23h or memory fails, stop
for a reviewed resource change rather than truncate training or alter batching.
The old segmented SPORC continuation is not ported implicitly.

Stages: scalar population selection -> independent ordinary-shard K2 matching
and bounded exhaustive acceptance -> foundation -> real Oscar GPU preflight
-> authorized science. The full canonical dry run is mandatory. Gate-only
submission may optionally authorize automatic science after gate completion;
otherwise science requires a separate explicit command. Exact-ID journals,
immutable output receipts, source/parent/byte authentication and exclusive
submission claims prevent accidental duplicate/ambiguous launches.

Native acceptance includes relocated paired reads, no-offline deployment parity,
installed-Weaver/native-wrapper and FP32/BF16 storage parity, longest-population
batch128 stress with AdamW updates, actual CE/KD mini-fits using the production
kernel, selected checkpoint and teacher bank round trips, full selected caches,
environment/GPU fingerprints, CPU/GPU peaks and conservative runtime projection.
Retain the K2 pair saved-tensor CPU offload; no global model patch or fallback.
GPU peak at most 90%, RSS at most 80%, cache budget <65%, positive finite timings, free-space
audit and genuine native evidence required. CPU test doubles cannot admit science.

No existing jobs are cancelled or submitted by local implementation. Source must
be clean, pushed and exact; remote reader/GPU acceptance remains external.

## Implemented interfaces and queue inputs

Implementation: `src/hlt_classification/noise_k2/`, with
`scripts/noise_k2.py`, `scripts/queue_noise_k2.sh` and
`sbatch/run_noise_k2.sh`. See the [v1 contract](../contracts/JETCLASS2_NOISE_V3_K2_OSCAR.md).
CLI modes: create, submit, run, gate, results, monitor. There is no final-test,
producer, arbitrary candidate, old-map reuse or old-weight reuse command.

Before queueing, include the independently developed relocated reader files
in the pushed source, along with this campaign. The old full K2 formula source
is the original SPORC JSON, for example the campaign previously registered at
`/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_dzfix_concat_k2_96hc891da0d_r1/campaign_spec.json`.
Record its raw `sha256sum` there; copy that JSON unchanged onto OSCAR. Neither
this document nor the new code guesses which salience candidate won. Only the
formula and original hashes are reused. No dependency on a SPORC job is needed.

The queue helper requires these environment variables:

- `PROJECT_DIR`: new exact clean, pushed OSCAR checkout;
- `NOISE_K2_COMMIT`: its full commit;
- `CAMPAIGN_ROOT`: a fresh separate OSCAR output directory;
- `K2_FORMULA_SPEC`: local OSCAR path to the unchanged donor JSON;
- `K2_FORMULA_SHA256`: raw SHA256 recorded on SPORC.

`bash "${PROJECT_DIR}/scripts/queue_noise_k2.sh"` creates and authenticates the
registration and dry runs only. Adding `--execute` authorizes fresh preparation
and native GPU gates, then automatic submission of the17 science tasks after
all checks pass. No jobs are canceled, and the helper is idempotent only after
a complete submission ledger exists. An ambiguous partial submission requires
an exact-ID journal review, not deleting a claim and retrying blindly.

For manual gate-only operation use the CLI `submit --stage gate --execute`
with the exact authorization phrase, without `--auto-science`. Once admitted,
`submit --stage science --execute` with the phrase submits the scientific DAG.
`results --spec ...` displays completed rows before the full ladder finishes.
All displayed metrics use the validation report subset, never final test.
