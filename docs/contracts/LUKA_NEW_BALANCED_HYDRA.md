# New balanced FullSim Hydra contracts v1

Authority: [scientific plan](../plans/LUKA_NEW_BALANCED_HYDRA_PLAN.md).
Namespace: `src/hlt_classification/luka_new_balanced_hydra/`.
Artifacts are `LUKA_NEW_BALANCED_HYDRA_<KIND>/v1`, schema 1, canonical SHA256,
explicit parents, sealed final test and unverified event independence/old overlap.
They are not interchangeable with original FullStat or natural-baseline artifacts.

## Population

Parent foundation hash `2f05adcca889288e705a69fc1e54670f1713d29ec16ab75c4f694ab7b5a23d02`
is fixed. Parent memberships and producer bytes are authenticated and replayed.
Only role 3 rows satisfying `200 <= offline jet_pt < 2000` can change roles.
Role 0/1/2 rows and quarantined role 255 rows retain their assignment. Add exactly
279,164/136,976/113,361 unused rows to TRAIN/validation/test. Sequential
largest-remainder quotas operate on source-stratum/class/100-GeV-bin capacities;
SHA256 ordering of original 2048-row blocks uses seed 2026111305 and a new domain.
Ties use file index and original entry. Replayed role arrays must match exactly.

POPULATION binds relative per-file extended-role payloads, canonical ordinary
identities/labels/pT, counts, 198-cell occupancy, particle budgets, full-support
maxima, quotas, policy and parent hash. Counts are 2,425,536 TRAIN, 500,000
validation, 1,200,000 final test and 689 unused in range. Outside-role counts are
853,628/136,976/413,361/205,760. No test/unused identity arrays are exported as
training targets. Scalar membership inspection is not test inference.

No class or bin may be empty. Sampling uses the original balanced PCG64 stream:
uniform cell then uniform row with replacement, 2,425,536 draws per epoch,
20 epochs, batch 128 including final 64. All fits share draw identities/order;
validation uses every row exactly once. The frozen new recipe includes the
Oscar seed domains, losses, schedule and native model configuration by value.

## Inputs and targets

Only new eligible TRAIN, in-range validation and original outside validation
particle slices are requested. Requested intervals never cross sealed test,
remaining-unused or excluded TRAIN rows. ROOT may internally decompress a
larger basket. D000 reads HLT branches only. Source SHA256 is checked before
construction and after all bounded futures finish. All resident views join
against the authenticated canonical identity/label order. No persistent
particle caches, truncation, source tags, pT-selection features or mapping
indices enter models. The existing provisional mm/error-only tracking
convention is retained; producer confirmation is not newly claimed.

New U100 and D050 assignments use the same existing deterministic physical
matcher. VIEW_BUILD binds ordered identity/mapping SHA256 per chunk and role;
DE050 must replay the UE100 mapping hashes. CONCAT retains the full physical
multiset including duplicates. Outside diagnostics use the same model and
transform as primary validation. Maximum support is measured separately.

BANK binds selected native checkpoint/report, spec, population, recipe,
coordinate, unique TRAIN order, precision and normalization. Embeddings are
native FP32; targets use FP64 unique-TRAIN moments with std floor 0.001. Native
logit probabilities are BF16-forward/FP32-softmax at T = 2. Validate every payload,
shape, dtype, finite value, probability sum and canonical identity/label join.
Banks cannot be loaded as validation/test targets. Only the five outgoing
teachers export banks. Models are fresh; no historical checkpoint substitutes.

## Selection, metrics and recovery

TRAINING_REPORT selects maximum macro AUC, then minimum CE, maximum macro
logR50, earliest update, on **in-range validation only**. METRICS reports that
same checkpoint on primary in-range, original outside-range and 18 in-range
bins. No subset-specific selection. R30/R50 use signal/(signal+QCD), descending
signal ranks, ties included, passing counts and censored zero backgrounds.
RESULTS uses new balanced M0HLT/OFFLINE anchors on the identical subset; a
nonpositive gap or censored/missing metric yields unresolved recovery.
The sealed primary and extrapolation final tests have no worker/CLI operation.

Each EPOCH_COMMIT is written after its immutable full model/auxiliary heads,
AdamW state, CPU/CUDA RNG, cursor and draw-order hash. Canonical epoch markers
are separate from per-job attempt payloads. Only contiguous commits with exact
spec, stage, job and teacher receipt bindings can resume. Incomplete payloads
remain preserved. A completed 20-epoch fit repeats integrated checks and
publication, with no additional scientific training updates.

RESULT is written last and binds all 20 committed reports, selected native
weights, both prediction payloads, metrics, integrated acceptance, view proof,
and optional teacher bank. Completion reuse recursively validates teachers,
not just path existence. Metadata-only progress explicitly does not claim
checkpoint/prediction verification.

## Execution

SPEC fixes exact clean pushed source, installed runtime and split producer,
L40S/L40/A6000 choice, 192000 MiB RAM, 6 CPUs, 4 bounded preparation workers and
3 contrastive workers. Six science jobs, no separate setup/preflight job.
Every allocation verifies Hydra, exact job, account, partition, QOS, GPU,
memory, CPUs, workdir, comment, runtime and walltime. Workers set user-site
isolation and prepend the environment library directory.

Inside each job: installed-Weaver FP32 parity, full objective two-epoch
miniature/reload, direct-CE classifier gradients, serial/optimized matching
parity, longest-support stresses including outside validation, 64 representative
timed updates, 16 validation and 16 dual-export probes. Discard probe state.
Timing includes remaining training, primary and outside evaluation, outgoing
export, 1.3 factor and 60 min margin. Record measurements before refusing an
insufficient allocation. Model quality never gates continuation or success.

New results budget 14 GiB includes all 120 epoch states, native models, five
banks, ordinary prediction payloads and headroom; particle views are RAM-only.
Reserve the natural baseline's remaining cache/results footprint up to 17 GiB.
The operator declares current free quota and filesystem availability is checked
at registration, submission and worker entry. Neither unrelated jobs nor their
storage are changed. Only committed outputs reduce the new campaign's remaining
reservation; abandoned partial attempts receive no quota credit. No automatic
cleanup is performed.

PLAN prints all six requests and performs `sbatch --test-only` before submitting.
Each actual call has an immutable intent then exact-ID receipt. Dependencies
are afterok on all relevant teachers; invalid dependencies are cancelled by
Slurm. Explicit resume requires all prior exact IDs terminal, verifies successful
completed stages, and queues only unpublished stages with committed continuation.
Ambiguous sbatch responses require reconciliation, never automatic retry.
