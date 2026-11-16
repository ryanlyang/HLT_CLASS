# New balanced Oscar main-KD contracts v1

Authority: [scientific plan](../plans/LUKA_NEW_BALANCED_OSCAR_PLAN.md).
Namespace: `src/hlt_classification/luka_new_balanced_oscar/`.
All new execution/training artifacts use `LUKA_NEW_BALANCED_OSCAR_<KIND>/v1`,
schema 1, canonical SHA256, explicit parents, sealed test and development-only
evaluation. They cannot be reused as Hydra or old-data single-DE000 artifacts.

## Shared population, different recipe

The original `LUKA_NEW_BALANCED_HYDRA_POPULATION/v1` is deliberately retained.
Its exact frozen hash is
`7f2413152ecb588764c862b4b83bd6f900320a1b915a3b77f95540dd6e283939`.
The unchanged donor builder/load/role-mask functions verify the parent, replay
the unused-only extension and authenticate ordered rows and payloads. Renaming
its hash domain would silently change the extension assignments and is forbidden.
The [shared population contract](LUKA_NEW_BALANCED_HYDRA.md) still governs roles,
excluded label 206, pT selection, canonical identities, TRAIN sampler, bounded
particle reads, full cardinality and sealed tests. Production requires the exact
registered transfer hash and Oscar scratch path.

RECIPE removes the donor's global `ce_weight=1` and binds `main_head` per stage:
OFFLINE/M0HLT/CONCAT have CE1, KD0; UE100/DE050/DE000 have CE.25, KD.75 with
predecessors OFFLINE/UE100/DE050 respectively. Main KD is T² forward KL at T2;
stored native T2 probabilities must not be softened a second time. All other
scientific settings are inherited by value from the Hydra recipe, whose hash is
recorded. NODE binds this main policy in addition to the unchanged seeds/view.
Native initialization and auxiliary initialization/dropout RNG are preserved.

## Training, targets and evidence

Five selected outgoing teachers publish unique-TRAIN native FP32 embedding
targets standardized with FP64 population moments/std floor .001, and native
BF16-forward FP32 T2 probabilities. BANK parents bind this new recipe, exact
checkpoint/report, stage, spec and shared population. All payload hashes, joins,
shapes, dtypes, finite values and normalization/probabilities are validated.
Only native HLT inputs enter deployed M0HLT/DE000; predicted targets are auxiliary.

EPOCH_COMMIT follows atomic checkpoint publication, includes full model/auxiliary
heads/AdamW/RNG, job, exact parents, cursor and deterministic draw evidence.
Rows bind `main_head` and raw `ce`/`main_kd`/per-teacher losses. The validator
reconstructs the coefficient-weighted total. A baseline cannot contain main KD;
a student cannot omit it. Matching remains DE000-only, .3 with its original ramp.
The first-batch gradient diagnostics retain raw component conventions; they are
not coefficient-scaled combined gradient norms.

TRAINING_REPORT selects maximum in-range AUC, then minimum CE, maximum macro
logR50, earliest update. Selected native weights are restored and re-evaluated.
METRICS reports in-range, outside-range and each 100-GeV bin at the same selected
checkpoint, including R50/R30 pass counts and censoring. No test operation exists.
RESULT is completion-last and binds all reports, states, views, acceptance,
metrics, ordinary predictions and outgoing bank. Recovery uses only new balanced
anchors on matching subsets. Nonpositive gaps/censored metrics remain unresolved.

## Execution and recovery

SPEC binds clean pushed source, frozen donor bytes (`donors.json`), exact runtime
versions and installed Weaver sources, backend flags, stage DAG, resources and
14 GiB new-result reservation. Raw input is on scratch, results/source in HOME;
views are RAM-only. No Hydra baseline spec/cache/reservation is needed. Runtime
is supplied explicitly and checked for real imports before registration.

All six Slurm requests use Oscar `slurmctld`, `gpu`, `default`, `norm-gpu`, one
`l40s`, six CPUs, 192000 MiB, 2880 minutes by default. Worker identity, workdir,
comment, allocation and device are checked. A source helper is loaded using
absolute PROJECT_DIR; user-site isolation and environment library precedence
are required. No standalone setup/preflight jobs. Each fit performs real
integrated acceptance: installed-Weaver parity, full-objective two-epoch
miniature/reload, classifier gradient equality to its registered main objective,
full-support stress, matching parity, memory/time probes and budget checks.
No model-quality acceptance threshold is permitted.

The detached launcher writes a separate log and continues after its tail exits.
Queueing prints the complete plan, performs `sbatch --test-only` for all requests,
then writes each intent before submission and its exact-ID receipt after.
An OS-held exclusive writer lock serializes launchers and releases on process
exit; unresolved intents still block retries. Explicit resume requires all exact
prior IDs terminal and authenticates completed stages. Unpublished fits resume
at the last contiguous committed epoch; full histories only finalize/export.
No existing files are overwritten or automatically deleted. Worker failure
preserves phase/failure evidence. Local tests do not substitute for real Oscar
timing and memory measurements inside the queued allocations.

## Persistent runtime restoration

Operational artifacts in `operations/oscar_runtime_restore.py` are stdlib-only
in the driver and carry canonical hashes: `LUKA_OSCAR_RUNTIME_RESTORE_PLAN/v1`
binds the failed spec, recorded runtime, exact Conda URLs/MD5/builds and pip
METADATA/RECORD hashes. `LUKA_OSCAR_RUNTIME_WHEELS/v1` binds every downloaded
wheel SHA256 and unpacked size. `LUKA_OSCAR_RUNTIME_RESTORED/v1` is published
only after exact package/build inventory, Conda file presence, pip RECORD file
hashes, dependency checks, recorded runtime/Weaver fingerprint equality and a
synthetic CPU forward/backward pass. GPU acceptance remains in science jobs.
Failures retain `LUKA_OSCAR_RUNTIME_RESTORE_FAILURE/v1` and the partial prefix.

Fresh Conda caches and `--copy` prevent linking the HOME prefix to scratch.
Every pip version is pinned from saved metadata; no dependency solving or
automatic upgrades. Torch uses the official recorded CUDA 11.8 wheel channel.
Require 28 GiB declared HOME headroom: at most 12 GiB runtime, 14 GiB scientific
outputs, 2 GiB margin. Download and extraction work stays in scratch. Remaining
quota passed to scientific submission deducts measured runtime payload and
the margin; this is a reservation from user-reported quota, not live quota
measurement or protection from concurrent unrelated writes.

The specific startup-only retry of `f9dfc80b_r1` writes
`LUKA_OSCAR_STARTUP_RETRY/v1` under the new execution's runtime launch directory.
It binds the original spec, authenticated submission IDs and terminal accounting,
and rejects saved epochs/checkpoints/results. A new source/runtime path registers
a new spec/root and undergoes the ordinary six-job dry review. No old spec,
journal, prefix, model, or dataset is modified. Restoration alone submits no jobs.
