# Brown CS Hydra execution contract v1

Authority: [Hydra execution plan](../plans/LUKA_U100_PAIR_HYDRA_PLAN.md), with the
unchanged [correction science](LUKA_U100_PAIR_CORRECTION.md). New execution
artifacts use `LUKA_FULLSIM_ALLJETS_PAIR_HYDRA_<KIND>/v1`. Extraction uses
`LUKA_U100_PAIR_HYDRA_EXTRACTION/v1`. Original scientific reports, particle banks,
miniatures, deployments and resource evidence retain their original contracts
and bind the new SPEC hash. Historical artifacts are never rewritten.

Only the plan's pinned archive and manifest are admitted. Extraction verifies
the archive before copying, validates every regular member while copying,
rehashes the archive afterward and publishes a completion receipt atomically.
Paths must be relative, explicit and free of symlinks/traversal. Existing or
incomplete destinations are not overwritten. Each subsequent input use validates
its original byte count and checksum. Original absolute source paths are evidence
only; no legacy validation routine may follow them on Brown. Missing input files
fail without reconstruction. TRAIN U100 and assignments never become model inputs.

The portable loader validates the original foundation and split identities,
block membership, row order, labels, shapes and full view fingerprints. Particle
target assembly and TRAIN statistics import the original implementation. U100
embedding bank validation imports the original Bank without teacher inference.
Only one native control checkpoint is loaded, for validation portability, never
to initialize any of the four fresh students.

The isolated environment must equal the recorded INSTALLED_ENVIRONMENT/v2,
including Python patch version, package versions, CUDA/cuDNN and every Weaver
Python source digest. The source guard allows only this additive Hydra runtime
relative to `65952ed3805f812c06d1ea54117c35b630cc684e`; original scientific runtime
must be unchanged. Source must be clean and present in fetched origin/main.

Hydra allocation: cluster hydra, partition gpus, account default, job QoS normal,
one NVIDIA L40, six CPUs, 120000 MiB, x86_64, registered venv and user-site disabled.
Workers prepend its lib directory to LD_LIBRARY_PATH and set deterministic cuBLAS
configuration. No environment or GPU substitution is automatic.

The controller must identify the exact running job, account, partition, QoS,
single node and six CPUs. Equal endpoint count ranges (`1-1`, `6-6`) are accepted
as exact counts. For task count and CPUs/task, concrete controller and exported
batch values must agree; at least one source must provide each. Missing, `N/A`
or `(null)` controller task fields require the explicit batch value. A missing
cluster export is checked through `scontrol show config`; missing memory export
requires exact `AllocTRES.mem` in MiB. When both sources provide a value, any
conflict fails. No memory rounding tolerance is introduced. The existing exact
accepted-job journal check still precedes task execution.

The log records only allowlisted scheduler fields before validation and names
every mismatch on rejection. This representation repair does not version the
scientific or artifact contracts: resources, inputs, source binding and study
semantics are unchanged. Slurm documents the
[batch variables](https://slurm.schedmd.com/sbatch.html#SECTION_OUTPUT-ENVIRONMENT-VARIABLES);
its [23.11.4 controller formatter](https://github.com/SchedMD/slurm/blob/slurm-23-11-4-1/src/api/job_info.c)
can emit `NumTasks=N/A` when the value is unspecified.

Shared setup first checks saved control BF16 probabilities on all validation
rows using the plan's numerical tolerances. Diagnostics are saved before numerical
admission. Then the unchanged production resource suite exercises all four real
miniatures, initial native parity, reload, full-support backward and timing.
Its L40 evidence is required before scientific submission. No accuracy, recovery,
or target-quality threshold decides whether a finite scientific run completes.

Full dry review covers setup and the four-fit/report envelope. The detached
dispatcher submits setup, polls its exact job ID, authenticates completion and
measured resources, then queues C/A/D/B in two afterany lanes plus a report.
Only measured walltime reductions are allowed. Setup failure releases no fits;
an individual student failure does not cancel independent later students.
Exclusive claims and durable exact-job journals prevent blind duplicate retries.

Reserve eight GiB for outputs plus one GiB filesystem headroom, separately from
the archive, extracted inputs and environment. No automatic deletion, cancellation,
new teacher, reference fit, final-test access or unrelated campaign mutation.
