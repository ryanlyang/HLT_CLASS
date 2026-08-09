# Tigris Workers

This directory contains thin, absolute-path Tigris Slurm workers and the
baseline submitter. Every worker validates the immutable campaign
specification and active clean source before considering reusable artifacts.

Create a source-bound specification first, then inspect the complete DAG:

```bash
python -s scripts/create_campaign.py \
  --mode smoke \
  --output /tmp/hlt_smoke_campaign_spec.json
bash sbatch/submit_baseline.sh \
  --campaign-spec /tmp/hlt_smoke_campaign_spec.json \
  --dry-run
```

`--smoke-simulate` performs a no-Slurm failure/recovery exercise.
`--smoke-submit` submits only a miniature. Full production additionally
requires an explicitly authorized production specification and authenticated
storage measurement plus successful smoke resource evidence.

PRAD uses the same absolute-path environment bootstrap and the same Tigris
account/partition contract. Create and inspect it before any submission:

```bash
python -s scripts/create_prad_campaign.py --mode smoke \
  --campaign-root /home/ryreu/atlas/HLT_Classification/artifacts/prad/smoke \
  --output /tmp/prad_smoke.json
bash sbatch/submit_prad.sh --campaign-spec /tmp/prad_smoke.json --dry-run
```

`--full-production-submit` is unavailable unless the immutable production
spec records explicit authorization plus hashes for the prior dry run, real
miniature, and measured resource evidence.

The complete smoke, monitoring, exact-ID recovery/cancellation, measured
resource, and authorized-production sequence is documented in
[`docs/PRAD_RUNBOOK.md`](../docs/PRAD_RUNBOOK.md).

HCWDL uses `run_hcwdl_task.sh`. It activates the exact project environment,
sets `PYTHONNOUSERSITE=1`, prepends `${CONDA_PREFIX}/lib`, and ends with
`exec python -s`, allowing Slurm `B:USR1` to reach the checkpointing process.
All future commands are generated locally from an immutable HCWDL spec. The
`shell_endpoint_qualification_lock` job includes `--hold`; release is a later,
separately authorized operation after the lineage-bound endpoint diagnostic
acknowledgement is written. Pilot/production resources remain planning values
until a genuine Tigris miniature publishes measured evidence. The measured
prelaunch candidate and the executable spec must share the same independently
hashed `HCWDL_COMMAND_PLAN/v1`; explicit submission authorization binds that
hash and the exact resource requests, avoiding any circular dependency on the
enclosing campaign-spec hash. A first bounded smoke can therefore use
explicitly authorized conservative bootstrap requests without being mislabeled
as measured; pilot and production cannot.

HCWDL-RKD has three disjoint worker pairs. The campaign workers dispatch only
an executable campaign; the bootstrap workers dispatch only the scalar prefix
through zero-coefficient acceptance; and
`run_hcwdl_representation_nonfinal_acceptance.sh` plus
`run_hcwdl_representation_nonfinal_acceptance_deterministic.sh` accept only a
phrase-issued `HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_AUTHORITY/v1` and one
literal registered action. Authority implementation is not execution
authorization, and none of these workers is submitted by a builder.
The authority must bind a deeply validated
`HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_ACTION_INPUTS/v1`; workers reopen
that artifact rather than trusting a path or caller-authored hash.
The Python runner dispatches only the authority-bound scalar action through
the production bridge and publishes an immutable execution receipt over its
semantic outputs and dependency results. The workers do not submit other jobs
or acquire any pilot/final capability.

The non-final registry contains bounded D0c/D0w target preparation, the exact
four M1 cold/warm RSET/RREL probes, the reference/interrupt/resume USR1
sequence, and one validation-only D0c/D100/TOFF proxy. Training is fixed at 512
training rows, 256 validation rows, and exactly two optimizer updates. The
interrupt route requires actual POSIX `SIGUSR1` delivery and an authenticated
receipt; resume runs in a fresh process. The validation proxy rejects
`final_test` and every shared-final contract. These workers accept no arrays,
campaign task keys, pilot submission, reservation, capability, or final role,
and all results are nonauthorizing evidence. A worker may publish only its
registered semantic outputs and execution receipt. After that job is
terminal, `run_hcwdl_representation_nonfinal_evidence_collector.sh` runs as a
separate Slurm worker, captures fresh raw `sacct` bytes itself, and binds the
exact submit script/argv, resource class, worker role, and pushed source.
It must run under exact job name `hcwdl-rkd-nonfinal-evidence-collector`,
account `reu-aisocial`, and partition `tigris`; the runtime rejects non-Tigris
compute-host identities and any `sacct` executable other than the root-owned
`/usr/bin/sacct` client.
The separate post-job builder combines that collector evidence with the
execution receipt into an
`HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_ACTION_RESULT/v1` bound to its exact
authority, action inputs, action ID, dependency results, Slurm job, worker
bytes/role, and semantic outputs before higher-level proof assembly.
