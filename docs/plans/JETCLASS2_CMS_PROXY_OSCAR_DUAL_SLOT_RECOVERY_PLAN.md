# JetClass2 CMS-proxy Oscar dual-slot recovery plan

Status: completed v5 preflight evidence; superseded before replacement science
submission by `JETCLASS2_CMS_PROXY_OSCAR_100K_DIRECT_COARSE_PLAN.md`

## Objective

Replace the first Oscar science submission's one-job-at-a-time resource
profile with a measured profile that permits two concurrent L40S jobs under
the user's `norm-gpu` QOS. The scientific population, matching foundation,
views, controls, seeds, model, loss, schedule, branch graph, and final-test
seal remain unchanged.

The v5 preflight completed successfully as job `6877779`. Ryan subsequently
changed the requested population and graph before authorizing replacement
science, so the 17-fit campaign described below must not be submitted. Its
resource evidence remains historical; the active v6 plan requires its own
100k/50k full-selected-population measurement.

## Evidence and scheduler constraint

The completed Oscar `GATE_SPEC/v4` preflight used 12 CPUs, 160000 MiB, and one
L40S. It completed successfully and observed approximately 31.4 GiB peak host
RSS. The first science submission then exposed a per-user QOS envelope of 12
CPUs, 192000 MiB, and two GPUs: each 12-CPU job consumed the full CPU quota,
so otherwise independent controls and branches were serialized with
`QOSMaxCpuPerUserLimit`.

## Replacement profile

`GATE_SPEC/v5` requests exactly:

- 6 CPUs and 6 cache workers;
- 90000 MiB host memory;
- one L40S GPU;
- the same 12-hour preflight ceiling.

Two production jobs therefore request 12 CPUs, 180000 MiB, and two GPUs in
aggregate. The memory value remains above the deterministic full-population
cache bound. Reducing cache workers changes only bounded preprocessing
parallelism; it does not change cache row order, bytes, model inputs, training
order, seeds, or scientific semantics.

## Execution sequence

1. Reuse and fully validate the existing Oscar
   `PORTABLE_MATERIALIZATION/v1`; do not rebuild assignments or copy data.
2. Create a fresh source-pinned `GATE_SPEC/v5` and dry-run its single
   preflight command.
3. Run the genuine full-population 6-worker/90000-MiB Oscar preflight.
4. Create and dry-run a fresh campaign from its `RUNTIME_PROFILE/v5`.
5. Only after the replacement campaign is ready, cancel exact pending job IDs
   from the superseded campaign. Never cancel by name or user-wide pattern.
6. Do not mutate or overwrite outputs from a running or completed old task.
   Completed old controls may be reported separately but are not silently
   adopted into the new campaign lineage.

The replacement campaign remains a fresh 17-fit/12-reducer execution. A
source or artifact hash mismatch, allocation mismatch, corrupt materialized
payload, final-test access, or cache-bound failure fails closed.
