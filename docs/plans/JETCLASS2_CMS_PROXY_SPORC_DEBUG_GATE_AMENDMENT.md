# JetClass2 CMS-proxy SPORC debug gate amendment

Status: active execution amendment

## Scope

This amendment changes only the execution site of the unfinished gate for the
200k-train/100k-validation CMS-proxy three-spine study. The selected rows,
proxy and offline endpoints, salience matcher, views, model, optimization,
controls, and DIRECT/COARSE/DENSE scientific graph remain unchanged.

The Tigris gate already published an authenticated ordinary-role release. A
new source-pinned `GATE_SPEC/v2` imports that exact release by content hash and
file hashes. It does not select rows again. It then runs:

1. imported-release authentication;
2. endpoint-specific assignment-foundation construction;
3. a full-population, one-pass A100 preflight.

## Execution policy

The three recovery-gate jobs run on SPORC `debug`, account `reu-aisocial`, QOS
`qos_tier3`, using `/home/ryreu/miniconda3/envs/atlas_kd_sporc`. Every task is
bounded to at most eight hours and 36 CPUs. Only the preflight requests one
A100. Foundation construction is CPU-only and uses 36 process workers.

Debug is measurement-only. An eligible `RUNTIME_PROFILE/v2` records the
measured `sporc_a100_debug` site and the explicit
`sporc_debug_to_tier3_same_a100_environment_resources_v1` transfer. Its
scientific execution site is `sporc_a100` (`tier3`), with identical A100,
Conda, CPU, memory, worker, and measured walltime requirements. Scientific
jobs must not run in `debug`.

## Recovery and cancellation boundary

Only the unfinished Tigris gate jobs may be cancelled, by exact job ID. The
completed release and unrelated proxy-generation jobs are retained. The new
gate root and later science root are fresh and immutable; neither mutates the
old Tigris gate.

If foundation construction or preflight exceeds the debug limit, the job must
fail closed. Increasing the time or silently moving the measurement is not an
authorized fallback.

## Right-sized preflight recovery

The first v2 foundation completed, but its pending preflight requested 36 CPUs
and 320000 MiB. Scheduler probes showed that this full-node-shaped profile
would delay tier3 placement by roughly two days. The completed foundation's
16-worker conservative cache bounds were measured as 39.90 GiB for train and
22.42 GiB for validation; the registered 75%-RAM admission calculation
requires 85069 MiB.

`GATE_SPEC/v3` therefore imports the exact v2 release and completed foundation
and runs only the preflight with 16 CPUs, 160000 MiB, one A100, and an
eight-hour ceiling. The memory request exceeds the conservative admission
minimum by 74931 MiB. Its `RUNTIME_PROFILE/v3` transfers those exact measured
resources from SPORC debug to tier3. The superseded v2 preflight must be
cancelled by exact job ID only after the v3 dry run is authenticated.
