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
