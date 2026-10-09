# Luka FullSim stage 2 on SPORC

This is particle validation and a technical GPU probe, **not** the scientific
direct/coarse/fusion campaign. [Plan](plans/LUKA_FULLSIM_STAGE2_PLAN.md) and
[contract](contracts/LUKA_FULLSIM_STAGE2.md). Historical jobs are unchanged.

## Recovery after GPU job 21834332: use preflight-v2

The 90000 MiB attempt reached fusion and exhausted host RAM (92158920 KiB
sampled MaxRSS). Its first two train/validation passes totaled only 107.6s;
the 2h23 elapsed includes repeated loading and technical checks. The old log
does not pinpoint parity versus worst-case stress. V2 logs these separately,
with 15-second heartbeats. Preserve both failed outputs and their pinned source.

Reuse completed preparation from job 21831385:

```bash
PREPARED_PROJECT=/home/ryreu/atlas/HLT_Classification_luka_1256c756
PREPARED=/home/ryreu/atlas/HLT_Classification/checkpoints/luka_fullsim_1256c756_r2/prepared_mm_r1
BASE=/home/ryreu/atlas/HLT_Classification/checkpoints
```

Commit/push the fix and create a **new clean detached worktree**. Set `COMMIT`
to its full new commit and `PROJECT` to that worktree. Do not edit the old one
or use `1256c756` as the v2 execution commit. Runtime authenticates both sources
and requires all pre-existing `src/` bytes to match. Unrelated scientific changes
will intentionally stop reuse; do not bypass it. No foundation/audit/preparation
rerun is needed when compatibility passes.

Dry-review this request on **sporcsubmit**, with a fresh output:

```bash
GPU="${BASE}/luka_preflight_v2_${COMMIT:0:8}_r1"
test -f "${PREPARED}/prepared.json"
test ! -e "${GPU}"
sbatch --test-only --nodes=1 --ntasks=1 --cpus-per-task=6 --mem=256000M \
  --gres=gpu:a100:1 --time=08:00:00 --partition=debug \
  --account=reu-aisocial --qos=qos_tier3 --export=NONE --no-requeue \
  --job-name=luka_gpu_preflight_v2 --output="${BASE}/luka-preflight-v2-%j.out" \
  "${PROJECT}/sbatch/run_luka_fullsim_stage2.sh" "${PROJECT}" "${COMMIT}" preflight-v2 \
  --prepared "${PREPARED}" --prepared-project "${PREPARED_PROJECT}" \
  --output "${GPU}" --site sporc_a100_debug
```

After review, repeat **once without `--test-only`**. This documentation submits
nothing. The 256000 MiB allocation is a new envelope, not guaranteed sufficient;
unchanged batch256 stress runs first and checks cache headroom. For tier3 change
both partition and site (`sporc_a100`). The worker sets cuBLAS workspace before
Python; no manual export is needed. Models/particles/batch size are unchanged.

Expect small TRAIN witnesses, installed/native and offload parity, all four
batch256 stress probes, then one multi-view ROOT pass per role and four full
population one-pass probes. Retain partial output/logs on failure. Read successful
`preflight.json` with the new CLI's `summary`. No science jobs auto-submit.

## Original preparation and historical v1 commands

Audit/convention/preparation below remain valid. The final 90000 MiB v1 GPU
example is historical: use the v2 recovery above for this prepared population.

Start with a completed [stage-1 foundation](LUKA_FULLSIM_FOUNDATION_RUNBOOK.md).
Commit/push both stages, then create a clean detached SPORC worktree at that
exact commit using the stage-1 runbook. Do not copy new Python into an older
running worktree. Do not run particle preparation on a login node.

Set these shell variables to the real paths (not the literal placeholders):

```bash
PROJECT=/home/ryreu/atlas/HLT_Classification_luka_YOUR_COMMIT_PREFIX
COMMIT=YOUR_FULL_40_CHARACTER_PUSHED_COMMIT
FOUNDATION=/home/ryreu/atlas/HLT_Classification/checkpoints/YOUR_COMPLETED_FOUNDATION
BASE=/home/ryreu/atlas/HLT_Classification/checkpoints
AUDIT="${BASE}/luka_particle_audit_${COMMIT:0:8}_r1"
PREPARED="${BASE}/luka_prepared_${COMMIT:0:8}_r1"
GPU="${BASE}/luka_preflight_${COMMIT:0:8}_r1"
```

Phase output directories must **not already exist**. Logs go in their existing
parent directory. An interrupted phase preserves partial files; don't delete or
resubmit blindly. Nothing provides an automatic retry or automatic follow-up.

## 1. Audit now, without guessing units

On `sporcsubmit`, review a CPU-only request first:

```bash
sbatch --test-only --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G \
  --time=04:00:00 --partition=debug --account=reu-aisocial --qos=qos_tier3 \
  --export=NONE --no-requeue --job-name=luka_particle_audit \
  --output="${BASE}/luka-audit-%j.out" \
  "${PROJECT}/sbatch/run_luka_fullsim_stage2.sh" "${PROJECT}" "${COMMIT}" audit \
  --foundation "${FOUNDATION}" --output "${AUDIT}"
```

After review, repeat that exact command **without `--test-only` once**. Initial
4h/8GiB is an unmeasured envelope, not a runtime forecast. No GPU is requested.
After successful completion, read the report with the scientific environment:

```bash
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PROJECT}/src"
python -s "${PROJECT}/scripts/luka_fullsim_stage2.py" summary "${AUDIT}/audit.json"
```

The JSON includes offline/HLT and charged/neutral/per-PID tracking statistics,
finite/nonfinite counts, zeros, negatives, absolute-value histograms, particle
counts and p4/PID structural warnings. SD is width, not uncertainty. These are
the exact selected 100k/50k rows, with separate roles, not extra jets or test.
ROOT baskets may contain other entries as bytes, but only contiguous selected
entry ranges are requested as arrays. Neither test files nor unused-row particle
arrays are requested. Source files are checksummed before and after consumption.

## 2. Review conventions before physical preparation

**The producer has not confirmed tracking units/sentinels. Stop here pending
that information or an explicit, documented provisional operator decision.**
Zero counts do not prove that zero is an unavailable sentinel. Compare d0/dz
values and errors for both sides, but never infer mm/cm from widths. Ask about
negative/nonfinite or special finite sentinels if the audit shows them.

The `conventions` subcommand requires all of:

- `--audit` pointing at this completed `audit.json`;
- `--offline-unit` and `--hlt-unit`, each `mm` or `cm`;
- `--zero-error error_only_unavailable` (finite value retained) **or**
  `--zero-error value_and_error_unavailable` (both marked unavailable);
- `--authority producer_confirmed` or `--authority operator_provisional`;
- `--evidence` with an actual explanation/source of the declaration;
- `--output` a fresh JSON path, for example `${BASE}/luka-conventions.json`.

This explicitly declares that all four tracking fields on each side share the
stated unit, momentum is GeV, signs are retained, neutral tracking is unavailable,
and there are no other finite charged-value sentinels. Unsupported conventions
need an explicit contract change, not editing raw data or relaxing validation.
Provisional labels and unknown cross-file event independence remain visible.

Once reviewed, submit `prepare` using the same CPU worker and reviewed 4h/8GiB
envelope, supplying `--foundation`, `--audit`, `--conventions` and `--output`.
Use `--test-only` first. It validates every particle and all seven views, writes
small assignment NPZs and publishes `prepared.json` last. No particle caches,
smearing, model training or permanent physical-data rewrite occurs.

## 3. Genuine SPORC GPU preflight

Only after preparation completes, use the **same committed source**. Review:

```bash
sbatch --test-only --nodes=1 --ntasks=1 --cpus-per-task=6 --mem=90000M \
  --gres=gpu:a100:1 --time=08:00:00 --partition=debug \
  --account=reu-aisocial --qos=qos_tier3 --export=NONE --no-requeue \
  --job-name=luka_gpu_preflight --output="${BASE}/luka-preflight-%j.out" \
  "${PROJECT}/sbatch/run_luka_fullsim_stage2.sh" "${PROJECT}" "${COMMIT}" preflight \
  --prepared "${PREPARED}" --output "${GPU}" --site sporc_a100_debug
```

Repeat without `--test-only` once after review. A tier3 alternative requires
changing **both** partition to `tier3` and site to `sporc_a100`; no claim is made
about which currently queues faster. Do not reuse Oscar/Tigris GPU evidence.

Preflight runs full 100k/50k one-pass U000 CE, direct HLT KD, adjacent fusion KD
and HLT-pair KD probes, with installed-Weaver/mask/offload parity, longest-TRAIN
batch256 stress, selected-state readback and TRAIN probability-bank readback.
One-pass weights are NONSCIENTIFIC and must not seed the eventual science jobs.
Only the new U000 probe supplies technical KD targets; old teachers are absent.
The future science recipe remains 60--100 passes, full validation each pass,
AdamW and C25/P75 T2 distillation. Probe accuracy is not an acceptance threshold.

Read `preflight.json` with `summary`. Inspect `resource_envelope_ok`,
`fits_debug_24h`, projected train/reducer minutes, measured RSS/GPU peaks and
all parity checks. A completed technical report can still recommend a different
resource allocation; it cannot automatically authorize science submission.
Stage 3 must bind these precise products/environment and register the comparison
graph. Final test stays sealed throughout.
