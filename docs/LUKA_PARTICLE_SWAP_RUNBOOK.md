# Whole-particle replacement ladders on SPORC debug

Authority: [scientific plan](plans/LUKA_PARTICLE_SWAP_PLAN.md) and
[v1 contracts](contracts/LUKA_PARTICLE_SWAP.md).

Two **separate** arms: `logit` and `attn` (attention coefficient 0.1).
Each starts from the SAME existing regular U100 checkpoint, then trains
SWAP066 -> SWAP033 -> SWAP000. Each replacement changes a complete particle
record, not a fraction of its features. The nested, HLT-only ordering balances
particle counts, pT and charged/neutral rank strata. Intermediate views remain
synthetic training oracles, not detector HLT. SWAP000 is exactly native HLT.

Same 299,234 TRAIN / 74,895 validation rows, batch128, fixed30, cold starts and
coordinate seeds as the earlier attention trial. Original50k is report-only.
The historical test was retired into TRAIN: **no independent test exists**.
These are matched exploratory single-seed comparisons, not a promised recovery
improvement or a new test-set result.
The discussion's "D100" start is the existing `COARSE_U100` endpoint; no new
U-side models need to be trained.

## Resources and storage

Per arm: one A100, six CPUs, 128000MiB; both phases use `debug`. One8h preflight
measures the entire pipeline and all three one-pass fits. Only an accepted gate
and successful exact Slurm accounting permit the separately authorized science
job. Science runs the three full30 fits sequentially in ONE allocation, with a
measured request capped24h. No automatic submission, retry or cancellation.

All four physical views are constructed once in RAM from selected raw ROOT
ranges and the existing assignments. Matching is not rerun. U100/native-HLT
fingerprints must reproduce the original inputs; all view fingerprints and
replacement-order hashes must reproduce preflight in science. No new durable
particle cache. No validation-based choice of replacement order.

Reserve at least **1GiB of personally available persistent quota per arm**.
This includes technical outputs, copied references and retained predictions.
The RAM input budget is30% of host allocation; resident teacher targets get15%.
The preflight measures actual combined high water and includes headroom.

One rolling FP32 teacher bank and new checkpoints live on private node-local
`/tmp`. Storage admission also checks `4*sum(n*n)+128*N+3GiB` local scratch.
NFS/Ceph/shared storage is rejected; there is no home fallback. If `/tmp` cannot
provide enough space, ask site support about `/local/scratch`; register that
explicitly through `create-gate --scratch-base`, not by editing an old spec.

Consumed banks are removed only after authenticated child predictions/report;
intermediate checkpoints only after their next export is validated. Final
checkpoint is removed after selected predictions/native readback. Keep per-jet
probabilities, labels, identities, original50k mask, full histories, input
audits, model/bank hashes and retirement receipts. Future R30/other operating
points remain calculable from predictions. **New inference requires retraining
after checkpoint retirement.** Nothing deletes raw data, old caches/models,
the shared U100, other runs or Fresh_check. Failed jobs are not auto-restarted.

## 1. Review, commit and push locally

No commit or push was performed by the implementation task. In PowerShell:

```powershell
git add -- src/hlt_classification/luka_particle_swap scripts/luka_particle_swap.py scripts/queue_luka_particle_swap.sh sbatch/run_luka_particle_swap.sh tests/test_luka_particle_swap.py tests/test_luka_particle_swap_views.py docs/plans/LUKA_PARTICLE_SWAP_PLAN.md docs/contracts/LUKA_PARTICLE_SWAP.md docs/LUKA_PARTICLE_SWAP_RUNBOOK.md
git diff --cached --stat
git diff --cached --check
```

Review before committing. Stage only this task's HANDOFF/LEGACY_SOURCE_MAP
hunks with `git add -p`; preserve unrelated changes. Do not `git add .`.
Commit, push without force, and obtain `git rev-parse HEAD`. Stop on a
non-fast-forward and reconcile first. Existing runtime source is pinned to
`901535f026d32868e19c159a073947dfdbbbdd3c`; only the new runtime namespace may
differ. Other simultaneous runtime changes require explicit compatibility
review, not bypassing the guard.

After reviewing the staged changes:

```powershell
git commit -m "Add matched whole-particle replacement ladders"
if ($LASTEXITCODE -ne 0) { throw "Commit stopped; inspect before proceeding" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push stopped; do not force-push" }
git rev-parse HEAD
```

## 2. Dry-review one fresh arm on sporcsubmit

Repeat for `logit` and `attn`. Give each arm its own quota allowance; do not
allocate the same remaining space twice. This creates metadata and performs
read-only authentication but submits no job. It does not cancel any old run.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
read -r -p 'Arm (logit or attn): ' ARM </dev/tty
read -r -p 'Personally available persistent GiB for THIS arm: ' QUOTA </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${ARM}" =~ ^(logit|attn)$ && "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_lukaswap_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_swap_${ARM}_${COMMIT:0:8}_r1"
PARENT="${MAIN}/checkpoints/luka_fixed30_74aa58c5_r1/science/campaign_spec.json"
REFERENCE="${MAIN}/checkpoints/luka_attention_94f56025_r1/science/campaign_spec.json"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_particle_swap.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${REFERENCE}" "${ARM}" "${QUOTA}" gate
printf '\nStudy root: %s\n' "${ROOT}"
BASH
```

Required old data: raw890-file Luka snapshot, ALLJETS prepared metadata and
assignment payloads, fixed30 U100/control reports/checkpoints and their source.
The original LOGIT/ATTN results use the explicit comparison-only archive proof:
their selected predictions, reports, receipts and aggregate must remain; their
deleted old scratch attention banks are NOT needed and are never accepted as
new training inputs. Do not restore or fake missing bytes to satisfy a hash.

Check new `LUKA_PARTICLE_SWAP_GATE/v1`, chosen arm, `--partition=debug`,
`--gres=gpu:a100:1`, `--mem=128000M`, `--time=08:00:00` and expected population.
If a partial root exists without a valid gate, preserve it and investigate;
do not delete it or blindly choose another root that could duplicate a job.

## 3. Authorize preflight, then science after acceptance

Use `MODE=gate` first. Once its actual Slurm job COMPLETED with0:0, repeat with
`MODE=science`. That call authenticates the complete preflight and exact
submission journal before creating the science spec; a printed success line
alone cannot bypass acceptance. The dry plan prints a new64-character
`content_hash`; paste its VALUE, not the text `content_hash`.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Exact study root printed above: ' ROOT </dev/tty
read -r -p 'Phase (gate or science): ' MODE </dev/tty
[[ "${MODE}" =~ ^(gate|science)$ ]]
PYTHON=/home/ryreu/miniconda3/envs/atlas_kd_sporc/bin/python
test -f "${ROOT}/gate_spec.json"
mapfile -t FIELDS < <("${PYTHON}" -B -s -c 'import json,sys; g=json.load(open(sys.argv[1])); print("\n".join(str(v) for v in [g["source_commit"],g["project_dir"],g["root"],g["anchor"]["spec"]["path"],g["reference_import"]["spec"]["path"],g["arm"],g["available_quota_gib"]]))' "${ROOT}/gate_spec.json")
[[ "${#FIELDS[@]}" -eq 7 && "${FIELDS[2]}" == "${ROOT}" ]]
COMMIT="${FIELDS[0]}"; PROJECT="${FIELDS[1]}"
PARENT="${FIELDS[3]}"; REFERENCE="${FIELDS[4]}"; ARM="${FIELDS[5]}"; QUOTA="${FIELDS[6]}"
HELPER="${PROJECT}/scripts/queue_luka_particle_swap.sh"
bash "${HELPER}" "${COMMIT}" "${ROOT}" "${PARENT}" "${REFERENCE}" "${ARM}" "${QUOTA}" "${MODE}"
read -r -p 'Reviewed PLAN content_hash value: ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Submit this ONE reviewed job and authorize owned scratch retirement? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]]
LOG="$(mktemp "${ROOT}/${MODE}-submit.XXXXXX.log")"
nohup bash "${HELPER}" "${COMMIT}" "${ROOT}" "${PARENT}" "${REFERENCE}" "${ARM}" "${QUOTA}" "${MODE}" --execute "${PLAN_HASH}" >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\nWatch with:\ntail -f "%s"\n' "$!" "${LOG}" "${LOG}"
BASH
```

Confirm the log contains the durable submission ledger/job ID and
`QUEUE_HELPER_EXIT=0`. A helper PID alone does not mean the job is queued.
If disconnected or uncertain, inspect its log, exact ledger and `squeue` before
retrying. Never submit science manually in parallel with another controller.
This implementation has no automatic follow-up controller.

## 4. Partial and final results

Only saved prediction/metadata validation, no new inference or test access:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Exact swap study root: ' ROOT </dev/tty
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
PROJECT="$(python -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${ROOT}/gate_spec.json")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -B -s "${PROJECT}/scripts/luka_particle_swap.py" results --spec "${ROOT}/science/campaign_spec.json"
BASH
```

Progress is also in `science/ladder-<jobid>.out`: construction-file counts and
each epoch's node/AUC. Durable per-rung directories are
`science/ladder/SWAP_LOGIT_D066` etc or `SWAP_ATTN_D066` etc. Histories include
all30 validation epochs and selection. Compare final SWAP*_D000, original
LOGIT_D000/ATTN_D000, M0HLT and DIRECT_D000; intermediate views contain offline
information and are not deployable performance. Show every registered row,
including decreases. Both arms are reported even if neither improves.

## Acceptance boundary

Local fixtures test logic and installed-Weaver numerics; they are not real
SPORC resource evidence. Queue readiness does not mean the remote gate has
passed or that jobs have been submitted. Do not alter walltime, batch, counts,
coefficient, memory, source, matching or split in-place after review. Any
necessary change requires a new versioned registration and measured preflight.
