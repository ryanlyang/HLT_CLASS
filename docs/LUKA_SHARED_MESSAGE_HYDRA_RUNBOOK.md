# Queue the shared-message campaign on Hydra

[Scientific plan](plans/LUKA_SHARED_MESSAGE_HYDRA_PLAN.md) ·
[Contracts](contracts/LUKA_SHARED_MESSAGE_HYDRA.md) ·
[Validation](LUKA_SHARED_MESSAGE_HYDRA_VALIDATION.md)

Twelve fresh HLT fits, all CE + U100 embedding weight5. The primary arm jointly
learns two message producers sharing the same downstream classifier. Compare it
with `control_s0`, which has the identical deployed attention predictor. Controls
also predict U100 embeddings; they are not CE-only.

Use the RTX A6000 pool (`gpu:nvidia_rtx_a6000:1`), currently gpu2201, without a
node pin. Six CPUs, 32,000 MiB RAM, individually measured walltimes up to 48h.
No additional SPORC transfer or reference retraining is needed.

## Prerequisites

- Original parent: `runs/cross_message_199928da_3693bcda_a6000_r1/campaign_spec.json`.
- Keep its original source worktree, completed setup144840 and selected
  teacher_m16/145410 artifacts. Existence alone is insufficient: creation
  authenticates both through the original worktree.
- Keep the original input bundle, adapter delta, U100 encoder delta and
  `environment_pair_hydra_v1/venv`; paths come from the parent spec.
- Reserve four GiB for new outputs plus one GiB filesystem headroom.
- Commit/push the new source. The queue script creates a fresh campaign root.
  It does not cancel, change or release any old jobs.

## Commit and push on Windows

Run in the repository's PowerShell terminal. This stages the new campaign files
explicitly, leaving unrelated working-tree edits alone. Review any files already
staged before committing. The source validator requires additive runtime relative
to `546077aa7fce428bf3f9f136ab2277ac5eed5f85`.

```powershell
Set-Location 'C:\Users\22rya\ComputerScience\CERN\HLT_Classification'
git diff --cached --stat
git add -- src/hlt_classification/luka_shared_message_hydra scripts/luka_shared_message_hydra.py scripts/queue_luka_shared_message_hydra.sh sbatch/run_luka_shared_message_hydra.sh tests/test_luka_shared_message_hydra.py docs/plans/LUKA_SHARED_MESSAGE_HYDRA_PLAN.md docs/contracts/LUKA_SHARED_MESSAGE_HYDRA.md docs/LUKA_SHARED_MESSAGE_HYDRA_RUNBOOK.md docs/LUKA_SHARED_MESSAGE_HYDRA_VALIDATION.md
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Check staged whitespace' }
git diff --cached --stat
git commit -m 'Add jointly learned shared-message Hydra campaign'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

The living README, HANDOFF and donor map also contain integration notes. Review
their pre-existing edits before staging those files separately.

## Queue on Brown

Paste this as a multiline block in the Brown terminal. It asks for the full
40-character commit printed above. The script reviews setup/release and the
complete twelve-fit resource envelope, then requests `YES` for that concrete
plan. Science is released only after real warmed production miniatures pass.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "${HOME}" && pwd -P)/hlt_u100_pairs"
MAIN="${BASE}/code_65952ed3"
read -r -p 'New pushed 40-character shared-message commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
PROJECT="${BASE}/code_shared_message_${COMMIT:0:8}"
if [[ -e "${PROJECT}" ]]; then
    [[ "$(git -C "${PROJECT}" rev-parse HEAD)" == "${COMMIT}" ]]
    [[ -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)" ]]
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_shared_message_hydra.sh"
BASH
```

Once **both setup and release job IDs** are printed, logging off is safe. Release
is a CPU Slurm job in `compute`, dependent on successful setup; it submits twelve
independent GPU fits and a CPU report. The twelve fits have no dependencies on
each other. There are15 scheduler jobs total, of which12 are scientific fits.
Poor accuracy cannot stop the release. Invalid inputs, nonfinite values, stale
source or exceeded measured resource bounds do stop it and preserve diagnostics.

```bash
squeue --me -o "%.18i %.38j %.12T %.12l %R"
```

Job prefix is `lukasm_`. Inspect `setup-JOBID.out` and `release-JOBID.out` inside
the printed campaign directory. Do not delete setup or prior campaigns to retry.

## Results and epoch-by-epoch progress

Replace `NEWCOMMIT` with the first eight characters of the new commit. `progress`
prints every published epoch, recovery against historical M0HLT/OFFLINE, the
same-seed architecture control and historical U100-w5. `results` authenticates
completed fits and reports selected checkpoints, including original50k separately.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "${HOME}" && pwd -P)/hlt_u100_pairs"
SPEC="${BASE}/runs/shared_message_NEWCOMMIT_a6000_r1/campaign_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "${SPEC}")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_shared_message_hydra.py" progress --spec "${SPEC}"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_shared_message_hydra.py" results --spec "${SPEC}"
BASH
```

These commands are read-only. Validation is exploratory, not an independent test;
original50k overlaps it. The matched-control delta is the important estimate of
what joint privileged training adds beyond the new HLT architecture and U100-w5.

## Release recovery

If setup completed but the CPU release failed before submitting science, inspect
the exact release log and `submission_science` journal. With `SPEC`, `PROJECT`,
`ENV_DIR` and exports set as above, the explicit recovery entrypoint is:

```bash
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_shared_message_hydra.py" finish --spec "${SPEC}"
```

It verifies the original authorization, completed setup and measured miniatures.
A fully submitted stage is idempotent. An active or partial live claim is rejected
because a scheduler response may have been lost; preserve journals for exact-ID
reconciliation. Never remove a claim to force resubmission. No automatic retries.

## Code map

- `src/hlt_classification/luka_shared_message_hydra/contracts.py`: twelve arms,
  coefficients, normalization/selection recipe and artifact contracts.
- `model.py`: native block4 hook, HLT attention predictor, U100 cross-read,
  shared message expansion and native downstream route; stripped deployment.
- `training.py`: exact objective, fresh seeds, HLT validation selection,
  immutable epoch files and executable checkpoint replay.
- `inputs.py`: TRAIN-only U100 token cache and frozen m16 teacher import.
- `campaign.py`: fresh root, source pin, original-worktree proofs and quotas.
- `resources.py`: fresh warmed miniatures, timing, memory and acceptance checks.
- `submission.py`, `worker.py`: durable setup/release, independent fit DAG,
  exact-job receipts, technical gates and publication.
- `reporting.py`: read-only completed results and live epoch comparisons.
- Thin CLI `scripts/luka_shared_message_hydra.py`; queue and Slurm wrappers
  beside it and under `sbatch/`.

No original runtime is modified. Changing scientific semantics requires a new
contract version and a fresh execution, not editing an existing spec or worktree.
