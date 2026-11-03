# Hydra precision check: one inference job

This diagnoses setup 131410's probability mismatch using the **existing Brown
bundle, venv and U100-w5 checkpoint**. It requires no SPORC GPU or new transfer.
It queues one `lukaph_precision` job on Hydra's `gpus` partition: one L40,
six CPUs, 120000 MiB, one hour. No reference retraining, target export, miniatures
or automatic scientific submission. Preserve all failed setup roots.

The full-validation check uses the original BF16 path on 74,895 jets. A fixed
527-row selection retains five original batch shapes and compares GPU BF16,
GPU BF16 with math attention, GPU FP32 with math attention, and CPU FP32.
Full AUC/rejection metrics use all validation jets, not this small selection.
The original tolerances remain unchanged; this diagnostic reports finite
differences without treating them as a training-admission decision.

## 1. Commit and push from Windows

The explicit list leaves unrelated dirty shared documentation/code unstaged.
The standalone validation record contains this change's donor and test evidence.

```powershell
Set-Location C:\Users\22rya\ComputerScience\CERN\HLT_Classification
if ((git branch --show-current) -ne 'main') { throw 'Use the intended main branch.' }
if (@(git diff --cached --name-only).Count -ne 0) { throw 'Review existing staged changes first.' }
$hydraPrecisionFiles = @(
    'src/hlt_classification/luka_u100_pair_hydra/campaign.py',
    'src/hlt_classification/luka_u100_pair_hydra/precision.py',
    'src/hlt_classification/luka_u100_pair_hydra/precision_campaign.py',
    'scripts/luka_u100_pair_hydra.py',
    'scripts/queue_luka_u100_pair_hydra_precision.sh',
    'tests/test_luka_u100_pair_hydra_precision.py',
    'docs/plans/LUKA_U100_PAIR_HYDRA_PLAN.md',
    'docs/contracts/LUKA_U100_PAIR_HYDRA.md',
    'docs/LUKA_U100_PAIR_HYDRA_RUNBOOK.md',
    'docs/LUKA_U100_PAIR_HYDRA_PRECISION_RUNBOOK.md',
    'docs/LUKA_U100_PAIR_HYDRA_VALIDATION.md'
)
git add -- @hydraPrecisionFiles
if ($LASTEXITCODE -ne 0) { throw 'Staging failed.' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Check the staged changes.' }
git diff --cached --stat
git commit -m 'Add inference-only Hydra precision diagnostic'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed.' }
git rev-parse HEAD
```

## 2. Queue the check on Brown

Paste this with line breaks intact. Enter the full commit from step 1. The
dedicated helper authenticates the source/environment, prints and saves its
one-job dry plan, checks the resource request with Slurm, and submits that one
diagnostic. No additional confirmation prompt or subsequent campaign is needed.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed diagnostic commit (40 characters): ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
BASE="$(cd "${HOME}" && pwd -P)/hlt_u100_pairs"
PROJECT="${BASE}/code_hydra_precision_${COMMIT:0:8}"
git -C "${BASE}/code_65952ed3" fetch origin main
git -C "${BASE}/code_65952ed3" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${BASE}/code_65952ed3" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_u100_pair_hydra_precision.sh" "${BASE}"
BASH
```

Outputs go into `runs/pair_precision_<commit8>_r1`. Log off once the helper prints
**`Diagnostic accepted as <job>; safe to log off.`** Slurm owns the job then;
there is no background dispatcher. The helper also prints its exact log and
saved-results command. Existing roots/claims are not overwritten or retried.

If the connection drops during submission, inspect that root's
`submission/submission_ledger.json` and `submission/submission_ledger_journal/`
before doing anything else. An accepted job can exist even when a terminal
disconnects before the final print.

## 3. Read saved results

Use the results command printed by the helper, or enter the same full commit:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Diagnostic commit (40 characters): ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
BASE="$(cd "${HOME}" && pwd -P)/hlt_u100_pairs"
ROOT="${BASE}/runs/pair_precision_${COMMIT:0:8}_r1"
SPEC="${ROOT}/campaign_spec.json"
ENV_DIR="${BASE}/environment_pair_hydra_v1/venv"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_u100_pair_hydra.py" precision-results --spec "${SPEC}"
BASH
```

The report displays the full validation AUC, accuracy, CE and QCD rejection at
30%/50% signal efficiency, including passing counts. Classwise AUC and detailed
probability quantiles are in `precision/report.json`. Zero passing QCD jets
are censored, not infinite measured rejection. Each model has its own signal
threshold using the same metric convention.

`precision/full_validation.json` and `hydra_validation_predictions.npz` are
published before the smaller probes. `fixed_batch_predictions.npz` stores the
four local modes, saved SPORC predictions, full-pass Hydra predictions and exact
row identities. A completed run also publishes `report.json` and `receipt.json`.
An incomplete run has no completion receipt; preserve the log and partial
diagnostic output rather than rerunning blindly.

## Interpretation and code map

- If the repeated original BF16 batches differ from their full-pass values,
  investigate backend/batch behavior before attributing this solely to the GPU
  architecture.
- CPU/GPU FP32 agreement checks local consistency. GPU math BF16 versus GPU math
  FP32 tests reduced precision locally; default BF16 versus math BF16 also
  changes attention/reduction settings. Neither reproduces an absent SPORC FP32
  reference or guarantees equivalence of rare QCD tails.
- The 527 rows are a deterministic numerical probe, not a representative test
  set. Full validation is development data and was already used for selection.
- A successful diagnostic does **not** unblock the old science campaign. The
  measurements support the next explicit portability decision. No probability
  tolerance, reference result or recovery anchor is silently replaced.

Runtime locations:

| File | Responsibility |
| --- | --- |
| `src/hlt_classification/luka_u100_pair_hydra/precision.py` | Original-batch selection, isolated backend contexts, four-mode replay, metrics and display |
| `src/hlt_classification/luka_u100_pair_hydra/precision_campaign.py` | Diagnostic-only spec, exact one-job submission, inference worker, receipts and results |
| `src/hlt_classification/luka_u100_pair_hydra/campaign.py` | Additive pushed-source allowlist, shared with the scientific adapter |
| `scripts/luka_u100_pair_hydra.py` | `precision-create/review/submit/results`, and `run --task precision` |
| `scripts/queue_luka_u100_pair_hydra_precision.sh` | Existing-environment queue helper |
| `sbatch/run_luka_u100_pair_hydra.sh` | Reused absolute-project/isolated-venv batch wrapper |
| `tests/test_luka_u100_pair_hydra_precision.py` | Actual local CPU/GPU replay, backend restoration, validation-only inputs, submission and artifact checks |

See the [plan](plans/LUKA_U100_PAIR_HYDRA_PLAN.md),
[contracts](contracts/LUKA_U100_PAIR_HYDRA.md), and
[validation/donor record](LUKA_U100_PAIR_HYDRA_VALIDATION.md).
