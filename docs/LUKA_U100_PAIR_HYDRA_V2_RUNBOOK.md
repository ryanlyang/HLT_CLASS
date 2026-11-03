# Queue the four Hydra fits after precision diagnostic 131484

The source-authentication repair after `d6f0126d` is included in the commands
below. That attempt stopped in campaign creation, before creating its output
root or submitting any jobs. Commit the repair and use its new commit on Brown;
keep the completed diagnostic and its original checkout.

This queues a **new v2 campaign** using the completed diagnostic at
`$HOME/hlt_u100_pairs/runs/pair_precision_ad55d3cd_r1`. Keep that root, its
original code checkout, the existing bundle and venv. No SPORC GPU, data transfer,
reinstallation, reference fit or repeated control inference is required.

The helper authenticates job 131484 and its original source, then submits one
shared setup on Hydra `gpus/default/normal`, one L40, six CPUs, 120000 MiB.
Setup exports the required particle targets and measures the four production
miniatures. Successful setup automatically queues these **four 30-epoch fits**
in two L40 lanes, followed by a report:

| Order | Task | Addition to CE + U100 embedding weight 5 |
| --- | --- | --- |
| 1 | `p4_pair_u100` | Correct particles and supervise derived pair features |
| 2 | `pair_u100` | Direct pair-feature correction |
| 3 | `dual_u100` | Separate particle and pair predictors |
| 4 | `p4_u100` | Correct particles without explicit derived-pair reconstruction loss |

The scientific model/loss/schedule are unchanged. Cross-site BF16 max/RMS
differences are report-only under the explicit v2 policy; exact lineage,
local CPU/GPU FP32 checks, repeated BF16 batches and AUC-drift qualification
remain required. The old diagnostic does not replace miniature/resource checks.
See the [active plan](plans/LUKA_U100_PAIR_HYDRA_PLAN.md) and
[versioned contract](contracts/LUKA_U100_PAIR_HYDRA.md).

## Commit and push on Windows

These paths cover this change without staging unrelated shared documentation.
The standalone validation record includes donor and test evidence.

```powershell
& {
    Set-Location C:\Users\22rya\ComputerScience\CERN\HLT_Classification
    if ((git branch --show-current) -ne 'main') { throw 'Use the intended main branch.' }
    if (@(git diff --cached --name-only).Count -ne 0) { throw 'Review existing staged changes first.' }
    $hydraSourceFiles = @(
        'src/hlt_classification/luka_u100_pair_hydra/portability.py',
        'tests/test_luka_u100_pair_hydra_portability.py',
        'docs/plans/LUKA_U100_PAIR_HYDRA_PLAN.md',
        'docs/contracts/LUKA_U100_PAIR_HYDRA.md',
        'docs/LUKA_U100_PAIR_HYDRA_V2_RUNBOOK.md',
        'docs/LUKA_U100_PAIR_HYDRA_VALIDATION.md'
    )
    git add -- @hydraSourceFiles
    if ($LASTEXITCODE -ne 0) { throw 'Staging failed.' }
    git diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw 'Check the staged changes.' }
    git diff --cached --stat
    git commit -m 'Validate reused Hydra diagnostic in its original checkout'
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
    git push origin main
    if ($LASTEXITCODE -ne 0) { throw 'Push failed.' }
    Write-Host "`nUse this commit on Brown:"
    git rev-parse HEAD
}
```

## Queue on Brown

Paste the entire block with newlines intact. Enter the full commit above.
This reuses the existing bundle/venv and creates a fresh campaign root.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed Hydra v2 commit (40 characters): ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
BASE="$(cd "${HOME}" && pwd -P)/hlt_u100_pairs"
PROJECT="${BASE}/code_hydra_${COMMIT:0:8}"
git -C "${BASE}/code_65952ed3" fetch origin main
git -C "${BASE}/code_65952ed3" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${BASE}/code_65952ed3" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_u100_pair_hydra.sh" "${BASE}"
BASH
```

Enter the available output allowance (at least **8 GiB**, after the environment
and bundle). The helper prints/saves the complete dry plan and automatically
uses its hash for the already-requested campaign. Invoking the helper authorizes
that shared setup, four fits and report; no additional YES prompt or separate
preflight/diagnostic is queued. The full dry run and measured-resource requirement
come from the repository research-compute rules.

The helper prints a **dispatcher PID and log path**. That detached process
survives logout. Its log should say **`Setup submitted as <job>. Automatic
measured science submission is active; safe to log off.`** Once setup succeeds,
it queues all four fits plus the report automatically. Pending dependencies
between fit lanes are ordering only (`afterany`), not teacher dependencies.
If setup fails, no fits are released and the log retains the cause.

Do not resubmit the older failed roots. A failed or interrupted submission may
already own jobs; preserve the printed root and its exact journals.

## Results and code

Use the regular CLI `results --spec <new-root>/campaign_spec.json` with the new
checkout and existing venv. The report retains historical references and adds
`U100_W5_HYDRA` (AUC 0.96209461 on full validation). Student delta-AUC uses this
Hydra baseline; recovery retains clearly labelled SPORC M0HLT/OFFLINE anchors.
Original50k overlaps validation and remains report-only.
The control retains its SPORC-trained weights; the Hydra row changes evaluation
hardware only, without a claim of identical training trajectories across sites.

`src/hlt_classification/luka_u100_pair_hydra/portability.py` owns diagnostic
import, v2 policy, saved-array qualification and setup evidence. `campaign.py`
creates SPEC/v2; `worker.py` uses it and produces RESULTS/v2. `submission.py`
keeps the original measured two-lane DAG with a distinct v2 authorization.
The correction model/training/resource modules are imported unchanged.
Before replay, the new process checks the old checkout's pinned Git commit and
clean source snapshot using the generic provenance validator. The old process
then runs its own module-ownership and published-source checks through
`precision_campaign.results -> check -> campaign.current -> foundation._source`.
Calling the new checkout's `_source` on the old checkout is invalid by design.
