# Queue the 0.3/debug and 1.0/tier3 attention ladders

Authority: [plan](plans/LUKA_ATTENTION_WEIGHT_SWEEP_PLAN.md),
[contract](contracts/LUKA_ATTENTION_WEIGHT_SWEEP.md). This is a separate fixed30,
batch128, fresh-initialization experiment; the privileged-gate run is untouched.
Original split: 299,234 train / 74,895 validation; original 50k report-only,
no independent test.

## Storage policy to review before submitting

Each arm requests one A100, 6 CPUs, 128000 MiB. A separate 8h gate runs on its actual
partition. Once accepted, one separately authorized science job runs all three
D rungs sequentially, with a measured whole-ladder request capped at 24h. There
is no automatic follow-up or retry in this helper. Other jobs are not cancelled.

The target and checkpoint workspace is a new private directory on node-local
`/tmp` (or explicitly registered `/local/scratch` via the CLI). Linux mount
inspection rejects NFS/Ceph/shared storage and never falls back to home.
If the worker reports no local filesystem/space, stop and ask site support for
the appropriate job-local scratch path; do not bypass this safety check.

Only one FP32 TRAIN attention bank is live at a time. After the student's
selected checkpoint, predictions and metrics validate, its consumed bank is
deleted. After the selected checkpoint's next bank validates, that checkpoint
is deleted. D000's checkpoint is deleted after its own validation readback.
Each job re-exports U100 locally; it never takes ownership of the shared anchor
or shares a disposable bank with the other ladder.
Permanent records explain each deletion and its last consumer. Partial failure
does not authorize cleanup of unknown files or automatic restart. Job scratch
may disappear at job end; a failed run needs a new reviewed root.

Persistent files: per-jet selected validation probabilities, labels, identities,
original50k membership, full histories, metrics, source/recipe hashes, reference
snapshots, bank manifests and retirement receipts. These support future R30 and
other prediction-based diagnostics without the models. **New inference on other
jets needs retraining after checkpoint retirement.** No older model, U100 anchor,
old bank, dataset, prepared cache, source worktree or Fresh_check is deleted.

Storage admission reserves 1 GiB persistent per arm (including technical results),
separately from scratch `4*sum(n²) +128*N +3GiB`. It is a conservative bound,
not a claim that these many persistent bytes will be written. Declare available
quota after checking your actual usage and account for concurrent campaigns.

## 1. Commit/push only these changes on Windows

```powershell
git add -- src/hlt_classification/luka_attention_weight tests/test_luka_attention_weight.py tests/test_luka_attention_reference_archive.py scripts/luka_attention_weight.py scripts/queue_luka_attention_weight.sh sbatch/run_luka_attention_weight.sh docs/plans/LUKA_ATTENTION_WEIGHT_SWEEP_PLAN.md docs/contracts/LUKA_ATTENTION_WEIGHT_SWEEP.md docs/LUKA_ATTENTION_WEIGHT_SWEEP_RUNBOOK.md
git diff --cached --stat
```

Review only the new HANDOFF/LEGACY_SOURCE_MAP sections with `git add -p` if
desired; those files contain unrelated work. Do not `git add .`. Commit reviewed
files, push without force, and copy the 40-character commit. On non-fast-forward,
stop and reconcile other chats' changes first. Source pinning allows only this
new runtime namespace relative to base 4e4a38a1; an unrelated runtime change needs
explicit compatibility review rather than weakening the check.

## 2. SPORC dry-review an arm

Run separately with `ARM=w03` (0.3/debug) and `ARM=w10` (1.0/tier3).
Supply personally available quota allocated to THIS arm, not shared total free
space. This block submits no job. Each gate uses its own fresh directory.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
read -r -p 'Arm: w03 (debug) or w10 (tier3): ' ARM </dev/tty
read -r -p 'Personally available persistent GiB allocated to this arm: ' QUOTA </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${ARM}" =~ ^(w03|w10)$ && "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_lukaweight_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_weight_${ARM}_${COMMIT:0:8}_r1"
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
bash "${PROJECT}/scripts/queue_luka_attention_weight.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${REFERENCE}" "${ARM}" "${QUOTA}" gate
printf '\nROOT=%s\n' "${ROOT}"
BASH
```

The reference campaign must be the completed original 94f56025 LOGIT/ATTN run.
Authentication runs in its pinned source using the comparison-only archival
reader. Required retained files: gate_spec.json, preflight/receipt.json,
science/campaign_spec.json, science/aggregate/{receipt,results}.json, and each
of the six science/train_LOGIT_*/train_ATTN_* directories' receipt.json,
training_report.json, validation_predictions.npz and costs.json. Hashes,
selection history, split membership, metrics and aggregate lineage are checked;
presence alone is insufficient. Old technical banks and science checkpoint/bank
payloads are not needed or certified for reuse. Their old strict validators
remain strict. Shared fixed30 U100/control dependencies must still be intact.
Compact reference predictions, metrics and a separately versioned archival
evidence record are snapshotted into the new root. Nothing old is deleted.

If registration at commit 3052c116 failed on a missing old preflight attention
file, commit/push this archival-import fix and use the NEW commit and fresh root
in step 2. That failure occurred before new root creation or job submission;
there is no reason to rerun old training or recreate its multi-GB banks. Do not
edit the pinned old worktree or reuse an old plan hash. The new arms still need
their own preflights and separately reviewed science submissions.

## 3. Submit the reviewed gate; later repeat for science

Use `MODE=gate` now. Only after that arm's gate is COMPLETED/0:0, rerun this
block with `MODE=science`. It authenticates the accepted receipt and measured
budget before producing the science plan. Review each NEW printed PLAN hash.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Commit: ' COMMIT </dev/tty
read -r -p 'Arm w03 or w10: ' ARM </dev/tty
read -r -p 'Mode gate or science: ' MODE </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${ARM}" =~ ^(w03|w10)$ && "${MODE}" =~ ^(gate|science)$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_lukaweight_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_weight_${ARM}_${COMMIT:0:8}_r1"
PARENT="${MAIN}/checkpoints/luka_fixed30_74aa58c5_r1/science/campaign_spec.json"
REFERENCE="${MAIN}/checkpoints/luka_attention_94f56025_r1/science/campaign_spec.json"
PYTHON=/home/ryreu/miniconda3/envs/atlas_kd_sporc/bin/python
QUOTA="$("${PYTHON}" -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["available_quota_gib"])' "${ROOT}/gate_spec.json")"
bash "${PROJECT}/scripts/queue_luka_attention_weight.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${REFERENCE}" "${ARM}" "${QUOTA}" "${MODE}"
read -r -p 'Paste the reviewed 64-character PLAN content_hash value: ' HASH </dev/tty
[[ "${HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Submit this ONE job, with declared owned scratch/checkpoint cleanup? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]] || exit 0
LOG="$(mktemp "${ROOT}/${MODE}-submit.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_luka_attention_weight.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${REFERENCE}" "${ARM}" "${QUOTA}" "${MODE}" --execute "${HASH}" >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\nWatch with: tail -f "%s"\n' "$!" "${LOG}" "${LOG}"
BASH
```

Wait for `QUEUE_HELPER_EXIT=0` and inspect the exact journal/job ID; a helper PID
alone is not proof of submission. These are independent partition jobs, not an
instruction to queue the same arm twice. Gate/science live claims forbid blind
retry; preserve partial submissions or failures and inspect before recovery.

## Results/progress

Logs are `<root>/preflight-<job>.out` and `<root>/science/ladder-<job>.out`.
The latter prints per-epoch AUC, attention KL and phase boundaries. Each completed
rung's permanent outputs are under `science/ladder/W03_D066` etc. Results work
after target/checkpoint retirement; original50k does not select checkpoints.

In the same isolated conda/PYTHONPATH environment as the helper:

```bash
python -u -B -s "${PROJECT}/scripts/luka_attention_weight.py" results --spec "${ROOT}/science/campaign_spec.json"
```

`science/results.json` saves contrasts with LOGIT_D000 and ATTN_D000 (lambda 0.1).
Do not call a development-validation sweep an independent test or select only
favorable rows. No scientific-quality metric gates later rungs or cleanup.
