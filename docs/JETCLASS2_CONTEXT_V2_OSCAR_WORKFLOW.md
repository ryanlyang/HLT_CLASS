# CONTEXT_V2: Oscar dataset and automatic direct/coarse training

This is a synthetic context-encoding experiment, not calibrated CMS HLT.
Only the shear amplitude changes, 0.65 to 1.0; the three layers, underlying
noise draws, counts, particle IDs, p4 and masks remain the same as CONTEXT_V1.
The desired 1.5–2 percentage-point baseline accuracy gap is not guaranteed.
There is no performance-based admission gate or repeated strength search.

## What runs

```text
2.25M generation array (CPU, max 8 concurrent × 6 CPUs)
  -> complete-dataset verification
  -> matching and 100k/50k training preparation
  -> ordinary L40S runtime/parity check
  -> automatic submission of all 16 DIRECT + COARSE science jobs
```

No extra tuning pilot. The first block of each generation shard is replayed
inside that job as an integrity check. The GPU runtime check measures memory
and walltime and exercises installed-Weaver parity; its models are discarded.
After successful setup, science needs no further manual approval or submission.
Invalid inputs, insufficient resources or a failed prerequisite stop followup.
Poor scientific performance does not stop any registered science row.

The science plan has fresh M0HLT, OFFLINE and U000 controls, DIRECT D000,
COARSE U050/U100/D066/D033/D000, five reducers, aggregation and completion.
The 100k training and 50k validation identities are copied byte-for-byte from
the original V1 release index. The full data splits remain 1M/250k/1M; final
test is generated but never evaluated by this workflow.

## Prepare the pinned worktree

First commit and push the implementation, including its new modules, scripts,
tests, plan and contract. Do not include SSH keys or unrelated work. Then on
Oscar use the exact new commit (not the historical V1 commit):

```bash
read -r -p 'New pushed 40-character commit: ' COMMIT
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]] || exit 1
MAIN=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
SHORT=${COMMIT:0:8}
PROJECT=/oscar/home/rlyang/hlt_classification_context_v2_${SHORT}
ROOT=/oscar/home/rlyang/hlt_classification/checkpoints/jc2_context_v2_${SHORT}_r1
DATASET_ROOT=/oscar/home/rlyang/datasets/jetclass2_context_v2_2250k_${SHORT}_r1

git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main || exit 1
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}" || exit 1
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)" || exit 1
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}" || exit 1
fi
checkquota
df -h /oscar/home/rlyang
```

The helper authenticates the original gate at
`/oscar/home/rlyang/hlt_classification/checkpoints/jc2_context_100k50k_direct_coarse_f889f359_r1/gate/gate_spec.json`.
Its release and the already-transferred V1 dataset must remain in place. The
new dataset also retains those provenance references; it is not yet a portable
self-contained export. No additional RIT transfer or source ROOT copy is needed.

## One dry review, then execute the whole workflow

Use **available quota**, not filesystem-wide free space. Require at least 30
GiB available for the bounded 20-GiB data reservation plus at least 10 GiB of
training/log headroom. Concurrent unrelated jobs can consume that headroom;
recheck quota before executing. Both chosen roots must be persistent and fresh
for creation. A completed dry review may be reused; do not delete partial roots.

```bash
read -r -p 'Available personal quota in GiB (at least 30): ' QUOTA
bash "${PROJECT}/scripts/queue_jetclass2_context_v2_workflow.sh" \
    "${COMMIT}" "${ROOT}" "${DATASET_ROOT}" "${QUOTA}"
```

Read the five initial submissions, downstream 16-job policy and plan hash.
Then authorize precisely that plan, including the automatic science followup:

```bash
read -r -p 'Reviewed 64-character plan content_hash: ' PLAN_HASH
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]] || exit 1
LOG=$(mktemp "${ROOT}/queue-submit.XXXXXX.log")
nohup bash -c '
  result=0
  bash "$@" || result=$?
  printf "\nQUEUE_HELPER_EXIT=%s\n" "$result"
  exit "$result"
' _ "${PROJECT}/scripts/queue_jetclass2_context_v2_workflow.sh" \
    "${COMMIT}" "${ROOT}" "${DATASET_ROOT}" "${QUOTA}" \
    --execute "${PLAN_HASH}" >"${LOG}" 2>&1 </dev/null &
printf 'Submission PID: %s\nLog: %s\n' "$!" "${LOG}"
```

This helper survives SSH disconnects. Confirm `QUEUE_HELPER_EXIT=0` and the
five initial job IDs before assuming submission succeeded. `Ctrl-C` on `tail`
does not cancel jobs. After submission, Slurm dependencies handle the chain;
the science launcher is itself a scheduled job, not a login-node polling loop.

## Inspect progress/results

```bash
tail -n 40 "${LOG}"
squeue --me -o '%.18i %.12P %.60j %.2t %.10M %R'
export PROJECT_DIR="${PROJECT}" JC2_SITE=oscar_l40s
source "${PROJECT}/sbatch/jetclass2_delphes_common.sh"
python -s "${PROJECT}/scripts/jetclass2_context_v2_workflow.py" status \
    --spec "${ROOT}/workflow_spec.json"
# Once science exists, saved validation rows (including unfinished rows):
python -s "${PROJECT}/scripts/jetclass2_context_v2_workflow.py" results \
    --spec "${ROOT}/workflow_spec.json"
```

The dataset publishes `dataset_manifest.json` only after every shard verifies.
`science_submission_receipt.json` identifies the automatic science submission.
Do not interpret file presence or Slurm completion as evidence the ladder won.
Submission is journaled with exclusive claims; interrupted/partial submissions
need explicit inspection, never blind rerunning or deleting claims. Existing
V1 data, jobs and worktrees are not modified or canceled.
