#!/usr/bin/env bash
# Separate validation-only analysis. No changes to scientific jobs or outputs.
set -euo pipefail
shopt -s nullglob
[[ $# -eq 1 || ( $# -eq 2 && "$2" == resume ) ]]
COMMIT="$1"; MODE="${2:-new}"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$(git -C "$PROJECT" rev-parse HEAD)" == "$COMMIT" ]]
git -C "$PROJECT" merge-base --is-ancestor "$COMMIT" origin/main
[[ -z "$(git -C "$PROJECT" status --porcelain=v1 --untracked-files=all)" ]]
BASE=/oscar/home/rlyang/hlt_classification
LOGS=("$BASE"/checkpoints/luka_sixclass_low_lr_oscar_*/train_OFFLINE-9442678.out)
[[ ${#LOGS[@]} -eq 1 ]] || { echo 'STOP: expected one campaign for OFFLINE job 9442678'; exit 1; }
SPEC="$(dirname "${LOGS[0]}")/campaign_spec.json"
OUT="$BASE/reports/sixclass_full_recovery_${COMMIT:0:8}_r1"
LAUNCH="${OUT}.launch"
test -f "$SPEC"
if [[ "$MODE" == new ]]; then
    mkdir -p "$BASE/reports"
    [[ ! -e "$OUT" && ! -L "$OUT" ]]
    mkdir "$LAUNCH"
    ATTEMPT="$LAUNCH/initial"
else
    test ! -f "$OUT/report.json"
    test -f "$LAUNCH/latest_job"
    OLD_JOB="$(cat "$LAUNCH/latest_job")"
    [[ "$OLD_JOB" =~ ^[1-9][0-9]*$ ]]
    [[ -z "$(squeue -h -j "$OLD_JOB" -o '%i')" ]]
    STATE="$(sacct -X -n -P -j "$OLD_JOB" -o State | head -n 1 | cut -d'|' -f1)"
    case "$STATE" in FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|CANCELLED*) ;; *) echo "STOP: previous job state $STATE"; exit 1;; esac
    ATTEMPT="$LAUNCH/resume_after_${OLD_JOB}"
fi
mkdir "$ATTEMPT"
while IFS='=' read -r KEY _; do
    case "$KEY" in SBATCH_*|SLURM_*) unset "$KEY";; esac
done < <(env)
ARGS=(--partition=gpu --account=default --qos=norm-gpu --nodes=1 --ntasks=1
      --cpus-per-task=6 --mem=192000M --gres=gpu:l40s:1 --time=24:00:00
      --job-name=luka6_full_eval --comment="full-validation:${COMMIT}" --export=NONE
      --chdir="$PROJECT" --output="$LAUNCH/evaluate-%j.out"
      "$PROJECT/sbatch/run_luka_sixclass_full_validation.sh" "$PROJECT" "$SPEC" "$OUT")
printf '%q ' sbatch "${ARGS[@]}" > "$ATTEMPT/command.txt"
printf '\n' >> "$ATTEMPT/command.txt"
sbatch --test-only "${ARGS[@]}"
JOB="$(sbatch --parsable "${ARGS[@]}")"
printf '%s\n' "$JOB" > "$ATTEMPT/submission.txt"
[[ "$JOB" =~ ^[1-9][0-9]*$ ]]
printf '%s\n' "$JOB" > "$LAUNCH/latest_job"
printf 'QUEUED validation-only evaluation: %s\nLog: %s/evaluate-%s.out\nReport: %s/report.json\n' "$JOB" "$LAUNCH" "$JOB" "$OUT"
