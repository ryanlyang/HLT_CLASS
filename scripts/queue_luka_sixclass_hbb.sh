#!/usr/bin/env bash
# Exactly one validation-only analysis; no training campaign submission.
set -euo pipefail
[[ $# -eq 1 || ( $# -eq 2 && "$2" == resume ) ]]
COMMIT="$1"; MODE="${2:-new}"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$(git -C "$PROJECT" rev-parse HEAD)" == "$COMMIT" ]]
git -C "$PROJECT" merge-base --is-ancestor "$COMMIT" origin/main
git -C "$PROJECT" diff --quiet HEAD -- src scripts sbatch
BASE=/oscar/home/rlyang/hlt_classification
SPEC="$BASE/checkpoints/luka_sixclass_balanced_oscar_8a25f126_r1/campaign_spec.json"
OUT="$BASE/reports/sixclass_hbb_${COMMIT:0:8}_r1"
LAUNCH="${OUT}.launch"
test -f "$SPEC"
if [[ "$MODE" == new ]]; then
    mkdir -p "$BASE/reports"
    mkdir "$LAUNCH"
    [[ ! -e "$OUT" && ! -L "$OUT" ]]
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
      --cpus-per-task=6 --mem=192000M --gres=gpu:l40s:1 --time=48:00:00
      --job-name=luka6_hbb_eval --comment="hbb-full-val:${COMMIT}" --export=NONE
      --chdir="$PROJECT" --output="$LAUNCH/evaluate-%j.out"
      "$PROJECT/sbatch/run_luka_sixclass_hbb.sh" "$PROJECT" "$SPEC" "$OUT")
printf '%q ' sbatch "${ARGS[@]}" > "$ATTEMPT/command.txt"
printf '\n' >> "$ATTEMPT/command.txt"
sbatch --test-only "${ARGS[@]}"
JOB="$(sbatch --parsable "${ARGS[@]}")"
printf '%s\n' "$JOB" > "$ATTEMPT/submission.txt"
[[ "$JOB" =~ ^[1-9][0-9]*$ ]]
printf '%s\n' "$JOB" > "$LAUNCH/latest_job"
printf 'QUEUED validation-only evaluation: %s\nLog: %s/evaluate-%s.out\nReport: %s/report.json\n' "$JOB" "$LAUNCH" "$JOB" "$OUT"
