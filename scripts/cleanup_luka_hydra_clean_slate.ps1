# Prepare a pushed standalone-policy checkout, then retire previous workspace artifacts.
param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$Commit,
    [switch]$Preview
)
$ErrorActionPreference='Stop'
$project=Split-Path -Parent $PSScriptRoot
$helper=Join-Path $project 'src/hlt_classification/operations/hydra_clean_slate.py'
$digest=(Get-FileHash -LiteralPath $helper -Algorithm SHA256).Hash.ToLowerInvariant()
$key=Join-Path $env:USERPROFILE '.ssh/id_rsa_brown_cs'
$remote="/cs/home/rlyang/hlt_u100_pairs/clean_slate_${digest}.py"
$mode=if ($Preview) {'preview'} else {'execute'}
& scp -O -i $key $helper "rlyang@ssh.cs.brown.edu:${remote}"
if ($LASTEXITCODE -ne 0) {throw 'Helper upload failed; cleanup was not started.'}
$launch=@'
set -euo pipefail
HELPER="$1"; DIGEST="$2"; COMMIT="$3"; MODE="$4"
BASE=/cs/home/rlyang/hlt_u100_pairs
REPO="$BASE/code_65952ed3"
PROJECT="$BASE/code_balanced_reco_${COMMIT:0:8}"
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
ARGS=(--project "$PROJECT" --commit "$COMMIT" --expected-sha256 "$DIGEST")
case "$MODE" in
    preview) ;;
    execute) ARGS+=(--execute --authorize AUTHORIZE_HYDRA_NEW_DATASET_CLEAN_SLATE_V1) ;;
    *) exit 2 ;;
esac
LOG="$(mktemp "${HELPER%.py}.${MODE}.XXXXXX.log")"
nohup /usr/bin/python3 -I -u -B "$HELPER" "${ARGS[@]}" </dev/null >>"$LOG" 2>&1 &
PID="$!"
printf 'Detached clean-slate PID: %s\nLog: %s\n' "$PID" "$LOG"
printf 'Ctrl+C closes this viewer; cleanup continues. Look for COMPLETE or STOP.\n'
tail -n 60 --pid="$PID" -f -- "$LOG"
if grep -q '^STOP:' "$LOG"; then exit 1; fi
if [[ "$MODE" == execute ]]; then
    grep -q '^COMPLETE:' "$LOG" || { echo 'STOP: cleanup did not publish completion'; exit 1; }
fi
'@
$launch | & ssh -i $key -o ServerAliveInterval=30 -o ServerAliveCountMax=6 `
    rlyang@ssh.cs.brown.edu "tr -d '\r' | bash -s -- '$remote' '$digest' '$Commit' '$mode'"
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'SSH/log viewing ended. Inspect the printed log and retirement records before retrying.'
}
