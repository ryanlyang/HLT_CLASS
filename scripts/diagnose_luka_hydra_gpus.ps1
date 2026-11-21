# Submit only three short diagnostics, with no dataset preparation or training release.
param([switch]$Status)
$ErrorActionPreference='Stop'
$project=Split-Path -Parent $PSScriptRoot
$helper=Join-Path $project 'src/hlt_classification/operations/hydra_reconstruction_gpu_diagnostic.py'
$digest=(Get-FileHash -LiteralPath $helper -Algorithm SHA256).Hash.ToLowerInvariant()
$key=Join-Path $env:USERPROFILE '.ssh/id_rsa_brown_cs'
$remote="/cs/home/rlyang/hlt_u100_pairs/gpu_diagnostic_${digest}.py"
if ($Status) {
    & ssh -i $key -o ServerAliveInterval=30 -o ServerAliveCountMax=6 `
      rlyang@ssh.cs.brown.edu "/usr/bin/python3 -I -u -B '$remote' status --expected-sha256 '$digest'"
    if ($LASTEXITCODE -ne 0) {throw 'Diagnostic status query failed.'}
    return
}
& scp -O -i $key $helper "rlyang@ssh.cs.brown.edu:${remote}"
if ($LASTEXITCODE -ne 0) {throw 'Diagnostic upload failed; no jobs submitted.'}
$launch=@'
set -euo pipefail
HELPER="$1"; DIGEST="$2"
LOG="$(mktemp "${HELPER%.py}.submit.XXXXXX.log")"
nohup /usr/bin/python3 -I -u -B "$HELPER" submit --expected-sha256 "$DIGEST" \
  --authorize AUTHORIZE_HYDRA_RECONSTRUCTION_GPU_DIAGNOSTIC_V1 </dev/null >"$LOG" 2>&1 &
PID="$!"
printf 'Detached diagnostic submission PID: %s\nLog: %s\n' "$PID" "$LOG"
tail -n 60 --pid="$PID" -f "$LOG"
grep -q '^DIAGNOSTIC OUTPUT:' "$LOG"
'@
$launch | & ssh -i $key -o ServerAliveInterval=30 -o ServerAliveCountMax=6 `
    rlyang@ssh.cs.brown.edu "tr -d '\r' | bash -s -- '$remote' '$digest'"
if ($LASTEXITCODE -ne 0) {Write-Warning 'Inspect the submission log/receipts before retrying; do not submit twice.'}
