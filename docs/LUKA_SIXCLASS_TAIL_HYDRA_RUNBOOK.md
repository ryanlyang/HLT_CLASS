# Transfer the Oscar teachers and run DE050 -> DE000 on Hydra L40

This runs **two fresh 20-epoch fits**, preserving the six-class 600–2600 GeV
balanced Oscar recipe. It leaves Oscar and the older Hydra campaign unchanged.
The new source must first be committed and pushed. Do not stage unrelated work
just to launch this campaign. Files for this change are the new
`luka_sixclass_tail_hydra/` package, the five new `*sixclass_tail*` scripts,
two Slurm scripts, focused tests, this runbook, plan/contract/validation docs,
and their new index/handoff/donor-map entries.

## 1. Export a small manifest on Oscar

In an authenticated Oscar Bash terminal, paste this block. Enter the pushed
40-character commit containing this implementation. The source campaign is
identified by the original OFFLINE job 9198207; its four completed jobs are
authenticated before export. Authentication reads saved outputs and TRAIN
metadata; it performs no inference or new training.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Pushed 40-character Hydra-tail commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
BASE=/oscar/home/rlyang/hlt_classification
REPO="$BASE/source/HLT_Classification_new_balanced_oscar_f9dfc80b"
PROJECT="$BASE/source/HLT_Classification_sixclass_tail_${COMMIT:0:8}"
OUT="$BASE/reports/sixclass_tail_teachers_${COMMIT:0:8}.json"
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
[[ ! -e "$OUT" && ! -L "$OUT" && ! -e "$OUT.launch" ]]
mkdir -p "$BASE/reports"
mkdir "$OUT.launch"
nohup bash "$PROJECT/scripts/export_luka_sixclass_tail.sh" "$OUT" >"$OUT.launch/launcher.log" 2>&1 </dev/null &
printf 'Export detached, PID %s. Log: %s.launch/launcher.log\n' "$!" "$OUT"
tail -n 60 -f "$OUT.launch/launcher.log"
BASH
```

After `BUNDLE:`, `HASH:` and `PAYLOAD:` appear, Ctrl-C stops only the tail.
Copy the manifest path and full 64-character hash. The manifest is small;
payloads remain in their original Oscar locations until streamed. Export
checks all four selected banks and all original epoch states, so it can take
several minutes. Do not repeat the export while its launcher is working.

## 2. Download to Windows, then upload to Hydra

In local PowerShell at the repository, paste this block. Keep this terminal
open during the actual file transfers. Existing verified files survive an
interruption; rerun with the same manifest/hash to resume verification.

```powershell
$Python = 'C:\Users\22rya\miniconda3\envs\tagging-hlt\python.exe'
$Manifest = Read-Host 'Oscar BUNDLE manifest path'
$BundleHash = Read-Host '64-character HASH printed by export'
if ($BundleHash -notmatch '^[0-9a-f]{64}$') { throw 'Invalid bundle hash' }
$LocalBundle = Join-Path $PWD "artifacts\transfers\sixclass_oscar_$($BundleHash.Substring(0,16))"
& $Python -B -s scripts\transfer_luka_sixclass_tail.py fetch `
    --manifest $Manifest --sha256 $BundleHash --root $LocalBundle
if ($LASTEXITCODE -ne 0) { throw 'Oscar download stopped; inspect the message above' }
& $Python -B -s scripts\transfer_luka_sixclass_tail.py send `
    --sha256 $BundleHash --root $LocalBundle `
    --identity-file "$env:USERPROFILE\.ssh\id_rsa_brown_cs"
if ($LASTEXITCODE -ne 0) { throw 'Hydra upload stopped; inspect the message above' }
```

OpenSSH handles CERN/Brown credentials through the terminal; the code never
reads key bytes. The Oscar hostname is `rlyang@ssh.ccv.brown.edu`, the Hydra
gateway is `rlyang@ssh.cs.brown.edu`. A non-default Oscar identity can be passed
to **fetch** via `--identity-file` if that is how the working Oscar login works.
The Hydra destination is fixed to
`/cs/home/rlyang/hlt_u100_pairs/imports/sixclass_oscar_<first16hash>`.
The raw dataset already exists on Hydra and is not transferred again.

## 3. Queue on Hydra, detached

After `TEACHER TRANSFER VERIFIED` for the Hydra destination, paste this in an
authenticated Hydra Bash terminal. Use current free GiB **after** the upload,
reserving space for other campaigns. At least 14 GiB must remain available for
this execution. The manifest gives the exact import size; do not assume the
earlier 31 GiB free-space reading is still current.

```bash
bash <<'BASH'
set -euo pipefail
BASE=/cs/home/rlyang/hlt_u100_pairs
df -h "$BASE"
read -r -p 'Pushed 40-character Hydra-tail commit: ' COMMIT </dev/tty
read -r -p '64-character bundle HASH: ' BUNDLE </dev/tty
read -r -p 'Free HOME GiB available for this run, after import (minimum 14): ' QUOTA </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$BUNDLE" =~ ^[0-9a-f]{64}$ && "$QUOTA" =~ ^[0-9]+$ ]]
REPO="$BASE/code_65952ed3"
PROJECT="$BASE/code_sixclass_tail_${COMMIT:0:8}"
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
bash "$PROJECT/scripts/queue_luka_sixclass_tail_hydra.sh" "$COMMIT" "$BUNDLE" "$QUOTA"
BASH
```

**You can disconnect after `DETACHED Hydra tail launcher PID ...`.**
Registration, bank authentication, scheduler dry review and submission continue
without the terminal. The tail is only a view. `LAUNCHER_EXIT=0` plus the two
`SUBMITTED` records confirms success; detachment alone does not mean submission
has succeeded. Do not launch the same run twice.

Both jobs request one L40, six CPUs, 192000 MiB RAM and 48 hours. DE000 waits
for the new Hydra DE050 to finish. No separate preflight or preparation jobs
are submitted. The in-allocation miniature and timing/memory checks remain.
The runtime is the existing
`$BASE/environment_pair_hydra_v1/venv`; incompatible or damaged installations
stop with evidence rather than being silently altered.

## 4. Progress and selected results

Set `COMMIT` to the same full commit, then use:

```bash
BASE=/cs/home/rlyang/hlt_u100_pairs
PROJECT="$BASE/code_sixclass_tail_${COMMIT:0:8}"
ROOT="$BASE/runs/sixclass_tail_${COMMIT:0:8}_r1"
ENV_DIR="$BASE/environment_pair_hydra_v1/venv"
squeue --me -o '%.18i %.42j %.2t %.10M %R'
PYTHONNOUSERSITE=1 "$ENV_DIR/bin/python" -B -s "$PROJECT/scripts/luka_sixclass_tail_hydra.py" \
    progress --spec "$ROOT/campaign_spec.json" --stage DE050
```

Change `DE050` to `DE000` for the last rung. Progress reads small saved
metadata, showing per-epoch accuracy/AUC and recovery against imported Oscar
six-class M0HLT/OFFLINE. It does not reread prediction/model payloads or query
scheduler state. For selected accuracy/AUC/R50/R30 and recoveries after
publication, use `results --spec "$ROOT/campaign_spec.json"` instead.
Results also show outside-range diagnostics separately. No test is evaluated.

## 5. Explicit recovery only

Preserve all claims, journals and failed attempts. If a run fails, first inspect
its exact job accounting and log. Automatic resubmission is deliberately absent.
After **all previously submitted Hydra jobs for this new root are terminal**,
the same source supports epoch-boundary continuation:

```bash
IMPORT="$BASE/imports/sixclass_oscar_${BUNDLE:0:16}"
bash "$PROJECT/scripts/launch_luka_sixclass_tail_hydra.sh" resume-detach \
    "$PROJECT" "$ROOT" "$COMMIT" "$QUOTA" "$ENV_DIR" "$IMPORT" "$BUNDLE" \
    --authorize AUTHORIZE_SIXCLASS_TAIL_HYDRA_V1
```

The implementation checks terminal accounting itself, verifies committed
states, and skips already published fits. A missing submission receipt is
ambiguous and requires reconciling its exact job ID first. Do not delete it.
