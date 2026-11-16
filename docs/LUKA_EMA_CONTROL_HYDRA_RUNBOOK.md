# Nine-fit CE / U100 / R24 EMA comparison on Hydra

One A6000 per fit; three objectives at seeds 7/8/9, batch 128, 30 epochs, FP32.
Every fit records RAW, RAW_BN, EMA05 and EMA10. EMA05 is primary.
The CE-only arm establishes whether averaging alone explains the earlier
EMA gains. See the [plan](plans/LUKA_EMA_CONTROL_HYDRA_PLAN.md),
[contracts](contracts/LUKA_EMA_CONTROL_HYDRA.md) and
[validation evidence](LUKA_EMA_CONTROL_HYDRA_VALIDATION.md).

No new data export is needed. The existing 3ccc35ae stability spec and inputs
remain read-only. This uses new code/output directories. Commit and push the
new implementation first; use its full 40-character commit below. The
registration-time serialized inventory prints the exact output allowance if
the supplied personal free-space budget is insufficient.
The local 298-slot inventory is 9 GiB plus 1 GiB reserve: at least 10 GiB personally
available. Hydra remeasures serialization and actual setup outputs.

Run this on a Hydra login node. It registers the separate campaign and submits
setup plus its durable dependent release. Once both IDs are recorded, logging
off is safe. Successful setup automatically submits the nine fits and report.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed EMA-control commit (40 characters): ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
BASE="$(cd "$HOME/hlt_u100_pairs" && pwd -P)"
PARENT="${BASE}/runs/reconstruction_stability_3ccc35ae_4x1080ti_r1/campaign_spec.json"
test -f "$PARENT"
MAIN="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$PARENT")"
git -C "$MAIN" fetch origin main
git -C "$MAIN" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="${BASE}/code_ema_control_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$PARENT")"
ROOT="${BASE}/runs/ema_control_${COMMIT:0:8}_a6000_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available GiB for new outputs: ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
LOG="$(mktemp "${BASE}/runs/ema_control_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_ema_control_hydra.py" create \
    --root "$ROOT" --parent-spec "$PARENT" --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB" >"$LOG" 2>&1; then
    tail -n 100 -- "$LOG"
    printf 'Registration stopped; preserve any new root and log.\n' >&2
    exit 1
fi
printf 'Registered spec: %s\n' "$ROOT/campaign_spec.json"
bash "$PROJECT/scripts/queue_luka_ema_control_hydra.sh" "$ROOT/campaign_spec.json"
squeue --me -o "%.18i %.30j %.12T %.12l %R"
BASH
```

Initially expect `lukaema_setup_a6000` and `lukaema_release_a6000`.
Setup requests 6 CPUs, 32000 MiB and one `gpu:nvidia_rtx_a6000:1` for at most 6 hours.
The CPU release depends afterok; setup failure releases no science. It checks
all nine miniatures, objective parity, actual resources, original checkpoint
replay and unchanged native export. Scientific queue order is:

```text
CE_s7 U_s7 R24_s7
CE_s8 U_s8 R24_s8
CE_s9 U_s9 R24_s9
report_a6000 (afterany on the nine fits)
```

These fits can run concurrently when GPUs are available. Timing is measured
on the allocated A6000, not inferred from older GPU runs. If dry scheduler
review fails, retain the registered spec/root; do not create duplicates or
alter the immutable hardware fields. If setup fails, retain its log and
diagnostics. Neither event should change the existing stability campaign.

To inspect completed results and epoch progress, run the following. It
requires exactly one matching campaign; if several exist, set SPEC explicitly
to the desired printed campaign_spec.json instead of selecting a newest path.
Authentication can take a few minutes. No checkpoints are evaluated anew.

```bash
bash <<'BASH'
set -euo pipefail
shopt -s nullglob
SPECS=("$HOME"/hlt_u100_pairs/runs/ema_control_*_a6000_r1/campaign_spec.json)
[[ ${#SPECS[@]} -eq 1 ]] || { printf 'Set SPEC to the exact campaign; found %s.\n' "${#SPECS[@]}"; exit 1; }
SPEC="${SPECS[0]}"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
for MODE in results progress; do
    printf '\n%s — EMA05 primary: %s\n' "$MODE" "$SPEC"
    "${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_ema_control_hydra.py" \
        "$MODE" --spec "$SPEC" --readout EMA05 |
        awk '/^[[:space:]]*\{[[:space:]]*$/ {json=1} !json {print}'
done
BASH
```

Use `--readout all` to compare the four separate streams, or RAW/RAW_BN/EMA10
individually. `control` is CE for U and U for R24, always at the same seed and
readout. Progress reports selected-control and same-epoch deltas separately.
Seed summaries remain partial until 3/3 complete. These are development results,
not independent-test evidence. An EMA05 CE result is genuinely CE-only; earlier
C128 EMA results included U100-w5 and are not substitutes for this control.
