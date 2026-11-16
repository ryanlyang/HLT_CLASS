# SOFT recovery and repeat-report audit

2026-11-16. This addendum governs a separate numerical recovery of three failed
fits from `reconstruction_stability_3ccc35ae_4x1080ti_r1`. It does not change the
registered 42-fit campaign or its running jobs. The user requested fixing the
failures before investigating CE-only with EMA.

## Evidence and scope

The current campaign's only failed fits in the supplied accounting were
SOFT_s7 (180535), SOFT_s8 (180541), and SOFT_s9 (180547). All failed in epoch4
because the 500-iteration transport solve exceeded its numerical tolerances.
Reported marginal residuals were 0.000131952, 0.00278396, and 0.000803573.
These were solver convergence failures, not scientific quality gates or OOMs.

The older repeatability report176453 failed an initialization-pairing check.
The supplied records locate a different native hash in R20/R22/R28/D20_s4
compared with C_s4. R24_s4 matches; all seed5/6 starts match. Predictor, sampler
and recorded dropout seeds match throughout. A whole-model hash cannot tell
whether the native difference is a tiny numerical initialization difference or
a substantial one. Its cause remains unresolved. Never bypass this check and
silently call all those runs matched.

## Numerical protocol and contracts

`soft_dual_newton/v2` preserves the original cost matrix, complete real/null
masses, entropy reference measure, temperature scale/schedule, loss coefficient,
and detached optimal-plan gradient. The original 500-iteration FP64 log-scaling
path runs first, giving exactly the original result when it converges. Otherwise
its column potentials initialize a row-normalized dual Newton solver, with a
fixed gauge, 1e-12 Hessian regularization, potential-step cap50, at most256
iterations and40 backtracking trials. The rare refinement uses CPU FP64 with
one BLAS thread per rank. Final marginal residual AND normalized primal/dual gap
must still be <=1e-4. No tolerance weakening, cropping or hard fallback occurs.
Numerically different trajectories are possible and are explicitly versioned.

`LUKA_RECONSTRUCTION_STABILITY_RECOVERY_SPEC/v1` binds the new clean pushed
implementation, the exact original spec/source3ccc35ae, the three failed IDs,
error logs, initializations and old calibration files. `RECOVERY_RECIPE/v1`
parents the original recipe and records numerical v2. All ordinary output
contracts retain their shape and bind the new spec/recipe hashes, so replacement
outputs cannot be mistaken for old outputs. The original v1 execution remains
unchanged. Six GiB personally available space is required (five GiB output
allowance plus one GiB reserve); measured setup storage may reject that allowance.

Reuse the exact512 calibration identities, scale and initialization from the
old setup. Recompute and compare the8192 TRAIN BN membership. Require every
replacement's native/predictor/decoder initialization, data order and dropout
seed identities to match its failed fit. Restart from scratch for30 epochs;
the failed jobs did not publish resumable training checkpoints. No results
are stitched across epochs or numerical protocols.

## Setup, automatic release and reports

Six jobs total: `lukarsf_setup_pascal`, its dependent CPU release, three
independent SOFT fits, then an afterany report. Setup/fits request one node,
four GTX1080Ti,12 CPUs,64000MiB. Existing six-hour setup /48-hour fit ceilings
and measured2x safety remain. Submission uses the same reviewed exact-ID DAG
and idempotent journals. Other original fits and its report are untouched.

Setup repeats common-start four-rank parity at3/3.01/4.5/8.99/9, all three
production miniatures and native export reloads, full-support stress and full
validation/readout timings. It additionally trains a **discarded seed8 model
through four complete TRAIN passes**, covering the epoch4 failure regime.
It never selects weights by validation in this probe. It tests difficult298-slot
transport matrices on every GPU, measures evolved-model steps at4.5/8.99/20,
and uses the slower of fresh/evolved timings in resource projections.
`EVOLVED_SOFT_PROBE/v1` authenticates this evidence. This setup will take longer
than the original bounded miniature; four passes do not prove all future
cost matrices will converge. If a future solve exhausts its bound, each failing
rank saves the exact finite cost/masses/temperature plus task/update context in
`transport_failure_rankN.pt` and fails closed.

Recovery `results` authenticates the original C128 controls, uses the same
readout for deltas, and withholds a delta if starting identities differ. RAW
remains primary; RAW_BN/EMA05/EMA10 are separate correlated readouts.
`progress` shows saved replacement epochs; missing controls are shown as absent.

The `repeat-audit` command reads the old18-fit artifacts without a future-write
space gate. It validates source, manifests, task receipts, histories, exported
files, identities and predictions. It preserves individually valid results and
prints no paired delta for the four mismatched seed4 fits. Its distinct
`REPEAT_FAILURE_AUDIT/v1` result records the mismatch; the old failed report is
not overwritten or declared repaired in place.

## Queue after committing and pushing the fix

On Brown, use a fresh detached checkout of the new pushed commit. The original
`code_reconstruction_stability_3ccc35ae` checkout must remain untouched.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed fix commit (40 characters): ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
BASE="$(cd "$HOME/hlt_u100_pairs" && pwd -P)"
PARENT="$BASE/runs/reconstruction_stability_3ccc35ae_4x1080ti_r1/campaign_spec.json"
MAIN="$BASE/code_reconstruction_stability_3ccc35ae"
git -C "$MAIN" fetch origin main
git -C "$MAIN" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/code_soft_recovery_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$PARENT")"
ROOT="$BASE/runs/stability_soft_recovery_${COMMIT:0:8}_4x1080ti_r1"
test ! -e "$ROOT"
read -r -p 'Personally available GiB (at least 6): ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONPATH
"$ENV_DIR/bin/python" -u -B -s "$PROJECT/scripts/luka_reconstruction_stability_hydra.py" recover-soft \
    --parent-spec "$PARENT" --root "$ROOT" --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB"
bash "$PROJECT/scripts/queue_luka_reconstruction_stability_hydra.sh" "$ROOT/campaign_spec.json"
squeue --me -o '%.18i %.32j %.12T %R'
BASH
```

Using the same PROJECT/ENV_DIR environment, inspect the older report with:

```bash
"$ENV_DIR/bin/python" -u -B -s "$PROJECT/scripts/luka_reconstruction_stability_hydra.py" repeat-audit \
    --repeat-spec "$HOME/hlt_u100_pairs/runs/reconstruction_repeat_15c35c42_4x1080ti_r1/campaign_spec.json"
```

## CE-only + EMA is the next experiment

C128's roughly55% EMA recovery is **CE + U100 embedding weight5 + EMA**. It is
not evidence for CE-only. After this repair, the clean comparison is three fresh
CE-only fits at the same seed indices7/8/9, native ParT, global128,30 epochs,
same initialization/data/dropout, and all four readouts with the same TRAIN BN
membership. Compare CE RAW/RAW_BN/EMA05/EMA10 against the corresponding C128
readout. This isolates the added benefit of U100 under EMA. This recovery does
not submit or silently add those CE-only fits.
