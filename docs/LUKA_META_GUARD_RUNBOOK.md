# Guarded meta-embedding continuation

Scientific authority: [plan](plans/LUKA_META_GUARD_PLAN.md) and
[contract](contracts/LUKA_META_GUARD.md). This does not modify the original
`lukame_fixed`/`lukame_meta` jobs. It cannot guarantee no harm to validation.

## What runs

`lukamg_setup` authenticates the completed original FIXED selected full student,
borrows the existing TRAIN target bank read-only, and checks native parity,
resource limits and real five-step meta gradients. It runs disposable two-block
timing probes for both arms. It does **not** rerun teacher training, target export,
or predictor calibration. There is no separate preflight job, but inline
technical acceptance is mandatory.

`lukamg_fixed` and `lukamg_meta` depend on successful setup and can run concurrently.
Both start at the same selected FIXED encoder/classifier/**trained predictor**,
use fresh AdamW, CE+5×embedding, LR 5e-6 and 20 half-pass blocks (ten extra passes).
Both use the same TRAIN guard. META additionally trains the bounded target
adapter with the student held stationary between real blocks. The original
OFFLINE encoder and classifier are never updated.

Guard: a 0.0002 AUC tolerance, patience two blocks, up to three student/optimizer
rollbacks with LR halving. Adapter progress survives. Reporting validation is
used only for final checkpoint selection; block zero is eligible. This guard is
not an independent test and was historically seen by the original initializer.

## Prerequisite

The original campaign must have completed its FIXED job and published:

```text
/home/ryreu/atlas/HLT_Classification/checkpoints/luka_meta_embedding_7a6dc268_debug_r1/fixed/receipt.json
/home/ryreu/atlas/HLT_Classification/checkpoints/luka_meta_embedding_7a6dc268_debug_r1/fixed/fit/training_state.pt
```

Its META job need not finish. Do not cancel original FIXED to obtain an
intermediate checkpoint: the old implementation publishes its selected full
state at completion. Receipt, state, native predictions, scheduler completion,
source and bank hashes are checked; path existence is only a preliminary check.

## Commit and queue

Commit this campaign's new files plus the documentation updates, preserving
unrelated work. Push the exact commit to `origin/main`. New source is additive
relative to `c10d2205677bae4769d45da447c338165bc6724f`; existing runtime is not edited.
Use a clean detached worktree on SPORC:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_metaguard_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_meta_guard.sh"
BASH
```

Reserve at least 4 GiB personally available persistent storage. Entering a larger
free-space number does not increase the registered 4-GiB budget. The helper
creates an immutable spec, prints a dry plan, asks for its literal hash and `YES`,
then backgrounds submission with `nohup`. Wait for all three IDs in the final
ledger before logging off. A submission PID alone is not scheduler acceptance.
No manual second submission. Do not blindly retry a partially claimed root.

## Progress and results

Logs contain `META_GUARD_ADAPTER`, `META_GUARD_PROGRESS`, and `META_GUARD` markers.
Progress is measured in half-pass **blocks**, not original fixed30 epochs.
The report includes guard decisions, LR, identity-versus-adapted five-step query
CE, residual size, real losses and full-validation selection. A tiny adapter
effect should be reported as such, not hidden by the guard. Each improved full
checkpoint is saved under `fixed/fit/snapshots/` or `meta/fit/snapshots/`, including
block zero; each completed block has immutable JSON. Partial outputs are not
finished reusable results. These snapshots do not authorize automatic resume.

For results in a new login shell (the queue helper does not activate your parent
shell's environment):

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Printed guarded study directory: ' ROOT </dev/tty
SPEC="${ROOT}/campaign_spec.json"
test -s "${SPEC}"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PROJECT="$(python -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -B -s "${PROJECT}/scripts/luka_meta_guard.py" results \
    --spec "${SPEC}"
BASH
```

`ROOT` is the printed new study directory; `PROJECT` is its clean pinned worktree.
Results compare original M0HLT/OFFLINE, original OFFLINE-w5, completed parent FIXED,
and new arms. Recovery always uses historical M0HLT/OFFLINE anchors. Report
META−FIXED for the adaptation effect, not improvement over the older w5 alone.

## Code map and reuse

- `luka_meta_guard/contracts.py`: exact recipe, adapter event schedule and versions.
- `policy.py`: pure guard state machine, with no validation input.
- `training.py`: identity split, stationary adapter phases, real half-blocks,
  rollback, diagnostics and block-zero-inclusive selection.
- `source.py`, `campaign.py`: original-source replay, immutable source/data binding.
- `worker.py`: borrowed bank, full/native state agreement, probes, receipts/results.
- `submission.py`, `scripts/queue_luka_meta_guard.sh`: exact three-job DAG and dry authorization.
- Reused unchanged: `luka_meta_embedding/model.py`, `meta.py` (full five-step
  functional AdamW and bounded adapter), its TRAIN split/batch helpers, and the
  original single-OFFLINE target normalization. `luka_ce_warmstart` supplies the
  matched AdamW parameter groups.

Changing guard thresholds, splits, loss weights, lookahead, teacher, schedule or
initialization is a scientific change: version the contract/plan and use a fresh
root. Do not patch a running pinned campaign or borrow validation for meta updates.

## Validation status

Local evidence: 16 new tests plus 20 existing meta-embedding regression tests
passed across focused runs. The new CUDA worker miniature exercises shared setup
and both 20-block fits (all 475 META science updates), real installed-Weaver
operations at reduced depth, native/full checkpoint readback and tamper rejection.
A separate full-size model check covers native parity and the five-step gradient
at 5e-6. Synthetic data and mocked scheduler/source bindings are not remote
acceptance. Detailed evidence is recorded in `docs/HANDOFF.md`.
No remote jobs are submitted by implementation. SPORC installed-environment/maximum-support measurements remain
required in shared setup before either science fit is admitted. FullStat/Oscar
and Tigris are not validated execution sites for this campaign.
