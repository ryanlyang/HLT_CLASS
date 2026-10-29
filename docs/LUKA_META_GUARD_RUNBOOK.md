# Guarded meta-embedding continuation

Scientific authority: [plan](plans/LUKA_META_GUARD_PLAN.md) and
[contract](contracts/LUKA_META_GUARD.md). This does not modify the original
`lukame_fixed`/`lukame_meta` jobs. It cannot guarantee no harm to validation.

## What runs

`lukamg_setup` authenticates the explicitly chosen full student initializer,
borrows the existing TRAIN target bank read-only, and checks native parity,
resource limits and real five-step meta gradients. It runs disposable two-block
timing probes for both arms. It does **not** rerun teacher training, target export,
or predictor calibration. There is no separate preflight job, but inline
technical acceptance is mandatory.

`lukamg_fixed` and `lukamg_meta` depend on successful setup and can run concurrently.
Both start at the same authenticated encoder/classifier/**trained predictor**,
use fresh AdamW, CE+5×embedding, LR 5e-6 and 20 half-pass blocks (ten extra passes).
Both use the same TRAIN guard. META additionally trains the bounded target
adapter with the student held stationary between real blocks. The original
OFFLINE encoder and classifier are never updated.

Guard: a 0.0002 AUC tolerance, patience two blocks, up to three student/optimizer
rollbacks with LR halving. Adapter progress survives. Reporting validation is
used only for final checkpoint selection; block zero is eligible. This guard is
not an independent test and was historically seen by the original initializer.

## Starting point after cancelling the old jobs

Use **`shared_w5`**. The old setup saved a full initializer before FIXED/META
started, so their cancellation does not require rerunning them. Both new arms
start from the original OFFLINE-w5 model plus its already calibrated predictor,
not from an unfinished continuation epoch. These setup artifacts must survive:

```text
/home/ryreu/atlas/HLT_Classification/checkpoints/luka_meta_embedding_7a6dc268_debug_r1/setup/receipt.json
/home/ryreu/atlas/HLT_Classification/checkpoints/luka_meta_embedding_7a6dc268_debug_r1/initial/student.pt
```

The setup's bank, classifier, initializer manifest, references and historical
native w5 checkpoint must also remain intact. Original-source replay validates
the completed setup and all its outputs; path existence alone is insufficient.
Neither original fit must finish. Nothing in the cancelled fits is reused,
modified or deleted. Their logged improvements cannot be reconstructed from logs.

The alternative **`completed_fixed`** mode still requires completed original
FIXED `fixed/receipt.json` and `fixed/fit/training_state.pt` and native readback.
Do not choose it for a cancelled run. No automatic fallback is made between modes.
This explicit initialization choice is a version-2 scientific registration;
existing version-1 specs/worktrees remain unchanged.

## Commit and queue

### Numerical diagnostic repair (v3)

SPORC setup `21977273` at `691cccf3` completed its native/maximum-support checks
and two disposable FIXED blocks, then rejected query CE differences
`5.364418029785156e-7` and `5.960464477539063e-8` because v2 required exact zero.
Its science fits `21977274`/`21977275` were cancelled before starting. These
probe metrics are not completed science results.

V3 admits at most `1e-6` absolute query-loss difference for FIXED, retains exact
zero target residuals and signed-delta arithmetic, and preserves the raw values.
Larger differences/nonfinite data still fail. Training and resource checks are
unchanged. Commit/push the repair, use a new clean worktree and fresh root through
the helper below with `shared_w5`. Keep the failed root and its evidence intact;
do not edit its spec, resubmit its scripts or release its cancelled dependents.
The existing authenticated parent bank/predictor are reused; shared inline
checks must run again because the previous setup did not complete. There is
still no standalone preflight, teacher retraining or target export.

Commit this campaign's changed files plus the documentation updates, preserving
unrelated work. Push the exact commit to `origin/main`. V3 permits changes only
to this guard's runtime relative to `b9cf63b0bf4a037e11ece91da21db465d2347b96`.
That compatibility pin includes the separately committed FullStat fine-tuning
namespace; older scientific runtime remains unchanged and is still authenticated
in its original pinned source. Do not include unrelated runtime edits in the fix.
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
bash "${PROJECT}/scripts/queue_luka_meta_guard.sh" shared_w5
BASH
```

Reserve at least 4 GiB personally available persistent storage. Entering a larger
free-space number does not increase the registered 4-GiB budget. The helper
creates an immutable spec, prints a dry plan, asks for its literal hash and `YES`,
then backgrounds submission with `nohup`. Wait for all three IDs in the final
ledger before logging off. A submission PID alone is not scheduler acceptance.
No manual second submission. Do not blindly retry a partially claimed root.
The new root includes the new commit and initializer mode:
`luka_meta_guard_<commit8>_shared_w5_debug_r1`. Do not patch the old `a65caa7`
worktree or edit a saved spec. The reported failed launcher stopped before
creating a guarded campaign root or submitting any jobs, so it needs no cleanup.
Calling the helper without its mode argument prompts for an explicit choice.

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
Results compare original M0HLT/OFFLINE, original OFFLINE-w5, the chosen initializer
(`PARENT_SHARED_W5` or `PARENT_FIXED`), and new arms. `PARENT_SHARED_W5` has the
same native predictions as original w5; calibration changed only its predictor.
Recovery always uses historical M0HLT/OFFLINE anchors. Report
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

V3: **54 guard tests** pass across focused runs (52 CPU/contract cases plus two
installed-Weaver 0.5.3 CUDA cases). Four regression cases first reproduced the
old failure using the observed deltas and their negative counterparts. Coverage
includes exact tolerance boundaries, larger-difference rejection, unchanged
targets, nonfinite/corrupt diagnostics, signed META outcomes and old-version
rejection. GPU verification (721.89 seconds) includes full-size native parity,
hypergradients and paired diagnostics, then shared-w5 production setup and both
20-block fits, checkpoint readback and authenticated reporting on synthetic data.
The worker miniature uses reduced depth and mocked source/scheduler boundaries;
it does not establish actual SPORC acceptance. The separate completed_fixed GPU
worker variant was not rerun. Sixteen selected parent regressions, Bash syntax,
CLI help and scoped whitespace checks pass. No remote job or artifact was changed.

V2: all **25 guard tests** passed across focused runs, including a 23-test
installed-Weaver CUDA run (789.50 seconds) and two subsequently added source/spec
tests. Both initializer formats complete shared setup and both 20-block fits;
the CPU check covers 22 cases. Sixteen selected original meta-embedding regressions
also pass (four heavier original cases deselected). Cancellation-reuse tests cover explicit mode binding,
setup-only original-source authentication, no reads of cancelled fit artifacts,
preservation of calibrated predictor parameters, native-only/nonfinite rejection,
changed-file/lineage rejection, and correct initializer labels in results.
Current execution evidence is recorded in `docs/HANDOFF.md`; local tests do not
authenticate the actual remote bank or authorize automatic submission.

Historical v1 evidence: 16 new tests plus 20 existing meta-embedding regression tests
passed across focused runs. The new CUDA worker miniature exercises shared setup
and both 20-block fits (all 475 META science updates), real installed-Weaver
operations at reduced depth, native/full checkpoint readback and tamper rejection.
A separate full-size model check covers native parity and the five-step gradient
at 5e-6. Synthetic data and mocked scheduler/source bindings are not remote
acceptance. Detailed evidence is recorded in `docs/HANDOFF.md`.
No remote jobs are submitted by implementation. SPORC installed-environment/maximum-support measurements remain
required in shared setup before either science fit is admitted. FullStat/Oscar
and Tigris are not validated execution sites for this campaign.
