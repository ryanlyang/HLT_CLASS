# Student-benefit OFFLINE-w5 continuation

Authority: [active scientific plan](plans/LUKA_META_EMBEDDING_PLAN.md) and
[contracts v1](contracts/LUKA_META_EMBEDDING.md).

Two 30-epoch continuations on **SPORC debug**. The control retains frozen
OFFLINE embedding targets. Treatment adjusts a small bounded teacher-side
adapter using classification loss after five differentiable HLT-student
updates. Both keep CE1 + embedding5, without KD. All auxiliary machinery is
absent at inference. Validation is never a meta-training objective.

The saved w5 checkpoint contains native encoder/classifier weights, **not its
old predictor**. Shared setup rebuilds/calibrates that small head once with
native weights and buffers frozen. Both fits start from the exact same resulting
full checkpoint, use fresh AdamW and the same lower-LR30 schedule. Historical
w5 already saw the TRAIN-internal query subset: do not call that an unseen test.

## Exact prerequisites

- Original fixed30 teacher/controls and their metadata/worktree:
  `checkpoints/luka_fixed30_74aa58c5_r1/science/campaign_spec.json`.
- Original selected OFFLINE-w5 and metadata/worktree:
  `checkpoints/luka_embedding_sweep_007885f7_tier3_r1/science_debug/campaign_spec.json`,
  `w5/fit/selected.pt`, report, predictions and receipts. Original worktree commit
  `855c4f8d40b4db7837238c7ac6a3799597466e85` must remain clean.
- Authenticated prepared HLT views, original raw OFFLINE TRAIN source, original
  pinned environment `atlas_kd_sporc`. Missing old target payloads are permitted;
  the new target export must reproduce their exact historical bank hash.
- Reserve4GiB personally available persistent storage. The budget is4GiB even
  if you declare20GiB available; this avoids the earlier rounded-free-space trap.
- Commit/push this additive implementation before queuing; do not include
  unrelated dirty source files. Existing runtime is pinned at
  `830a2afb2b3350c14ae147d915b8d3072701269c`.

## Queue after committing and pushing

Run on SPORC, not locally. This creates a new detached source worktree and new
campaign root; it neither changes existing worktrees nor cancels other jobs.

```bash
bash <<'BASH'
set -euo pipefail
trap 'echo "Stopped; preserve outputs and journals. Paste the error." >&2' ERR
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_metaembed_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_meta_embedding.sh"
BASH
```

The helper authenticates the original selected checkpoints, writes an immutable
spec, prints a full dry plan, then asks for the printed hash and explicit YES.
It submits three exact jobs: `lukame_setup`, `lukame_fixed`, `lukame_meta`.
Both fits depend on setup with `afterok`; neither depends on the other.
There is **no standalone preflight job**. Setup performs useful shared export
and calibration, plus mandatory inline installed-Weaver/GPU/runtime checks.
No fit runs if those checks fail. Do not reduce requested resources/horizon or
edit immutable specs to force admission; preserve diagnostics instead.

The detached submitter survives terminal logout, but verify that its log ends
with a completed ledger containing all three job IDs before considering the
queue handoff complete. Then Slurm owns the whole DAG; no login-node followup
controller is needed. Never blindly rerun a failed submission or remove claims.

## Results / progress

Using the same environment/exports as the queue helper:

```bash
python -u -B -s "${PROJECT}/scripts/luka_meta_embedding.py" results \
    --spec "${ROOT}/campaign_spec.json"
squeue --me -o "%.18i %.12P %.40j %.2t %.10M %R"
```

Use the exact root printed by the helper (not an inferred newest directory).
Fit logs contain `META_EMBED_PROGRESS` (last emitted batch and percentage),
epoch validation AUC, CE/embedding losses and count of adapter updates. Reports
include every meta query/preservation loss and gradient norm. Results authenticate
receipts and saved selected probabilities, list pending arms, show fixed recovery
anchors and the primary delta versus continued fixed-target control.

Selected native checkpoint: `<arm>/fit/selected.pt`.
Audit/future explicit recovery state: `<arm>/fit/training_state.pt` (full student,
adapter, both optimizer states, epoch and RNG). No automatic resume/deletion.

## Code map / porting

- `luka_meta_embedding/contracts.py`: exact schedule and coefficients.
- `model.py`: native HLT-only Student, separate predictor, bounded target Adapter.
- `meta.py`: functional AdamW, five-step differentiable lookahead, state/RNG
  isolation, query CE and frozen-teacher preservation terms. No first-order
  approximation or validation labels.
- `training.py`: identity-based95/5 TRAIN split, shared head calibration, ordinary
  detached-target steps, periodic meta updates, frozen final5epochs, selection.
- `source.py`: original-source checkpoint authentication and additive Git pin.
- `worker.py`: compact original-bank reproduction, shared initial state,
  maximum-support/whole-population inline resource checks, native readback.
- `campaign.py`, `submission.py`, CLI and shell helpers: immutable scope and
  exact3-job submission with explicit authorization.

To port, preserve paired TRAIN identities, original normalization, frozen teacher
classifier geometry, query-label separation, native-only inference, matching
control and provenance. Do not transplant old numerical hyperparameters and
call them measured resource bounds on another architecture/GPU.
