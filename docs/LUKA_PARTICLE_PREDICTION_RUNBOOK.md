# Trial particle-feature prediction: SPORC debug

Authority: [plan](plans/LUKA_PARTICLE_PREDICTION_PLAN.md) and
[v1 contracts](contracts/LUKA_PARTICLE_PREDICTION.md). Two fresh HLT fits,
same frozen COARSE_U100, CE+5*particle-feature prediction, fixed30/batch128.
Aligned versus exact unordered Hungarian matching, no logit KD. Compare with
completed whole-jet U100-w5 from `luka_u100_direct_ed6d0af4_tier3_r1`.
This is 299,234/74,895 exploratory trial validation, NOT FullStat or an independent test.

## Queue after local tests and scoped commit/push

Do not use `git add .`: unrelated work exists. Commit only the new package,
CLI, queue script, Slurm wrapper, focused test, plan, contract and this runbook;
include only this task's HANDOFF/source-map hunks. Do not force-push.
Use the resulting exact pushed 40-character commit on SPORC:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_particlepred_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_particle_prediction.sh"
BASH
```

Helper authenticates completed U100-w5/teacher/controls in original source,
creates a fresh immutable spec, renders a full dry plan, and requires its exact
hash plus YES. Three jobs: `lukapp_preflight` 8h; `lukapp_aligned` and `lukapp_set`
24h each, both afterok the preflight. A100/6CPU/90000MiB, debug. The long science
request is an admission ceiling, not a predicted duration. Preflight measures
each method and refuses projections beyond 24h. No controller or second manual
submission is needed; failure blocks both fits. Preserve failures, never blind retry.
Old pending/running/held jobs, Fresh_check and all datasets are untouched.

Reserve 2GiB personally available persistent space. Workers use registered
`/tmp` and verify a node-local disk filesystem (not Ceph/NFS/tmpfs); there is
no home fallback and no dependence on an unset `SLURM_TMPDIR`. Free space must cover the
actual TRAIN token count*128*4 plus 512MiB. Temporary banks are generated once
per job and retired only within their unique owned directory; receipts keep
hashes/moments/identity metadata. Deterministic FP32 export is used in each job;
both fits require regenerated target/index hashes to match the preflight exactly.
Native selected checkpoints, all histories,
predictions and reference snapshots remain. Old models/caches are not deleted.

## Results and progress

Use the NEW pinned PROJECT and printed ROOT:

```bash
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
python -u -B -s "${PROJECT}/scripts/luka_particle_prediction.py" results --spec "${ROOT}/campaign_spec.json"
```

Results support one completed arm and one pending. Validation/original50k use
the same full-validation-selected weights, with AUC recovery and delta versus
whole-jet U100-w5. Every epoch logs AUC/best AUC/best epoch/CE/aux loss and batch
progress. R50 information is retained in the standard metric records; validation
probabilities are retained for later R30 calculations without rerunning inference.
This single-seed adaptive comparison cannot establish independent-test gains.

## Code and evidence

New package: `src/hlt_classification/luka_particle_prediction/`.
Donor 01d45d6057e799b4e262f8c590b51706803ebfd1; native model/training semantics
derive from U100 embedding and attention modules, with no existing runtime edits.
See HANDOFF for local test evidence. Local CUDA synthetic miniature is not a
genuine remote acceptance: the queued preflight supplies the new SPORC evidence.

Local verification: 23 new focused tests passed after the final scratch change;
82 existing U100 embedding/embedding-auxiliary/attention regression tests passed.
Installed Weaver 0.5.3 full-model CPU/CUDA FP32/BF16 parity and a reduced eight-block
CUDA production-worker miniature passed, including both complete 30-epoch fits,
checkpoint readback, reports, failure injection and owned scratch cleanup.
This does not certify real-population memory, timing or the remote environment.
