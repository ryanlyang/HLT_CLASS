# Group B on Brown Hydra: questions and particle-pattern frequencies

This adds9 independent Q/V fits alongside Group A. It uses the same
`luka_next_supervision_hydra` trainer, worker, submission journals, native
export and read-only results/progress interfaces. Read the
[scientific plan](plans/LUKA_SEVEN_CANDIDATE_IMPLEMENTATION_PLAN.md),
[Group B contract](contracts/LUKA_NEXT_SUPERVISION_GROUP_B.md) and
[validation boundary](LUKA_NEXT_SUPERVISION_GROUP_B_VALIDATION.md).

The queue order is `Q1 V1 Q2 V2 Q5 V6 Q3 V7 Q6`: preferred OFFLINE questions
and K128 count+pT targets first, then replications and HLT-source/weight/
marginal/count-only controls. Every row retains native CE+U100-w5, full30 epochs
and unchanged HLT-only deployment. C0/C1 are reused, never queued again.
Group A jobs, source worktrees, claims and output roots are not modified.

## Existing inputs and environment

No SPORC export or data copy is needed. The existing Group A spec supplies
authenticated raw TRAIN HLT/OFFLINE physical measurements, original U100
targets, D000 views, the same-seed reference proofs, original A6000 qualification
and pinned Python environment. The old Group A spec is validated by its own
source worktree in a separate process; a running scientific fit does not need
to finish. No A scientific result supplies targets or admission to B.

Use a new clean pushed worktree and output root. Keep the old reference/source
worktrees available. Reserve4GiB after existing inputs/environment, plus one
GiB filesystem headroom. New target preparation stores compact geometries,
codebooks and frequency banks; it never stores a dense jets-times-questions
answer bank or a dataset-wide pair table. The existing environment already
includes the required sklearn dependency; do not install/update packages.

## Commit locally

Stage the shared package, both focused test files, the two existing CLI/queue
scripts and the Group B/planning/index/handoff documents. Review the staged
diff and `git diff --cached --check`, commit and push. Do not use `git add -A`
in this workspace because it contains unrelated user work and scratch files.
The source guard requires the complete new package, tests and contracts to
be tracked in the exact pushed commit. No change to an old runtime donor is
needed; source lineage is recorded in [LEGACY_SOURCE_MAP.md](LEGACY_SOURCE_MAP.md).

## Brown: register B and submit its technical setup

The default Group A path below corresponds to the user's342e5d3a execution-v2
run. If the actual running Group A spec has a different path, set GROUP_A_SPEC
to that exact file. Do not substitute a target directory or guess a completion
receipt. Registration validates all original references; its log can be quiet
while those subprocesses run.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
MAIN="${BASE}/code_65952ed3"
GROUP_A_SPEC="${BASE}/runs/next_A_342e5d3a_a6000_v2_r1/campaign_spec.json"
test -f "$GROUP_A_SPEC"
read -r -p 'New pushed 40-character Group B commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="${BASE}/code_next_B_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$GROUP_A_SPEC")"
ROOT="${BASE}/runs/next_B_${COMMIT:0:8}_a6000_r1"
test ! -e "$ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
REG_LOG="$(mktemp "${BASE}/runs/next_B_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$REG_LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s \
    "${PROJECT}/scripts/luka_next_supervision_hydra.py" create-b \
    --root "$ROOT" --group-a-spec "$GROUP_A_SPEC" \
    --expected-commit "$COMMIT" --quota-gib 4 >"$REG_LOG" 2>&1; then
    tail -n 100 -- "$REG_LOG"
    exit 1
fi
SPEC="${ROOT}/campaign_spec.json"
python3 -B -s -c 'import json,sys; s=json.load(open(sys.argv[1])); print("B campaign:",s["root"]); print("New fits:",s["remaining"]); print("Reused:",list(s["reused"]))' "$SPEC"
bash "${PROJECT}/scripts/queue_luka_next_supervision_hydra.sh" "$SPEC" setup </dev/tty
squeue --me -o '%.18i %.30j %.12T %R'
BASH
```

Review the exact setup command and type only YES. `lukanb_setup` requests one
RTX A6000,6CPUs,32000MiB, up to12h. This prepares the targets once and runs
genuine warmed production miniatures; it does not queue the9 scientific fits
automatically. Four CPU-only workers handle full-TRAIN question support and
runtime oracles; GPU work stays in the parent. Preparation prints support
progress and realized question coverage. Dictionaries are frozen on TRAIN.

Setup measures full128-row and partial batches for six distinct paths. It
includes native HLT extremes and the largest source/valid-track Q target tails.
It records target-preparation memory, per-fit load/oracle/IPC and optimizer costs,
per-epoch probes, frozen target hashes and dictionary/coverage diagnostics.
The same48h fit ceiling and85% host/GPU bounds apply. Poor finite target quality
or AUC never fails admission; invalid data/lineage/nonfinite required values
and resource violations do. Preserve diagnostics if setup fails.

The exact realized nonconstant question counts may be below1024/family; the
fixed rule retains all supported entries without a rare-prevalence cutoff or
duplicate padding. A wholly constant family is disclosed and masked, not
fabricated or used to cancel a row. This coverage appears in the science review.

## After exact COMPLETED/0:0 setup: queue all9 independent B fits

Use the same Group B commit. The helper authenticates setup receipts, targets,
resources, source and exact job accounting before showing the science dry plan.
Review the coverage/dictionary diagnostics and commands, then type only YES.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
read -r -p 'Same 40-character Group B commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
PROJECT="${BASE}/code_next_B_${COMMIT:0:8}"
SPEC="${BASE}/runs/next_B_${COMMIT:0:8}_a6000_r1/campaign_spec.json"
bash "${PROJECT}/scripts/queue_luka_next_supervision_hydra.sh" "$SPEC" science </dev/tty
squeue --me -o '%.18i %.30j %.12T %.12l %R'
BASH
```

Each new fit requests one A6000/6CPUs/32000MiB with its measured walltime.
No fit dependencies, holds, nice adjustments, lanes or concurrency caps are
added. All9 submit in one pass and can run alongside Group A as Slurm permits.
The B authorization token cannot admit an A plan. There is no automatic retry
or dependent report job; complete submission is durable after logout. A partial
journal/claim without a complete ledger must be inspected before recovery.

## Read completed results or running epochs

Set the exact B SPEC, then run the corresponding saved-results command:

```bash
SPEC="/absolute/path/to/next_B_COMMIT_a6000_r1/campaign_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_next_supervision_hydra.py" results --spec "$SPEC"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_next_supervision_hydra.py" progress --spec "$SPEC"
```

These commands do no inference/training, writes or job changes. They show B
rows and reused C0/C1 only, selected AUC/historical recovery, same-seed control
deltas, published epochs and last logged markers. Pending/failed/incomplete
rows remain visible. Original50k overlaps validation and is report-only; no
independent test is introduced. Keep provisional running snapshots distinct
from authenticated completed fits.
