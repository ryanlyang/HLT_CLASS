# FullStat class/pT audit on Oscar

This runs a short, single-thread CPU metadata audit, not a training job. It
reads saved scalar memberships and writes a small report. It needs no GPU,
ROOT scan or training preflight. Defaults: the six-teacher34.92% campaign,
TRAIN only, offline `jet_pt`,50/100/200GeV diagnostic widths,3M draws x20
epochs for hypothetical repetition estimates. No binning is selected yet.

## Commit and push locally

Only the new audit files are included below. Existing staged changes and
unrelated work remain outside this commit. The evidence additions to the
already-dirty HANDOFF/LEGACY_SOURCE_MAP can be staged separately by hunk.

```powershell
$auditFiles = @(
  'src/hlt_classification/operations/fullstat_pt_audit.py',
  'src/hlt_classification/operations/fullstat_pt_metadata.py',
  'scripts/audit_luka_fullstat_pt.py',
  'scripts/audit_luka_fullstat_pt.sh',
  'tests/test_luka_fullstat_pt_audit.py',
  'docs/plans/LUKA_FULLSTAT_PT_AUDIT_PLAN.md',
  'docs/contracts/LUKA_FULLSTAT_PT_AUDIT.md',
  'docs/LUKA_FULLSTAT_PT_AUDIT_RUNBOOK.md',
  'docs/LUKA_FULLSTAT_PT_AUDIT_VALIDATION.md'
)
git add -- $auditFiles
git commit --only -m "Add FullStat TRAIN class and pT occupancy audit" -- $auditFiles
git push origin main
git rev-parse HEAD
```

## Paste once on Oscar

Paste the commit printed above when prompted. This fetches code into a new
detached source worktree; it does not change any existing campaign worktree.
The command finishes by printing the report. It does not submit a Slurm job.
Keep the terminal open until `Saved report:` appears.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Pushed 40-character audit commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
BASE=/oscar/home/rlyang/hlt_classification
REPO="${BASE}/source/HLT_Classification_subclass_16546c0e"
PROJECT="${BASE}/source/HLT_Classification_pt_audit_${COMMIT:0:8}"
SPEC="${BASE}/checkpoints/luka_fullstat_six_teacher_memory_2fb09f38_r1/campaign_spec.json"
OUT="${BASE}/reports/fullstat_pt_audit_${COMMIT:0:8}_r1"
test -r "${SPEC}"
[[ ! -e "${OUT}" && ! -L "${OUT}" ]]
git -C "${REPO}" fetch origin main
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ ! -e "${PROJECT}" ]]; then
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
git -C "${PROJECT}" diff --quiet HEAD -- src scripts sbatch
bash "${PROJECT}/scripts/audit_luka_fullstat_pt.sh" run \
    --spec "${SPEC}" --output "${OUT}"
BASH
```

If the existing scientific Python environment is unavailable, this audit only
needs Python3.10+ and NumPy. Set `AUDIT_PYTHON=/absolute/path/to/healthy/python`
before running the wrapper to bypass environment activation. Do not repair or
install packages into an active training environment merely for this audit.

The first audit covers the complete TRAIN pT extent. Empty tail cells are an
expected possible result, not a failed audit. Sparse cells (<100 by default)
are warnings, not an instruction to discard jets. The summary reports how many
bins contain every class and the contiguous spans meeting the threshold;
inspect `cells.csv` for the full class-by-bin detail. We
will use these counts to choose the actual training range/bins later.

For a follow-up **diagnostic only**, an explicit range can be supplied with
`--pt-range LOW HIGH`. Each width must divide the range exactly. Underflow and
overflow are reported separately for every class; nothing is silently clipped.
Use a new output directory for each audit. Failed/partial outputs are preserved.

## Bring the result back

Paste `summary.txt` here first. The CSV identifies exactly which classes and
pT cells are sparse. To copy all three files to your local repository, substitute
the eight-character audit commit and your usual Oscar login hostname:

```powershell
scp -r rlyang@ssh.ccv.brown.edu:/oscar/home/rlyang/hlt_classification/reports/fullstat_pt_audit_COMMIT8_r1 .
```

Then reprint/authenticate the local copy without accessing Oscar:

```powershell
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -I -B scripts/audit_luka_fullstat_pt.py show fullstat_pt_audit_COMMIT8_r1/report.json
```

`show` requires `report.json`, `summary.txt` and `cells.csv` together. It verifies
the saved report/export hashes, not the remote files anew. No row-level data
are exported. The report distinguishes shared scalar metadata reads from
TRAIN-only aggregation; final-test particles and evaluation remain sealed.

## Code and provenance

- [Audit implementation](../src/hlt_classification/operations/fullstat_pt_audit.py)
- [NumPy-only metadata adapters](../src/hlt_classification/operations/fullstat_pt_metadata.py)
- [CLI](../scripts/audit_luka_fullstat_pt.py) and [Oscar wrapper](../scripts/audit_luka_fullstat_pt.sh)
- [Tests](../tests/test_luka_fullstat_pt_audit.py), [plan](plans/LUKA_FULLSTAT_PT_AUDIT_PLAN.md),
  [contract](contracts/LUKA_FULLSTAT_PT_AUDIT.md), [validation](LUKA_FULLSTAT_PT_AUDIT_VALIDATION.md)

Donor/review baseline: `1fc128dca7dc7ef547d242a1e2941e9a74e25ac8`.
Adapted `data/cache_contracts.py` canonical JSON/array hashes and publication,
`cms_proxy_ladder/contracts.py` file references, `luka_fullstat/contracts.py`
population/role/source policy, and `jetclass2_delphes/schema.py` labels. These
small frozen adapters avoid eager ROOT/model imports in those packages.
Original files remain unchanged; parity tests check the duplicated semantics.
The new publication adapter intentionally refuses overwrite, including equal
bytes, because an audit always uses a fresh directory.
