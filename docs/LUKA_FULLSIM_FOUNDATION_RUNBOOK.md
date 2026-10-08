# Luka FullSim stage 1: build and inspect the frozen population

This is **only** source authentication, scalar selection and split publication.
It neither submits nor trains the ladder. [Plan](plans/LUKA_FULLSIM_FOUNDATION_PLAN.md)
and [contract](contracts/LUKA_FULLSIM_FOUNDATION.md) define the new semantics.

## Input and outputs

Input container (read-only):

```text
/home/ryreu/atlas/datasets/cms_fullsim_630k_trial_luka_v1/
  source.txt
  source.sha256
  raw/jetclass2/train_higgs2p/fullsim_offline+hlt/*.root
  raw/jetclass2/train_qcd/fullsim_offline+hlt/*.root
```

Output must be a **fresh, separate directory**. The build publishes
`inventory.json`, `split_manifest.json`, then `foundation.json` last. Only the
last file marks completion, and it must validate along with its parents.
Existing/partial outputs are preserved; use a separately reviewed new attempt
directory after diagnosing a failure. Do not delete source files or edit hashes.

Before building, commit/push the new `luka_fullsim` package, CLI, worker, tests,
plan/contract/runbook and donor entry. Review the existing unrelated HANDOFF
changes separately; never `git add .`. Use a clean detached SPORC checkout of
that exact pushed commit after `git fetch origin main`. The build checks all
tracked bytes, clean state and ancestry in the fetched `origin/main`;
keep generated outputs outside the checkout. It does not commit or fetch itself.

## Execute stage 1 on SPORC

The thin worker is `sbatch/run_luka_fullsim_foundation.sh`. It requests a single
CPU, 8 GiB, one hour, debug/`reu-aisocial`/`qos_tier3`, no GPU or requeue. These
are **initial metadata-build requests**, not measured training requirements.
Review resource feasibility before submission. The worker takes four arguments:

```text
absolute clean project path
absolute input container path
absolute fresh output root
full 40-character pushed commit
```

Example after setting `PROJECT_DIR`, `CONTAINER`, `OUTPUT`, and `COMMIT` to
those reviewed values (ensure the output's parent exists). First test resources:

```bash
sbatch --test-only --output="${OUTPUT}.slurm-%j.out" \
  "${PROJECT_DIR}/sbatch/run_luka_fullsim_foundation.sh" \
  "${PROJECT_DIR}" "${CONTAINER}" "${OUTPUT}" "${COMMIT}"
```

Then submit **once**, saving the printed ID. Do not rerun because the queue is
briefly empty; use that exact ID with `sacct` if it has already finished:

```bash
sbatch --parsable --output="${OUTPUT}.slurm-%j.out" \
  "${PROJECT_DIR}/sbatch/run_luka_fullsim_foundation.sh" \
  "${PROJECT_DIR}" "${CONTAINER}" "${OUTPUT}" "${COMMIT}"
```

This launches one metadata job, **not the full campaign**. No auto-followup is
implemented or armed. The exclusive output-directory claim prevents a second
job from overwriting a first attempt, but is not an exactly-once submission
journal. Use a normal clean submission environment without inherited SBATCH
resource overrides. A failed job cannot authorize stage 2.

## Inspection after completion

In the SPORC environment (`PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`,
`PYTHONPATH=${PROJECT_DIR}/src`, `${CONDA_PREFIX}/lib` prepended to
`LD_LIBRARY_PATH`), these commands submit nothing:

```bash
python -s "${PROJECT_DIR}/scripts/luka_fullsim_foundation.py" summary --root "${OUTPUT}"
python -s "${PROJECT_DIR}/scripts/luka_fullsim_foundation.py" verify --root "${OUTPUT}"
```

Summary authenticates saved artifacts and replays the deterministic split.
Verify additionally rehashes every ROOT file and rescans only scalar metadata.
Use `verify --container /new/transfer/container` after relocation. It still
requires the original transfer marker, manifest and identical payload bytes.
Neither command decodes test particles or performs test inference. Full-source
hashing and pre-split scalar labels/pT metadata are explicitly declared, not
misreported as zero test-byte access.

## Consumption boundary

```python
from hlt_classification.luka_fullsim.foundation import load_foundation
from hlt_classification.luka_fullsim.splits import ordinary_rows

foundation, inventory, splits = load_foundation(output_root)
for row in ordinary_rows(inventory, splits, role="train"):
    # Both offline and HLT refer to this SAME file, latest tree and entry.
    # No assumption of one-to-one constituent correspondence is implied.
    print(row["identity"], row["file"], row["tree"], row["entry"], row["label"])
```

This is join metadata, not an approved particle loader. `final_test` is rejected
by the ordinary accessor. Later workers must authenticate the foundation hash,
source file checksums and their selected role before reading arrays. Source
keys/labels/split IDs must never become deployable feature channels.

Exact 100k/50k class quotas approximate the **global post-cut** proportions to
within one row per class by largest remainders. Whole-file reservoirs target
40/20/40 only to provide sampling headroom. Unused rows in train/validation
files stay unused in that role; they are not test. Test size is measured from
its disjoint files, not `eligible - 150000`. File independence does not prove
generator-event independence across files.

The prior scalar audit suggests 374129 eligible jets under the agreed offline
pT cut, including 276333 QCD and 97796 signal. These are user-supplied estimates
to compare against the real build, not prefilled success artifacts. The new
build must establish counts and capacity itself.

Stage 2 still needs tracking-unit/sentinel and producer-label conventions,
selected particle validity, pairing/view construction and genuine installed-
Weaver/SPORC GPU acceptance. Neither old CONTEXT acceptance nor this metadata
foundation admits training on Luka's new FullSim sample.
