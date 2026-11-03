# Minimal U100 correction inputs for Brown CS Hydra

Status: the **transfer exporter/verifier is implemented**. No real SPORC export,
Brown upload or Hydra training validation has been performed by this change.
The existing SPORC worker cannot run unchanged on Hydra. This runbook prepares
the data portion; it does not claim a queue-ready Hydra campaign.

The user measured these required prepared payloads on SPORC:

| Payload | GiB |
| --- | ---: |
| TRAIN D000 |1.174 |
| Validation D000 |0.291 |
| TRAIN U100 |1.174 |
| TRAIN assignments |0.069 |
| U100 embedding targets |0.154 |
| Total, before metadata/reference evidence |2.862 |

All listed files existed with expected sizes. The exporter checks the hashes of
the bytes copied. Its explicit inventory adds compact comparison evidence and
one native control checkpoint for later cross-site parity. It does not copy the
151GB raw dataset, Fresh_check, other research payloads, teacher weights or old
source worktrees. See the [transfer contract](contracts/LUKA_U100_PAIR_TRANSFER.md).

## Code map

- [bundle.py](../src/hlt_classification/luka_u100_pair_transfer/bundle.py):
  original-source authentication, explicit file inventory, streamed export and
  checksum-only archive verification. Standard library only.
- [CLI](../scripts/luka_u100_pair_transfer.py): `inspect`, `export`, `verify`.
- [Tests](../tests/test_luka_u100_pair_transfer.py): scope exclusions, round trip,
  corruption, partial publication, bounds and unsafe archives.

The canonical SPORC source spec remains:

```text
/home/ryreu/atlas/HLT_Classification/checkpoints/luka_u100_pair_correction_c08f6df6_debug_r1/campaign_spec.json
```

Its setup job need not finish to export the inherited inputs. Reference
authentication uses the already completed `u100_w5_s0` control, job21993056.
Keep the original pinned source worktrees available on SPORC for this replay.

## Export on SPORC

Commit/push the new transfer files first. Use a separate clean worktree for the
export helper; do not update the source directory of the queued SPORC campaign.
The commands below do not queue or cancel any jobs.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Pushed transfer-helper commit (40 characters): ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_pairtransfer_${COMMIT:0:8}"
SPEC="${MAIN}/checkpoints/luka_u100_pair_correction_c08f6df6_debug_r1/campaign_spec.json"
OUT="/home/ryreu/atlas/luka_pair_transfer_${COMMIT:0:8}"

git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi

source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

python -u -B -s "${PROJECT}/scripts/luka_u100_pair_transfer.py" export \
    --spec "${SPEC}" --output "${OUT}"
BASH
```

The successful command prints the archive path and SHA256. It creates:

```text
inventory.json             # reviewable explicit source/destination list
campaign-inputs.tar        # only campaign payloads + embedded manifest
SHA256SUMS                 # archive digest
```

For a read-only preview instead, run the same CLI with `inspect --spec "$SPEC"`.
It authenticates references but does not create an output directory or claim
prepared payload checksum verification. A failed export retains the output
directory and partial archive. Diagnose it; do not overwrite it or treat the
partial file as a completed transfer.

Transfer **only `campaign-inputs.tar` and `SHA256SUMS`** using the user's working
authenticated route to a dedicated Brown HLT input directory. Do not rsync the
entire atlas directory. Source code should come from its pinned Git commit, not
from copying every SPORC worktree. The automated SSH attempt could not authenticate
with the user's encrypted Brown key; this tool does not obtain or move that key.
No new transport endpoint or accessible SPORC hostname is assumed here.

## Verify on Brown, before creating the training environment

With the transfer-helper commit checked out and `PROJECT` pointing to that code,
run plain Python (no NumPy/PyTorch/Weaver required):

```bash
python3 -B -s "${PROJECT}/scripts/luka_u100_pair_transfer.py" verify \
    --archive "${INPUT_DIR}/campaign-inputs.tar" \
    --sha256 "${SHA256_FROM_SPORC}"
```

Use the digest printed on SPORC. Successful verification reads only the archive
and prints the exact inventory totals. It does not extract into original paths.
Keep the archive intact until the Hydra consumer is implemented. Old manifests
are preserved verbatim, and must not be edited to make their hashes appear valid
at new locations.

## Remaining Hydra execution work

Add a new portable consumer/site contract instead of loosening the original
SPORC contract. It must resolve the declared relative bundle files, preserve
array identities/labels/slot joins and scientific recipe, use the existing
targets/control, and validate the native control checkpoint on the new runtime.
Install a separate HLT environment and record its exact Weaver/software versions.
Hydra resources and timing must be measured during genuine shared setup before
automatically releasing the same four fresh fits. No new teacher or reference
training is intended. Existing ALBERT jobs/environments and SPORC jobs remain
outside this transfer's scope.
