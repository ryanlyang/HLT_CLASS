#!/usr/bin/env bash
# OSCAR only. No original files or jobs are modified; --execute is explicit.
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:?absolute clean pushed recovery checkout}"
: "${NOISE_K2_RECOVERY_COMMIT:?full pushed recovery commit}"
: "${NOISE_K2_SOURCE_SPEC:?original immutable campaign_spec.json}"
: "${RECOVERY_ROOT:?fresh sibling recovery root}"
MODE="${1:---dry-run}"
[[ "$#" -le 1 && ( "${MODE}" == --dry-run || "${MODE}" == --execute ) ]] || {
    echo "Use --dry-run or --execute" >&2; exit 2;
}
[[ "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" == "${NOISE_K2_RECOVERY_COMMIT}" ]]
export JC2_SITE=oscar_l40s
unset LD_PRELOAD
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
SPEC="${RECOVERY_ROOT}/recovery_spec.json"
EXCLUDE_ARGS=()
IFS=',' read -r -a EXCLUDED_NODES <<< "${NOISE_K2_EXCLUDE_NODES:-gpu3001}"
for NODE in "${EXCLUDED_NODES[@]}"; do
    EXCLUDE_ARGS+=(--exclude-node "${NODE}")
done
if [[ "${MODE}" == --execute ]]; then
    test -f "${SPEC}"
    test -f "${RECOVERY_ROOT}/dry_run.json"
elif [[ ! -e "${RECOVERY_ROOT}" ]]; then
    python -s "${PROJECT_DIR}/scripts/noise_k2_recovery.py" create \
        --source-spec "${NOISE_K2_SOURCE_SPEC}" --recovery-root "${RECOVERY_ROOT}" \
        --source-commit "${NOISE_K2_RECOVERY_COMMIT}" "${EXCLUDE_ARGS[@]}"
fi
python -s - "${SPEC}" <<'PY'
import os, sys
from pathlib import Path
from hlt_classification.noise_k2.recovery import load, require
s = load(sys.argv[1], 'SPEC')
require(s['source_commit'] == os.environ['NOISE_K2_RECOVERY_COMMIT'], 'Recovery commit differs')
require(Path(s['project_dir']).resolve() == Path(os.environ['PROJECT_DIR']).resolve(), 'Recovery checkout differs')
require(Path(s['source_campaign']['path']).resolve() == Path(os.environ['NOISE_K2_SOURCE_SPEC']).resolve(), 'Source differs')
require(Path(s['recovery_root']).resolve() == Path(os.environ['RECOVERY_ROOT']).resolve(), 'Recovery root differs')
require(s['excluded_nodes'] == sorted(set(os.environ.get('NOISE_K2_EXCLUDE_NODES', 'gpu3001').split(','))), 'Node exclusions differ')
PY
if [[ "${MODE}" == --execute ]]; then
    python -s "${PROJECT_DIR}/scripts/noise_k2_recovery.py" submit --spec "${SPEC}" --execute \
        --authorization-phrase "AUTHORIZE OSCAR NOISE K2 TARGETED RECOVERY"
else
    python -s "${PROJECT_DIR}/scripts/noise_k2_recovery.py" submit --spec "${SPEC}"
    echo "Review command_plan.json and dry_run.json before --execute. No jobs submitted or cancelled."
fi
