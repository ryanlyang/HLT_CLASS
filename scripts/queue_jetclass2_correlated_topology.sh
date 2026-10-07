#!/usr/bin/env bash
# Existing clean, pinned worktree; no implicit fetch, overwrite, cancellation.
set -euo pipefail
if [[ $# -lt 5 ]]; then
    echo 'Usage: helper COMMIT ROOT ORIGINAL_RELEASE NOISE_V3_SPEC AVAILABLE_QUOTA_GIB [gate|science] [--execute PLAN_HASH]' >&2
    exit 2
fi
COMMIT=$1 ROOT=$2 ORIGINAL=$3 NOISE=$4 QUOTA=$5 MODE=${6:-gate}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$MODE" == gate || "$MODE" == science ]]
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_DIR JC2_SITE=sporc_a100_debug
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=all)"
git -C "$PROJECT_DIR" merge-base --is-ancestor "$COMMIT" origin/main
CLI="${PROJECT_DIR}/scripts/jetclass2_correlated_topology.py"
if [[ "$MODE" == gate ]]; then
    SPEC="${ROOT}/gate/gate_spec.json"
    if [[ ! -e "$SPEC" ]]; then
        python -u -s "$CLI" create-gate --original-release "$ORIGINAL" --noise-spec "$NOISE" \
            --gate-root "${ROOT}/gate" --project-dir "$PROJECT_DIR" --source-commit "$COMMIT" \
            --available-quota-gib "$QUOTA" --persistent
    fi
else
    SPEC="${ROOT}/science/campaign_spec.json"
    if [[ ! -e "$SPEC" ]]; then
        python -u -s "$CLI" create-campaign --gate-root "${ROOT}/gate" --campaign-root "${ROOT}/science"
    fi
fi
# Fail on reuse of a root prepared by another source or population request.
python -s - "$SPEC" "$COMMIT" "$PROJECT_DIR" "$ORIGINAL" "$NOISE" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
g = s if 'request' in s else json.loads((Path(s['gate_root'])/'gate_spec.json').read_text())
assert s['source_commit'] == sys.argv[2] and Path(s['project_dir']).resolve() == Path(sys.argv[3]).resolve()
assert Path(g['request']['original_release']['path']).resolve() == Path(sys.argv[4]).resolve()
assert Path(g['request']['noise_spec']['path']).resolve() == Path(sys.argv[5]).resolve()
PY
if [[ ${7:-} == --execute ]]; then
    [[ $# -eq 8 && "$8" =~ ^[0-9a-f]{64}$ ]]
    python -u -s "$CLI" submit --spec "$SPEC" --mode "$MODE" --execute --plan-hash "$8"
else
    [[ $# -le 6 ]]
    python -u -s "$CLI" submit --spec "$SPEC" --mode "$MODE"
    python -u -s "$CLI" plan --spec "$SPEC" --mode "$MODE"
    echo 'Dry review only. Repeat with MODE --execute and the printed plan content_hash.'
fi
