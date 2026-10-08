#!/usr/bin/env bash
# Source-pinned S3 gate or science stage; no implicit follow-up or cancellation.
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; preserve files and paste the error." >&2' ERR
if [[ $# -lt 4 ]]; then
    echo 'Usage: helper COMMIT SWEEP_SPEC ROOT gate|science [--execute PLAN_HASH]' >&2
    exit 2
fi
COMMIT=$1 SWEEP=$2 ROOT=$3 MODE=$4
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$MODE" == gate || "$MODE" == science ]]
if [[ $# -gt 4 ]]; then
    [[ $# -eq 6 && "$5" == --execute && "$6" =~ ^[0-9a-f]{64}$ ]]
fi
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_DIR JC2_SITE=sporc_a100_debug
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=all)"
git -C "$PROJECT_DIR" merge-base --is-ancestor "$COMMIT" origin/main
CLI="${PROJECT_DIR}/scripts/jetclass2_s3_ladder.py"
SPEC="${ROOT}/study_spec.json"
if [[ ! -f "$SPEC" ]]; then
    [[ "$MODE" == gate && $# -eq 4 ]]
    python -u -s "$CLI" create --sweep-spec "$SWEEP" --root "$ROOT" \
        --project-dir "$PROJECT_DIR" --source-commit "$COMMIT"
fi
python -s - "$SPEC" "$COMMIT" "$PROJECT_DIR" "$SWEEP" "$ROOT" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
if (s['source_commit'] != sys.argv[2] or Path(s['project_dir']).resolve() != Path(sys.argv[3]).resolve()
        or Path(s['sweep_spec']['path']).resolve() != Path(sys.argv[4]).resolve()
        or Path(s['root']).resolve() != Path(sys.argv[5]).resolve()):
    raise SystemExit('STOP: existing S3 campaign belongs to different source or screen')
PY
if [[ $# -eq 6 ]]; then
    python -u -s "$CLI" submit --spec "$SPEC" --mode "$MODE" --execute --plan-hash "$6"
else
    python -u -s "$CLI" submit --spec "$SPEC" --mode "$MODE"
    echo 'Dry review only. No jobs submitted. Repeat with --execute and the exact plan content_hash.'
fi
