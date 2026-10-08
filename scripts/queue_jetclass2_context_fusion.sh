#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 4 && $# != 6 ]]; then
  echo 'Usage: bash queue_jetclass2_context_fusion.sh COMMIT PARENT_SPEC ROOT gate|science [--execute REVIEWED_HASH]' >&2
  exit 2
fi
COMMIT="$1"; PARENT_SPEC="$2"; ROOT="$3"; MODE="$4"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$MODE" == gate || "$MODE" == science ]]
if [[ $# == 6 ]]; then
  [[ "$5" == --execute && "$6" =~ ^[0-9a-f]{64}$ ]]
fi
export PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=all)"
git -C "$PROJECT_DIR" merge-base --is-ancestor "$COMMIT" origin/main
export JC2_SITE=oscar_l40s
source "$PROJECT_DIR/sbatch/jetclass2_delphes_common.sh"
CLI="$PROJECT_DIR/scripts/jetclass2_context_fusion.py"
SPEC="$ROOT/study_spec.json"
if [[ ! -e "$ROOT" ]]; then
  [[ "$MODE" == gate && $# == 4 ]]
  echo 'Authenticating the completed CONTEXT_V1 parent and creating a separate fusion study; this may take several minutes.'
  python -u -s "$CLI" create --parent-spec "$PARENT_SPEC" --project-dir "$PROJECT_DIR" --source-commit "$COMMIT" --root "$ROOT"
fi
python -s - "$SPEC" "$COMMIT" "$PROJECT_DIR" "$PARENT_SPEC" "$ROOT" <<'PY'
import json, sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text())
if (spec['source_commit'] != sys.argv[2] or Path(spec['project_dir']).resolve() != Path(sys.argv[3]).resolve()
        or Path(spec['parent_import']['spec']['path']).resolve() != Path(sys.argv[4]).resolve()
        or Path(spec['root']).resolve() != Path(sys.argv[5]).resolve()):
    raise SystemExit('STOP: helper arguments differ from saved study')
PY
if [[ $# == 4 ]]; then
  printf 'Validating %s and preparing its exact dry plan; no jobs will be submitted.\n' "$MODE"
  python -u -s "$CLI" plan --spec "$SPEC" --mode "$MODE"
  echo 'Dry review only. Repeat this phase with --execute and its reviewed plan hash.'
else
  printf 'Reauthenticating %s before exact-plan submission; preserve the journal if interrupted.\n' "$MODE"
  python -u -s "$CLI" submit --spec "$SPEC" --mode "$MODE" --plan-hash "$6" \
    --authorization 'AUTHORIZE CONTEXT V1 OSCAR FUSION EXACT PLAN'
fi
