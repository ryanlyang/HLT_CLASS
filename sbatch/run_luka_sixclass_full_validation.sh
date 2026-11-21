#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 3 && "$1" == /* && "$2" == /* && "$3" == /* ]]
AUDIT_PROJECT="$1"; SPEC="$2"; OUT="$3"
export PROJECT_DIR
PROJECT_DIR="$(/usr/bin/python3 -I -B - "$SPEC" <<'PY'
import json, sys
from pathlib import Path
s=json.loads(Path(sys.argv[1]).read_text())
assert s['contract']=='LUKA_SIXCLASS_LOW_LR_OSCAR_SPEC/v1'
p=Path(s['project']).resolve(strict=True)
assert p.is_relative_to(Path('/oscar/home/rlyang/hlt_classification/source'))
print(p)
PY
)"
export ENV_DIR=/oscar/home/rlyang/hlt_classification/environments/new_balanced_oscar_v1
source "$PROJECT_DIR/sbatch/luka_sixclass_low_lr_oscar_common.sh"
exec "$ENV_DIR/bin/python" -X faulthandler -u -B -s \
    "$AUDIT_PROJECT/scripts/compare_luka_sixclass_full_validation.py" --spec "$SPEC" --output "$OUT"
