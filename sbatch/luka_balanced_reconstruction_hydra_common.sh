#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_DIR:?absolute source required}"
: "${ENV_DIR:?persistent environment required}"
[[ "$PROJECT_DIR" == /* && "$ENV_DIR" == /* ]]
test -x "$ENV_DIR/bin/python"
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
unset TORCH_CUDNN_V8_API_DISABLED
export CONDA_PREFIX="$ENV_DIR"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
cd "$PROJECT_DIR"
printf 'LUKABREC_SHELL checking Python imports (five-minute total limit)\n'
if timeout --signal=TERM --kill-after=15s 300s "$ENV_DIR/bin/python" -X faulthandler -I -u -B - <<'PY'
import faulthandler
import importlib

faulthandler.dump_traceback_later(60, repeat=True)
for name in ('ssl', 'torch.nn.functional', 'uproot', 'scipy', 'threadpoolctl',
             'weaver.nn.model.ParticleTransformer'):
    print('LUKABREC_IMPORT BEGIN ' + name, flush=True)
    importlib.import_module(name)
    print('LUKABREC_IMPORT END ' + name, flush=True)
faulthandler.cancel_dump_traceback_later()
PY
then
    printf 'LUKABREC_SHELL Python import checks passed\n'
else
    LUKABREC_IMPORT_STATUS=$?
    printf 'LUKABREC_SHELL Python import checks failed exit=%s (124 means timeout)\n' "$LUKABREC_IMPORT_STATUS" >&2
    return "$LUKABREC_IMPORT_STATUS"
fi
