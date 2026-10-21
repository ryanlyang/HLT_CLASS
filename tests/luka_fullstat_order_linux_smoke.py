"""Run the same kernel fallback test without scientific/pytest dependencies.

Usage: python3 -B tests/luka_fullstat_order_linux_smoke.py
Only creates, stops and terminates its own disposable sleeping child.
"""
import ast
import importlib.util
from pathlib import Path
import sys

if sys.platform != 'linux':
    raise SystemExit('Linux only')
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('order_smoke_runtime',
    root/'src/hlt_classification/operations/fullstat_embedding_order.py')
o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o)
if o.platform.machine() != 'x86_64':
    raise SystemExit('Linux x86_64 only')
print('Native available:', hasattr(o.os, 'pidfd_open'), hasattr(o.signal, 'pidfd_send_signal'))
print('Read-only capability probe:', o.probe_pidfd())
test_path = root/'tests/test_luka_fullstat_embedding_order.py'
tree = ast.parse(test_path.read_text())
test = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name == 'test_real_linux_fallback_pidfd_roundtrip')
namespace = dict(o=o)
exec(compile(ast.Module(body=[test], type_ignores=[]), str(test_path), 'exec'), namespace)
namespace[test.name]()
print('PASS: missing-Python-wrapper Linux pidfd fallback; own child stopped, terminated and reaped.')
