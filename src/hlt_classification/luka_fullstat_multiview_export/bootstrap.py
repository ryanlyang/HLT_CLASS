"""Import scientific dependencies from the immutable, original worktree."""
import json
import importlib.util
from pathlib import Path
import subprocess
import sys

PARENT_COMMIT = 'a1eb1a746875a2697216d45a7c8ea4b6acdabc8a'


def configure(project, path, *, registered=False):
    project = Path(project).resolve()
    record = json.loads(Path(path).read_text())
    if registered:
        record = json.loads(Path(record['reuse']['parent_spec']['path']).read_text())
    original = Path(record['project_dir']).resolve(strict=True)
    def git(*args):
        return subprocess.run(['git', '-C', str(original), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    if (record['source_commit'] != PARENT_COMMIT or git('rev-parse', 'HEAD') != PARENT_COMMIT
            or git('status', '--porcelain', '--untracked-files=all')):
        raise ValueError('Original a1eb1a74 source must be clean and unchanged')
    if 'hlt_classification' in sys.modules:
        raise RuntimeError('Bootstrap must precede package imports')
    sys.path.insert(0, str(original / 'src'))
    import hlt_classification
    if Path(hlt_classification.__file__).resolve() != original/'src/hlt_classification/__init__.py':
        raise ValueError('Wrong historical scientific package')
    # Register only this namespace. Do not expose other newer packages as a
    # fallback: every existing scientific import must resolve in the old tree.
    name = 'hlt_classification.luka_fullstat_multiview_export'
    folder = project/'src/hlt_classification/luka_fullstat_multiview_export'
    adapter_spec = importlib.util.spec_from_file_location(name,folder/'__init__.py',
        submodule_search_locations=[str(folder)])
    adapter = importlib.util.module_from_spec(adapter_spec)
    sys.modules[name] = adapter; adapter_spec.loader.exec_module(adapter)
    hlt_classification.luka_fullstat_multiview_export = adapter
    return original
