"""Load the standalone ordering helper, not scientific code from a new worktree."""
import importlib.util
from pathlib import Path

if __name__ == '__main__':
    project = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('fullstat_embedding_order',
        project/'src/hlt_classification/operations/fullstat_embedding_order.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(project)
