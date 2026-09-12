"""Run a bounded sample, one compact assignment shard, or a foundation reducer."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.foundation import audit_sample, build_assignment_shard, build_foundation_lock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--foundation-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--task", choices=["sample", "assign", "lock"], required=True)
    p.add_argument("--file-index", type=int)
    p.add_argument("--rows-per-file", type=int, default=8)
    a = p.parse_args()
    if a.foundation_root.resolve().is_relative_to(a.data_root.resolve()):
        p.error("Foundation output cannot be inside raw snapshot")
    spec = load_json(a.foundation_root / "foundation_spec.json")
    if a.task == "sample":
        result = audit_sample(spec, data_root=a.data_root, rows_per_file=a.rows_per_file)
        write_immutable_json(a.foundation_root / "sample_audit.json", result)
    elif a.task == "assign":
        if a.file_index is None:
            p.error("--file-index required for assign")
        result = build_assignment_shard(spec, data_root=a.data_root, output_root=a.foundation_root, file_index=a.file_index)
    else:
        result = build_foundation_lock(spec, a.foundation_root)
    print(result["content_hash"])


if __name__ == "__main__":
    main()
