#!/usr/bin/env python3
"""Sparse FullStat recovery and explicitly authorized science follow-up."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullstat import recovery as r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('review', 'run'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--original-gate')
    parser.add_argument('--review-hash')
    parser.add_argument('--authorize')
    a = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    path = Path(a.root)/'gate_spec.json'
    if a.operation == 'review' and not path.exists():
        if not a.original_gate:
            parser.error('--original-gate is required for a fresh recovery')
        r.create(a.original_gate, a.root, project=project, commit=a.commit)
    gate = load_json(path)
    if gate['source_commit'] != a.commit or Path(gate['project_dir']).resolve() != project:
        raise ValueError('Recovery source pin differs')
    if a.original_gate and Path(a.original_gate).resolve() != Path(gate['original_gate']['path']):
        raise ValueError('Original gate locator differs')
    if a.operation == 'review':
        value = r.review(gate)
        print(json.dumps(value, indent=2, sort_keys=True), flush=True)
        print('DRY REVIEW: no jobs submitted. Three missing files, assembly, GPU preflight; then 16 science jobs ONLY if the full gate passes.')
        print('REVIEW_HASH='+value['content_hash'], flush=True)
    else:
        r.run(gate, review_hash=a.review_hash, authorization=a.authorize)


if __name__ == '__main__':
    main()
