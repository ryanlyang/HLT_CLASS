#!/usr/bin/env python3
"""TRAIN-only local ROOT/input/installed-Weaver witness, NOT Oscar admission."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from hlt_classification.data.cache_contracts import write_immutable_json
from hlt_classification.cms_proxy_ladder import views
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.cache import RamCache
from hlt_classification.literature_context.transform import build_inputs
from hlt_classification.luka_fullsim.particles import physical
from hlt_classification.luka_fullsim.preflight_cache import _block
from hlt_classification.luka_fullstat import cache, data
from hlt_classification.luka_fullstat.contracts import COORDINATES, artifact, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('container','foundation','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    foundation, memberships = data.load(args.foundation)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = {co: [] for co in COORDINATES}
    config = cache.conventions(foundation['content_hash'])
    for record, member in zip(foundation['files'], memberships):
        path = data.authenticate_source(args.container, record)
        indices = data.ordinary_indices(member, 'train')[:8]
        # Single-entry requests are small witnesses, never test/validation reads.
        for entry in indices:
            role, actual, raw_hlt, raw_off = next(cache.read_selected_chunk(path, record, member, int(entry), int(entry)+1))
            if role != 'train' or actual != entry:
                raise AssertionError('Witness escaped TRAIN')
            hlt, off = physical(raw_hlt, config, 'hlt'), physical(raw_off, config, 'offline')
            row_id = identity(record, int(entry))
            mapping = views.match_particles(hlt, off)
            light = SimpleNamespace(identity=row_id, label=int(member['labels'][entry]))
            for co in COORDINATES:
                view = views.build_view(identity=row_id, proxy=hlt, offline=off, coordinate=co, mapping=mapping)
                rows[co].append((light, build_inputs(view, capacity=512)))
        print('Verified TRAIN witnesses:', record['path'], flush=True)
    bundles = {co: RamCache([_block(0, values)], role='train', coordinate_name=co,
                          foundation_sha256=foundation['content_hash']) for co,values in rows.items()}
    reference = bundles['D000'].identities
    for bundle in bundles.values():
        np.testing.assert_array_equal(reference, bundle.identities)
    parity = installed_parity(bundles['D000'], device=args.device)
    result = artifact('LOCAL_WITNESS', parents=dict(foundation=foundation['content_hash']),
        rows=len(reference), coordinate_rows={co:len(b) for co,b in bundles.items()}, parity=parity,
        class_counts=np.bincount(bundles['D000'].labels, minlength=11).tolist(),
        device=args.device, torch_version=str(torch.__version__),
        production_submission_authorized=False, oscar_acceptance=False,
        full_batch_resources_measured=False, final_test_accessed=False, validation_accessed=False)
    write_immutable_json(args.output/'witness.json', result)
    print('PASS: real TRAIN pairing, seven views and installed-Weaver forward/gradient parity. NOT Oscar acceptance.')


if __name__ == '__main__':
    main()
