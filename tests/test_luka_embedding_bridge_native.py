"""Byte-exact relocation and missing-cache replay using synthetic ROOT data."""
import copy
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from test_luka_u100_pair_correction import dataset, prepared, all_prepared
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, load_npz_arrays, with_content_hash, write_immutable_json,
)
from hlt_classification.luka_embedding_bridge_hydra import native, contracts, campaign


def snapshot():
    identity = dict(git_commit='a'*40, git_tree='b'*40, tracked_files_sha256='c'*64)
    return with_content_hash(dict(contract='hlt_classification_source_snapshot_v1', schema_version=1,
        **identity, tracked_file_count=1, worktree_clean=True, source_snapshot_sha256=canonical_sha256(identity)))


@pytest.fixture
def source(all_prepared, tmp_path, monkeypatch):
    gate, cache, prep, inv, splits = all_prepared
    root = tmp_path/'source'; shutil.copytree(Path(gate['root'])/'prepared', root)
    meta = (cache, prep, inv, splits)
    proof = dict(counts=prep['counts'], prepared=prep['content_hash'], cache=cache['content_hash'], campaign='b'*64)
    origin = contracts.artifact('TEST_ORIGIN', root=str(tmp_path/'old_run'), anchor=dict(evidence=dict(
        prepared_root=str(root), raw_container=gate['input_container'])))
    path = tmp_path/'origin.json'; write_immutable_json(path, origin)
    monkeypatch.setattr(native.original_transfer, 'authenticate', lambda *a: proof)
    monkeypatch.setattr(native.regular_cache, 'metadata', lambda _: meta)
    monkeypatch.setattr(native, 'COUNTS', prep['counts'])
    monkeypatch.setattr(native.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100*2**30))
    bundle = SimpleNamespace(manifest=dict(origin=proof), json=lambda name: cache if 'cache_manifest' in name else prep)
    return root, path, meta, bundle


@pytest.mark.parametrize('missing', [False, True])
def test_exact_export_with_existing_or_deleted_caches(source, tmp_path, missing):
    root, origin, meta, bundle = source
    expected = native.references(meta[0])
    if missing:
        for ref in expected:
            (root/ref['path']).unlink()
    result = native.export(root, origin, tmp_path/'delta', source_snapshot=snapshot())
    delta = native.Delta(result['directory'], result['manifest_hash'], bundle); delta.verify_all()
    assert result['payload_bytes'] == sum(r['bytes'] for r in expected)
    assert delta.manifest['files'] == expected
    assert set(delta.manifest['coordinates']) == {'train', 'validation'}
    if missing:
        assert all(not (root/r['path']).exists() for r in expected)  # source caches stay absent
    for ref in expected:
        assert file_ref(delta.path(ref['path']), root=delta.root) == ref
    with pytest.raises(PermissionError):
        delta.path('inputs/test/D033/0000.npz')
    broken = delta.root/expected[0]['path']
    content = bytearray(broken.read_bytes()); content[-1] ^= 1; broken.write_bytes(content)
    with pytest.raises(ValueError):
        delta.verify_all()


def test_corrupt_existing_cache_never_rebuilt(source, tmp_path, monkeypatch):
    root, origin, meta, _ = source
    ref = native.references(meta[0])[0]; path = root/ref['path']
    path.write_bytes(b'broken cache')
    monkeypatch.setattr(native.regular_cache, 'selected_file_rows', lambda *a: pytest.fail('Corruption must not trigger replay'))
    with pytest.raises(ValueError):
        native.export(root, origin, tmp_path/'delta', source_snapshot=snapshot())
    assert not (tmp_path/'delta/manifest.json').exists()


def test_replay_roundoff_does_not_publish_success(source, tmp_path, monkeypatch):
    root, origin, meta, _ = source
    ref = native.references(meta[0])[0]; (root/ref['path']).unlink()
    original = native._values
    def changed(*args, **kwargs):
        values = original(*args, **kwargs)
        for value in values.values():
            value.features[0,15] = np.nextafter(value.features[0,15], np.float32(np.inf))
        return values
    monkeypatch.setattr(native, '_values', changed)
    with pytest.raises(ValueError, match='historical bytes'):
        native.export(root, origin, tmp_path/'delta', source_snapshot=snapshot())
    assert not (tmp_path/'delta/manifest.json').exists()


def test_native_sources_join_and_route_exact_files(source, tmp_path, monkeypatch):
    root, origin, meta, bundle = source
    result = native.export(root, origin, tmp_path/'delta', source_snapshot=snapshot())
    bundle.path = lambda name: root/name.removeprefix('prepared/')
    monkeypatch.setattr(native, 'Bundle', lambda _: bundle)
    monkeypatch.setattr(native, 'OfflineDelta', lambda *a: SimpleNamespace(root=root, path=lambda name:root/name))
    parent = tmp_path/'parent.json'; write_immutable_json(parent, dict(offline_root=str(root), offline_hash='f'*64))
    sources = native.Sources(dict(bundle_root=str(root), parent_spec=file_ref(parent), native_root=result['directory'], native_hash=result['manifest_hash']))
    sources.verify_all()
    for role in ('train','validation'):
        for coord in native.HISTORICAL:
            ref = meta[0]['blocks'][role][coord][0]; a = load_npz_arrays(root/ref['path'])
            physical = dict(identities=a['identities'], labels=a['labels'], HLT_offsets=a['offsets'], OFFLINE_offsets=a['offsets'])
            block = sources.block(role, coord, ref['file_index'], physical)
            assert block.features.tobytes() == a['features'].tobytes()
            assert block.vectors.tobytes() == a['vectors'].tobytes()
            wrong = dict(physical, identities=np.roll(a['identities'], 1, axis=0))
            with pytest.raises(ValueError, match='join'):
                sources.block(role, coord, ref['file_index'], wrong)
            wrong = dict(physical, HLT_offsets=a['offsets']+1, OFFLINE_offsets=a['offsets']+1)
            with pytest.raises(ValueError, match='support'):
                sources.block(role, coord, ref['file_index'], wrong)


def test_delta_rejects_resealed_inventory_or_role_change(source, tmp_path):
    root, origin, _, bundle = source
    result = native.export(root, origin, tmp_path/'delta', source_snapshot=snapshot())
    manifest = Path(result['directory'])/'manifest.json'; value = load_json(manifest)
    for field, replacement in [('validation_usage','optimization'), ('coordinates',{'test':['U100']}), ('files',value['files'][:-1])]:
        bad = copy.deepcopy(value); bad[field] = replacement; bad.pop('content_hash')
        bad = with_content_hash(bad); manifest.unlink(); write_immutable_json(manifest, bad)
        with pytest.raises(ValueError):
            native.Delta(result['directory'], bad['content_hash'], bundle)


def test_spec_v2_and_old_physical_recipe_identity():
    from subprocess import run
    root = Path(__file__).resolve().parents[1]
    # The original recipe has the same schema/content; only execution binding changes.
    old = run(['git','-C',str(root),'show','0a00441fee7b7974f6e7b2692e79519a65faabd3:src/hlt_classification/luka_embedding_bridge_hydra/contracts.py'],
        check=True, capture_output=True, text=True).stdout
    namespace = dict(__name__='hlt_classification.luka_embedding_bridge_hydra._old_contracts', __package__='hlt_classification.luka_embedding_bridge_hydra')
    exec(compile(old, '<original-contracts>', 'exec'), namespace)
    assert contracts.recipe() == namespace['recipe']()
    assert contracts.artifact('SPEC')['schema_version'] == 2
    with pytest.raises(ValueError): contracts.validate(namespace['artifact']('SPEC'), 'SPEC')
    assert contracts.artifact('TRANSFER') == namespace['artifact']('TRANSFER')
    assert campaign.registration()['native_input_policy'] == native.POLICY
