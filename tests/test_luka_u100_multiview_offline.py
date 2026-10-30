"""Real synthetic ROOT replay of retired OFFLINE inputs; no production data writes."""
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pytest

from test_luka_fullsim_stage2 import dataset, prepared
from test_luka_fullsim_alljets import all_prepared
from hlt_classification.data.cache_contracts import sha256_file
from hlt_classification.luka_u100_multiview import views


def test_offline_execution_policy_is_versioned_and_old_specs_are_rejected():
    from hlt_classification.data.cache_contracts import with_content_hash
    from hlt_classification.luka_u100_multiview import contracts, campaign
    from hlt_classification.luka_u100_multiview.offline_cache import POLICY
    spec = contracts.artifact('SPEC', **campaign.registration())
    contracts.validate(spec, 'SPEC')
    assert spec['contract'] == 'LUKA_U100_MULTIVIEW_SPEC/v2'
    assert spec['offline_input_policy'] == POLICY
    assert spec['recipe']['contract'] == 'LUKA_U100_MULTIVIEW_RECIPE/v1'
    old = with_content_hash(dict(spec, contract='LUKA_U100_MULTIVIEW_SPEC/v1', schema_version=1))
    with pytest.raises(ValueError):
        contracts.validate(old, 'SPEC')


def test_repair_source_guard_keeps_all_other_runtimes_pinned():
    from hlt_classification.luka_u100_multiview import source
    assert source.compatible_changes(['M\tsrc/hlt_classification/luka_u100_multiview/views.py',
                                      'A\tsrc/hlt_classification/luka_u100_multiview/offline_cache.py'])
    for row in ('M\tsrc/hlt_classification/luka_fullsim/regular_cache.py',
                'M\tsrc/hlt_classification/luka_fullstat_retained/model.py',
                'M\tscripts/luka_u100_embedding.py', 'A\tsrc/unreviewed.py',
                'D\tsrc/hlt_classification/luka_u100_multiview/views.py'):
        assert not source.compatible_changes([row])


@pytest.fixture
def replay_fixture(all_prepared, tmp_path):
    gate, manifest, p, inventory, splits = all_prepared
    root = tmp_path/'prepared'
    shutil.copytree(Path(gate['root'])/'prepared', root)
    spec = dict(memory_mb=120000, counts=p['counts'], anchor=dict(evidence=dict(
        prepared_root=str(root), raw_container=gate['input_container'],
        cache=manifest['content_hash'], prepared=p['content_hash'])))
    return spec, (manifest, p, inventory, splits), root


def inventory_bytes(root):
    return {str(p.relative_to(root)): sha256_file(p) for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('role', ['train', 'validation'])
@pytest.mark.parametrize('all_missing', [False, True])
def test_missing_offline_replays_exact_original_without_writes(replay_fixture, role, all_missing, capsys):
    spec, meta, root = replay_fixture
    expected = views.load(spec, role, 'FUSION', meta)
    refs = meta[0]['blocks'][role]['OFFLINE']
    retired = refs if all_missing else refs[:1]
    for ref in retired:
        (root/ref['path']).unlink()
    before = inventory_bytes(root)
    actual = views.load(spec, role, 'FUSION', meta)
    assert actual.context.foundation_sha256 == expected.context.foundation_sha256
    for indices in (np.arange(len(actual)), np.arange(len(actual))[::-1]):
        for key, value in expected.batch(indices).items():
            np.testing.assert_array_equal(actual.batch(indices)[key], value)
    assert inventory_bytes(root) == before
    output = capsys.readouterr().out
    assert 'OFFLINE_REPLAY' in output and 'historical_bytes_verified=True' in output


def test_present_offline_corruption_is_not_rebuilt(replay_fixture, monkeypatch):
    spec, meta, root = replay_fixture
    ref = meta[0]['blocks']['train']['OFFLINE'][0]
    (root/ref['path']).write_bytes(b'corrupt')
    monkeypatch.setattr(views.regular_cache, 'selected_file_rows',
                        lambda *a, **k: pytest.fail('Corruption must not trigger raw fallback'))
    with pytest.raises(ValueError, match='bytes differ'):
        views.load(spec, 'train', 'OFFLINE', meta)


@pytest.mark.parametrize('changed', ['checksum', 'summary', 'conventions', 'raw_source'])
def test_offline_rebuild_rejects_changed_evidence(replay_fixture, changed, tmp_path):
    spec, meta, root = replay_fixture
    meta = deepcopy(meta)
    ref = meta[0]['blocks']['train']['OFFLINE'][0]
    (root/ref['path']).unlink()
    if changed == 'checksum':
        ref['sha256'] = '0'*64
    elif changed == 'summary':
        meta[1]['summaries']['train']['OFFLINE']['input_sha256'] = '0'*64
    elif changed == 'conventions':
        from test_luka_fullsim_stage2 import config
        meta[1]['conventions'] = config(unit='cm')
    else:
        copied = tmp_path/'raw_copy'
        shutil.copytree(spec['anchor']['evidence']['raw_container'], copied)
        spec['anchor']['evidence']['raw_container'] = str(copied)
        record = meta[2]['files'][ref['file_index']]
        (copied/'raw'/record['path']).write_bytes(b'corrupt source')
    before = inventory_bytes(root)
    with pytest.raises(ValueError, match='checksum|fingerprint|bytes|replay'):
        views.load(spec, 'train', 'OFFLINE', meta)
    assert inventory_bytes(root) == before


def test_offline_rebuild_budget_and_sealed_roles_precede_raw_access(replay_fixture, monkeypatch):
    spec, meta, root = replay_fixture
    (root/meta[0]['blocks']['train']['OFFLINE'][0]['path']).unlink()
    monkeypatch.setattr(views.regular_cache, 'selected_file_rows',
                        lambda *a, **k: pytest.fail('No raw access allowed'))
    for role in ('test', 'final_test', 'unused'):
        with pytest.raises(PermissionError):
            views.load(spec, role, 'OFFLINE', meta)
    spec['memory_mb'] = 1
    with pytest.raises(MemoryError):
        views.load(spec, 'train', 'OFFLINE', meta)


@pytest.mark.parametrize('coordinate', ['D000', 'U100'])
def test_missing_student_or_u100_cache_does_not_use_offline_replay(replay_fixture, coordinate, monkeypatch):
    spec, meta, root = replay_fixture
    (root/meta[0]['blocks']['train'][coordinate][0]['path']).unlink()
    monkeypatch.setattr(views.regular_cache, 'selected_file_rows',
                        lambda *a, **k: pytest.fail('Only OFFLINE has a replay policy'))
    with pytest.raises(FileNotFoundError):
        views.load(spec, 'train', coordinate, meta)
