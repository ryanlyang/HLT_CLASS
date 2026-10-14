"""All-eligible split retirement and production-path CPU tests; not Slurm evidence."""
from copy import deepcopy
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_luka_fullsim_stage2 import dataset, prepared, config
from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, with_content_hash, write_immutable_json,
)
from hlt_classification.luka_fullsim import (
    alljets_data as d, alljets_campaign as c, alljets_reporting as reporting,
    regular_campaign as old, regular_cache as cache, regular_submission as sub, regular_worker as w,
)


def check_fixture(inventory, splits):
    d.validate_splits(splits, inventory)


def test_all_eligible_file_roles_natural_counts_and_subset(dataset):
    container, inv, original = dataset
    new = d.derive_splits(inv, original)
    d.validate_splits(new, inv)
    assert new['selected_counts']['final_test'] == 0 and new['unused_eligible_rows'] == 0
    assert sum(new['selected_counts'].values()) == sum(inv['selected_class_counts'])
    rows = {}
    for role in ('train', 'validation'):
        rows[role] = {(m['file_index'], e) for m in new['memberships'][role] for es in m['entries_by_class'] for e in es}
        assert len(rows[role]) == new['selected_counts'][role]
        jets = list(d.ParticleReader(container, inv, new, role=role, include_offline=True))
        assert {(j.file_index, j.entry) for j in jets} == rows[role]
        assert np.bincount([j.label for j in jets], minlength=11).tolist() == new['class_counts'][role]
    assert not rows['train'] & rows['validation']
    assert {i for i, _ in rows['train']}.isdisjoint(i for i, _ in rows['validation'])
    for i, group in enumerate(original['groups']):
        role = 'validation' if group['role'] == 'validation' else 'train'
        assert new['groups'][i]['role'] == role
        assert {(i, e) for es in inv['files'][i]['entries_by_class'] for e in es} <= rows[role]
    ids, labels = d.row_arrays(inv, new['memberships']['validation'])
    ix = d.comparison_indices(inv, new, ids, labels)
    old_ids, old_labels = d.row_arrays(inv, original['memberships']['validation'])
    np.testing.assert_array_equal(ids[ix], old_ids)
    np.testing.assert_array_equal(labels[ix], old_labels)


def test_new_contract_does_not_unlock_old_readers(dataset):
    from hlt_classification.luka_fullsim.particles import ParticleReader
    from hlt_classification.luka_fullsim.splits import validate_splits
    container, inv, original = dataset
    new = d.derive_splits(inv, original)
    with pytest.raises(ValueError):
        validate_splits(new, inv)
    with pytest.raises(ValueError):
        ParticleReader(container, inv, new, role='train')
    with pytest.raises(ValueError):
        d.ParticleReader(container, inv, original, role='train')
    for role in ('test', 'final_test', 'unused'):
        with pytest.raises(PermissionError):
            d.ParticleReader(container, inv, new, role=role)
    for key, value in [('historical_test_repurposed_for_training', False), ('independent_test_available', True),
                       ('final_test_accessed', True)]:
        with pytest.raises(PermissionError):
            d.validate(with_content_hash(dict(new, **{key:value})), 'SPLITS')
    for key, value in [('unused_eligible_rows', 1), ('comparison_rows', 1), ('algorithm', 'resplit')]:
        with pytest.raises(ValueError):
            d.validate_splits(with_content_hash(dict(new, **{key:value})), inv)
    bad = deepcopy(new)
    bad['memberships']['train'][0]['entries_by_class'][0].append(999999)
    with pytest.raises(ValueError):
        d.validate_splits(with_content_hash(bad), inv)


@pytest.fixture(scope='module')
def all_prepared(dataset, prepared, tmp_path_factory):
    container, inv, original = dataset
    splits = d.derive_splits(inv, original)
    root = tmp_path_factory.mktemp('alljets_gate')
    foundation = root/'foundation'
    foundation.mkdir()
    source = prepared[1]['source_snapshot']
    for name, value in [('inventory.json', inv), ('split_manifest.json', splits)]:
        write_immutable_json(foundation/name, value)
    f = d.artifact('FOUNDATION', source_snapshot=source, input_container=str(container),
        counts=splits['selected_counts'], source_particles_validated=False, pre_split_scalar_metadata_accessed=True,
        parents=dict(inventory=inv['content_hash'], splits=splits['content_hash'], source=source['content_hash']),
        artifacts={name:dict(sha256=sha256_file(foundation/name), bytes=(foundation/name).stat().st_size)
                   for name in ('inventory.json', 'split_manifest.json')})
    write_immutable_json(foundation/'foundation.json', f)
    counts = {r:splits['selected_counts'][r] for r in ('train', 'validation')}
    gate = c.make_artifact('GATE', root=str(root), input_container=str(container),
        source_snapshot=source, **{**c.gate_policy(), 'counts':counts})
    write_immutable_json(root/'gate_spec.json', gate)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(c, 'validate_gate', lambda _: None)
        mp.setattr(c, 'check_population', check_fixture)
        build = cache.build_rows
        mp.setattr(cache, 'build_rows', lambda *a, **kw: build(*a, **{**kw, 'workers':1}))
        cache.prepare(gate)
    manifest, p, _, _ = cache.metadata(root/'prepared')
    return gate, manifest, p, inv, splits


def test_new_prepare_parallel_and_original_row_inputs(dataset, all_prepared, tmp_path):
    container, inv, original = dataset
    gate, manifest, p, _, splits = all_prepared
    root = Path(gate['root'])/'prepared'
    assert manifest['serial_reference_passed'] and p['counts'] == gate['counts']
    parallel = cache.build_rows(container, inv, splits, p['conventions'], tmp_path, workers=2)
    assert parallel['blocks'] == manifest['blocks']
    assert parallel['summaries'] == p['summaries']
    reference = cache.build_rows(container, inv, original, p['conventions'], tmp_path/'old', workers=1)
    for role in ('train', 'validation'):
        for co in cache.s.COORDINATES:
            larger = cache.load(root, role=role, coordinate=co, max_ram_bytes=2**30)
            lookup = {bytes(key):i for i,key in enumerate(larger.identities)}
            for ref, member in zip(reference['blocks'][role][co], original['memberships'][role]):
                small = cache.read_block(tmp_path/'old', ref, member, inv)
                for i, key in enumerate(small.identities):
                    j = lookup[bytes(key)]
                    batch = larger.batch(np.asarray([j]))
                    lo, hi = small.offsets[i:i+2]
                    n = hi-lo
                    np.testing.assert_array_equal(batch['features'][0, :, :n].T, small.features[lo:hi])
                    np.testing.assert_array_equal(batch['vectors'][0, :, :n].T, small.vectors[lo:hi])


def test_storage_estimate_scalar_only(dataset, monkeypatch):
    import uproot
    container, inv, original = dataset
    splits = d.derive_splits(inv, original)
    original_arrays = uproot.behaviors.TBranch.HasBranches.arrays
    calls = []
    def guarded(tree, fields, *a, **kw):
        assert fields == ['jet_nparticles', 'hlt_jet_nparticles']
        calls.append(fields)
        return original_arrays(tree, fields, *a, **kw)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, 'arrays', guarded)
    assert d.storage_bound(container, inv, splits) > 5*2**30
    assert len(calls) == sum(len(m) for m in splits['memberships'].values())


def fixture_spec(tmp_path, all_prepared):
    gate, _, p, _, _ = all_prepared
    root = tmp_path/'science'
    root.mkdir()
    return c.make_artifact('SPEC', root=str(root), project_dir=str(tmp_path/'project'), source_commit='a'*40,
        source_snapshot=p['source_snapshot'], gate_path=str(Path(gate['root'])/'gate_spec.json'),
        prepared_sha256=p['content_hash'], preflight_sha256='d'*64,
        admission=dict(prepared_root=str(Path(gate['root'])/'prepared'), preflight=dict(path='fixture')),
        train_minutes=240, reduce_minutes=30, **{**c.registration(), 'counts':p['counts']})


@pytest.mark.parametrize('gate_mode', [True, False])
def test_exact_new_namespace_submission(tmp_path, all_prepared, monkeypatch, gate_mode):
    spec = (c.make_artifact('GATE', root=str(tmp_path), project_dir=str(tmp_path/'project'), **c.gate_policy())
            if gate_mode else fixture_spec(tmp_path, all_prepared))
    monkeypatch.setattr(c, 'validate_gate', lambda _: None)
    monkeypatch.setattr(c, 'validate_spec', lambda _: None)
    monkeypatch.setattr(sub, 'check_site', lambda _: None)
    calls = []
    def scheduler(args, **kw):
        assert not any('${JOB_' in a for a in args)
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, f'{700+len(calls)};sporc\n', '')
    monkeypatch.setattr(sub.subprocess, 'run', scheduler)
    plan = sub.submit(spec)
    assert not calls and plan['jobs'] == (3 if gate_mode else 16)
    for row in plan['commands']:
        assert '--partition=debug' in row['command']
        assert any('run_luka_fullsim_alljets.sh' in a for a in row['command'])
        assert '--job-name=lukaa_'+row['task_id'] in row['command']
    with pytest.raises(PermissionError):
        sub.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=old.AUTHORIZATION)
    ledger = sub.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == plan['jobs']
    assert sub.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION) == ledger
    assert len(calls) == plan['jobs']


def test_all_nine_fits_banks_and_common_validation_reports(all_prepared, tmp_path, monkeypatch):
    import torch
    from test_luka_fullsim_science import TinyModel
    from hlt_classification.jetclass2_delphes import runner
    spec = fixture_spec(tmp_path, all_prepared)
    p = all_prepared[2]
    monkeypatch.setattr(c, 'check_population', check_fixture)
    monkeypatch.setattr(c, 'validate_spec', lambda _: (p, {}))
    monkeypatch.setattr(w, 'execution', lambda *a: 'cpu-fixture')
    monkeypatch.setattr(w, 'memory_snapshot', lambda _: dict(rss_bytes=1, peak_rss_bytes=1, gpu_peak_bytes=0))
    monkeypatch.setattr(w, 'new_model', lambda n: (torch.manual_seed(n['initialization_seed']), TinyModel())[1])
    monkeypatch.setattr(w, 'train_kernel', lambda model, train, val, **kw:
        runner.train_kernel(model, train, val, **{**kw, 'device':'cpu'}))
    monkeypatch.setattr(w, 'predict', lambda model, val, **kw:
        runner.predict(model, val, **{**kw, 'device':'cpu'}))
    monkeypatch.setattr(TinyModel, 'to', lambda self, *a, **k: self)
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for task in c.tasks():
            receipt = w.run(spec, task['task_id'])
            assert receipt['historical_test_repurposed_for_training']
            assert w.run(spec, task['task_id']) == receipt
        rows = w.result_rows(spec)
        assert len(rows) == 9 and all(r['state'] == 'COMPLETE' for r in rows)
        for row in rows:
            assert row['validation']['rows'] == p['counts']['validation']
            assert row['original_validation']['rows'] == all_prepared[4]['comparison_rows']
        assert w.completed(spec, 'complete')['result']['fits'] == 9
        w.print_results(spec)
        directory = Path(spec['root'])/'train_M0HLT'
        report = load_json(directory/'training_report.json')
        diagnostic = reporting.validate(spec, 'train_M0HLT', report)
        assert not diagnostic['secondary_selection_used']
        arrays = cache.load_npz_arrays(directory/'validation_predictions.npz')
        swapped = {**arrays, 'identities':arrays['identities'][::-1]}
        with pytest.raises(ValueError, match='identities'):
            reporting.describe(spec, 'train_M0HLT', report, swapped)
        with pytest.raises(ValueError, match='task'):
            reporting.describe(spec, 'train_OFFLINE', report, arrays)
        # Saved predictions are sufficient for later rejection thresholds.
        with (directory/'validation_predictions.npz').open('ab') as handle:
            handle.write(b'corruption')
        with pytest.raises(ValueError):
            w.completed(spec, 'train_M0HLT')
    finally:
        torch.set_num_threads(before)


def test_pinned_recipe_does_not_change_old_experiment():
    assert c.nodes() == old.nodes() and c.recipe() == old.recipe()
    assert c.COUNTS == dict(train=299234, validation=74895)
    assert old.COUNTS == dict(train=100000, validation=50000)
    assert c.registration()['validation_predictions_saved']
    assert c.gate_policy()['automatic_followup'] is False
    assert all(n['context_coordinate'] is None for n in c.nodes())
    root = Path(__file__).resolve().parents[1]
    for name in c.REQUIRED:
        assert (root/name).is_file(), name
    worker = (root/'sbatch'/c.WORKER).read_text()
    assert worker.index('CUBLAS_WORKSPACE_CONFIG=:4096:8') < worker.index('exec python')
    assert '${PROJECT_DIR}/sbatch/common.sh' in worker and 'PYTHONNOUSERSITE=1' in worker


def test_old_smaller_gate_cannot_admit_alljets():
    from test_luka_fullsim_regular import evidence_fixture
    from hlt_classification.luka_fullsim import regular_preflight as pre
    report, gate, p, manifest = evidence_fixture()
    gate.update(contract='LUKA_FULLSIM_ALLJETS_GATE/v1',
                preflight_cache_policy=c.gate_policy()['preflight_cache_policy'])
    with pytest.raises(ValueError, match='contract'):
        pre.validate_evidence(report, gate, p, manifest)
    renamed = with_content_hash(dict(report, contract='LUKA_FULLSIM_ALLJETS_PREFLIGHT/v2', **d.FLAGS))
    with pytest.raises(ValueError, match='evidence/policy'):
        pre.validate_evidence(renamed, gate, p, manifest)
