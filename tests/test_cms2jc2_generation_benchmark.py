"""Separate engineering authorization, lossless replay, bounded throughput study."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zipfile

import awkward as ak
import numpy as np
import pytest
import uproot

from hlt_classification.cms2jc2_response import (
    generation_benchmark_campaign as c, generation_benchmark_data as d,
    generation_benchmark_engine as e, generation_benchmark_worker as w,
    generation_benchmark_queue as q, dev_campaign as dev, dev_submission as submission,
    dev_worker, storage)
from hlt_classification.cms2jc2_response.contracts import artifact, load_json, with_content_hash, sha256_file
from hlt_classification.cms2jc2_response.dev_data import file_ref
from hlt_classification.cms2jc2_response.readers import Pair
from hlt_classification.cms2jc2_response.response import collect, fit_response
from hlt_classification.cms2jc2_response.association import policy
from hlt_classification.cms2jc2_response.assumptions import provisional_compatibility
from hlt_classification.jetclass2_delphes.split_registry import pack_entries
from test_cms2jc2_response_science import particles, pairs
from test_cms2jc2_bdz_tuning import mapping
from test_cms2jc2_bdz_joint import joint_mapping
from test_cms2jc2_bounded import LocalMeasurement
from test_jetclass2_delphes_split_registry import registered
from test_jetclass2_delphes import snapshot


@pytest.fixture(autouse=True)
def native_threads(monkeypatch):
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(key, '1')


@pytest.fixture(scope='module')
def bundle():
    rules = policy()
    loc, lr = collect(pairs('location', 4), rules, cap=7200)
    res, rr = collect(pairs('residual', 4), rules, cap=7200)
    fitted = fit_response(loc, res, location_report=lr, residual_report=rr, candidate_id='B_L',
        review=provisional_compatibility(inventory_hash='a'*64), rules=rules,
        budget='MINIATURE', source_hash='b'*64)
    return artifact('GEN_BUNDLE', response=fitted, historical=mapping(), joint=joint_mapping())


def sample(n=12):
    return [Pair(hashlib.sha256(f'jet{i}'.encode()).hexdigest(), 'a'*64, particles(1+i/100), None) for i in range(n)]


def metadata(monkeypatch, files=None):
    if files is None:
        files = [dict(path=f'S{i%3}/{i}.root', sha256=f'{i+1:064x}', source=f'S{i%3}',
            tree_key='tree;1', entries=5000) for i in range(6)]
    inv = artifact('FAKE_INVENTORY', files=files)
    prof = artifact('FAKE_PROFILE', groups=[dict(path=r['path'], role='train') for r in files],
        memberships={'train': {'files': [dict(path=r['path'], entry_mask=pack_entries(np.arange(r['entries']), r['entries'])) for r in files]}})
    monkeypatch.setattr(d, 'validate_inventory', lambda v: v['content_hash'])
    monkeypatch.setattr(d, 'validate_splits', lambda p, i: p['content_hash'])
    monkeypatch.setattr(d, 'is_subset_profile', lambda p: True)
    return inv, prof


def test_membership_is_exact_deterministic_train_only_and_bounded(monkeypatch):
    inv, prof = metadata(monkeypatch)
    value = d.build(inv, prof)
    assert value == d.build(inv, prof)
    assert value['jets'] == sum(r['jets'] for r in value['files']) == 10000
    assert len(value['files']) == 6 and len({r['source'] for r in value['files'][:3]}) == 3
    assert all(np.all(np.diff(r['entries']) == 1) for r in value['files'])
    prof['groups'][0]['role'] = 'final_test'
    with pytest.raises(PermissionError, match='TRAIN'): d.build(inv, prof)


def test_requires_explicit_profile_and_at_least_four_files(monkeypatch):
    inv, prof = metadata(monkeypatch)
    monkeypatch.setattr(d, 'is_subset_profile', lambda p: False)
    with pytest.raises(PermissionError): d.build(inv, prof)
    monkeypatch.setattr(d, 'is_subset_profile', lambda p: True)
    prof['memberships']['train']['files'] = prof['memberships']['train']['files'][:3]
    with pytest.raises(ValueError, match='four'): d.build(inv, prof)


def test_metadata_mutation_rejected(monkeypatch):
    inv, prof = metadata(monkeypatch)
    study = {'membership': d.build(inv, prof)}
    monkeypatch.setattr(d, 'metadata', lambda s: (inv, prof))
    d.validate_membership(study)
    study['membership']['files'][0]['entries'][0] += 1
    with pytest.raises(PermissionError): d.validate_membership(study)


def test_real_inventory_profile_validators_and_frozen_reader(registered, monkeypatch):
    from hlt_classification.jetclass2_delphes.split_registry import select_profile
    root, inv, _, registry = registered
    profile = select_profile(registry, inv, 'TRAIN_44')
    monkeypatch.setattr(d, 'COUNT', 8)  # synthetic engineering population only
    membership = d.build(inv, profile)
    study = dict(data_root=str(root), membership=membership, review=provisional_compatibility(inventory_hash='a'*64))
    monkeypatch.setattr(d, 'metadata', lambda s: (inv, profile))
    d.validate_membership(study)
    rows = list(d._iterate(study))
    assert len(rows) == 8 and d.identity_hash([p.identity for p in rows]) == membership['ordered_identities']
    changed = deepcopy(profile); changed['groups'][0]['role'] = 'final_test'
    with pytest.raises(ValueError): d.build(inv, changed)


def test_claim_and_role_precede_any_particle_access(monkeypatch, tmp_path):
    study = {'root': str(tmp_path)}
    spec = dict(root=str(tmp_path), name='generation_gate_r1', study={'path': str(tmp_path/'study_spec.json')},
        content_hash='a'*64, tasks=c.tasks('generation_gate'))
    opened = []
    monkeypatch.setattr(d, '_iterate', lambda *a, **k: opened.append(True))
    with pytest.raises(FileNotFoundError):
        with d.stream(spec, study, 'jg_gate', limit=64): pass
    with pytest.raises(PermissionError):
        with d.stream(spec, study, 'jg_gate'): pass
    with pytest.raises(PermissionError):
        with d.stream(spec, study, 'jg_report', limit=64): pass
    with pytest.raises(PermissionError):
        with d.stream(spec, study, 'jg_gate', limit=1): pass
    assert not opened


def test_real_root_reader_only_opens_frozen_offline_rows(monkeypatch, tmp_path):
    root = tmp_path/d.RELEASE/'jetclass2'; root.mkdir(parents=True)
    columns = {}
    values = dict(px=[2., 10.], py=[0., 0.], pz=[0., 0.], energy=[2.1, 10.1], charge=[1, 0],
        isChargedHadron=[1, 0], isNeutralHadron=[0, 1], isPhoton=[0, 0], isElectron=[0, 0], isMuon=[0, 0],
        d0val=[.2, 0], dzval=[-.1, 0], d0err=[.01, 0], dzerr=[.02, 0])
    columns['jet_nparticles'] = np.full(3000, 2, dtype=np.int32)
    columns.update({'part_'+k: ak.Array([v]*3000) for k, v in values.items()})
    # No native HLT or label branches even exist in this synthetic fixture.
    files = []
    for i in range(4):
        path = root/f'{i}.root'
        with uproot.recreate(path) as handle: handle.mktree('tree', columns)
        files.append(dict(path=path.name, sha256=sha256_file(path), source=str(i), tree_key='tree;1', entries=3000))
    inv, prof = metadata(monkeypatch, files)
    study = dict(root=str(tmp_path/'study'), data_root=str(root), membership=d.build(inv, prof),
                 review=provisional_compatibility(inventory_hash='a'*64))
    monkeypatch.setattr(d, 'metadata', lambda s: (inv, prof))
    seen = []
    actual = d.authenticated_open
    @contextmanager
    def opened(path, digest):
        seen.append(Path(path).name)
        with actual(path, digest) as h: yield h
    monkeypatch.setattr(d, 'authenticated_open', opened)
    result = list(d._iterate(study))
    assert len(result) == 10000 and len(set(p.identity for p in result)) == 10000
    assert len(seen) == 4 and all(p.hlt is None and len(p.offline) == 2 for p in result)
    assert d.identity_hash([p.identity for p in result]) == study['membership']['ordered_identities']
    assert set(d.BRANCHES) == set(columns)
    bad = deepcopy(study); bad['membership']['files'][0]['sha256'] = 'f'*64
    with pytest.raises(ValueError): list(d._iterate(bad, limit=64))


@pytest.mark.parametrize('compression', ['stored', 'deflate'])
def test_lossless_cache_has_no_source_keys_and_is_deterministic(tmp_path, compression):
    p = particles()
    rows = [(f'{i:064x}', p if i else p.take([]), 'a'*64) for i in range(3)]
    arrays = e.pack(rows)
    blob = e.encode(arrays, compression)
    assert blob == e.encode(arrays, compression)
    with zipfile.ZipFile(io.BytesIO(blob)) as handle:
        assert set(handle.namelist()) == {k+'.npy' for k in (*e.FIELDS, 'offsets', 'jet_identity')}
    path = tmp_path/'block.npz'; path.write_bytes(blob)
    assert e.readback(path, arrays) == [e.physical_digest(i, p) for i, p, _ in rows]
    bad = dict(arrays, tracking=arrays['tracking']*2)
    with pytest.raises(ValueError, match='readback'): e.readback(path, bad)


def test_real_generator_serial_spawn_trace_chunk_compression_parity(tmp_path, bundle):
    results = []
    for name, workers, chunk, compression, trace in (
            ('serial', 1, 8, 'stored', False), ('trace', 1, 32, 'stored', True),
            ('process', 4, 8, 'stored', False), ('compressed', 4, 128, 'deflate', False)):
        results.append(e.process(iter(sample()), bundle, root=tmp_path, relative=name,
            workers=workers, chunk=chunk, compression=compression, trace=trace, expected_count=12, block_jets=5))
    assert e.parity(results)
    assert all(len(r['blocks']) == 3 and r['generation_cpu_seconds'] > 0 for r in results)
    assert results[0]['ordered_identities'] == d.identity_hash([p.identity for p in sample()])
    altered = deepcopy(results[0]); altered['generation_key_digest'] = 'f'*64
    with pytest.raises(ValueError, match='Generation changed'): e.parity([results[0], altered])


def test_generator_refuses_native_hlt_and_unregistered_threads(bundle, monkeypatch):
    e.initialize(bundle)
    p = sample()[0]
    with pytest.raises(PermissionError): e.generate_chunk([Pair(p.identity, p.source_group, p.offline, p.offline)])
    monkeypatch.setenv('OMP_NUM_THREADS', '8')
    with pytest.raises(ValueError, match='native'): list(e.parallel(iter(sample()), bundle, 4, 32))


@pytest.mark.parametrize('expected', [1, 3])
def test_no_skipped_or_extra_jets(tmp_path, bundle, expected):
    with pytest.raises(ValueError, match='(exceeded|omitted)'):
        e.process(iter(sample(2)), bundle, root=tmp_path, relative='incomplete', workers=1,
                  chunk=8, compression='stored', expected_count=expected)


def test_plan_resource_shapes_explicit_authorization_and_no_production():
    study = dict(root='/study', project_dir='/project', site=c.site('tier3'))
    spec = artifact('GEN_STAGE', root='/study', name='generation_screen_r1', stage='generation_screen', tasks=c.tasks('generation_screen'))
    plan = dev.command_plan(spec, study)
    assert len(plan['commands']) == 11 and plan['cpu_upper_bound'] == 120 and plan['gpus'] == 0
    assert plan['commands'][-1]['depends_on'] == [t['task_id'] for t in spec['tasks'][:-1]]
    assert all('--partition=tier3' in r['argv'] and not any('gpu' in a for a in r['argv']) for r in plan['commands'])
    assert dev.PHRASES['generation_screen'] == 'AUTHORIZE CMS2JC2 GENERATION_SCREEN EXACT PLAN'
    with pytest.raises(ValueError): c.tasks('production')
    with pytest.raises(ValueError): c.site('tier3,debug')


def gate_row():
    return dict(serial=dict(processing_seconds=5., output_bytes=1_000_000),
                input_read_seconds=1., measurement=dict(sampled_peak_tree_rss_bytes=100_000_000))


def test_projections_use_serial_no_scaling_and_enforce_time_memory_disk():
    row = gate_row(); p = c.projection(row)
    assert p['seconds'] == 3600+2*6*10000/64 and p['resource_envelope_ok']
    for key, value in [('time', 10000.), ('memory', 100*storage.GIB), ('disk', 100_000_000)]:
        r = deepcopy(row)
        if key == 'time': r['serial']['processing_seconds'] = value
        if key == 'memory': r['measurement']['sampled_peak_tree_rss_bytes'] = value
        if key == 'disk': r['serial']['output_bytes'] = value
        assert not c.projection(r)['resource_envelope_ok']
    row['serial']['processing_seconds'] = float('nan')
    with pytest.raises(ValueError): c.projection(row)


def test_read_only_partition_probes_use_real_shapes(monkeypatch):
    calls = []
    def call(argv):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout='estimated start', stderr='')
    monkeypatch.setattr(q, 'scheduler', call)
    row = q.probe()
    assert len(calls) == 10 and not row['jobs_submitted']
    assert all(a[0:2] == ['sbatch', '--test-only'] for a in calls)
    assert 'automatic' in row['caveat']


def fake_run(setting, repeat):
    return dict(setting=setting, repeat=repeat, jets=10000, particles=20000, ordered_identities='a'*64,
        physical_digest='b'*64, generation_key_digest='c'*64, flags={},
        processing_seconds=100/setting['cpus'], total_worker_seconds=150/setting['cpus'], output_bytes=10_000_000)


def test_report_requires_every_repeat_and_exposes_no_scaling_guarantee():
    rows = [fake_run(s, r) for s in c.SETTINGS for r in range(2)]
    result = w.summarize(rows)
    assert len(result['settings']) == 5 and len(result['projections']) == 3
    assert all(r['jobs'] >= 2 and r['end_to_end_hours'] >= r['processing_hours'] for r in result['projections'])
    with pytest.raises(ValueError, match='both repeats'): w.summarize(rows[:-1])


def test_new_stage_routes_without_old_cms_context(monkeypatch):
    marker = object()
    monkeypatch.setattr(w, 'run', lambda s, t: marker)
    monkeypatch.setattr(dev_worker, 'context', lambda *a: pytest.fail('CMS data context entered'))
    assert dev_worker.run({'contract': c.CONTRACT}, 'jg_gate') is marker


def test_source_freeze_refuses_changed_donor_before_generation(monkeypatch, tmp_path):
    # Full donor auditing is inherited; test that the new capability doesn't
    # relax scientific bytes or the old no-JC2 DEV_STUDY boundary.
    source = artifact('SOURCE', commit='a'*40, files={'response.py': 'b'*64})
    proto = c.protocol()
    membership = artifact('GEN_MEMBERSHIP')
    study = artifact('GEN_STUDY', parents={'source': source['content_hash'], 'protocol': proto['content_hash'],
        'membership': membership['content_hash']}, source=source, protocol=proto, membership=membership,
        site=c.site('tier3'), root=str(tmp_path/'study'), project_dir=str(tmp_path/'project'),
        data_root=str(tmp_path/d.RELEASE/'jetclass2'), particle_roles=['train'], native_hlt_access=False,
        production_qualified=False, donor='parent', review={}, numerical_environment={})
    previous = dict(source={'files': {'response.py': 'c'*64}}, review={}, numerical_environment={})
    monkeypatch.setattr(c, 'checked_file', lambda v: v)
    monkeypatch.setattr(c, 'load_json', lambda p: {'study': 'previous'} if p == 'parent' else previous)
    monkeypatch.setattr(d, 'validate_membership', lambda s: None)
    monkeypatch.setattr(c, 'freeze', lambda *a: pytest.fail('Should reject source before donor runtime'))
    with pytest.raises(ValueError, match='scientific source'): c.validate_study(study, source=False)
    bad = dict(study, particle_roles=['train', 'final_test'])
    with pytest.raises(PermissionError, match='boundary'): c.validate_study(with_content_hash(bad), source=False)


def test_campaign_creation_separate_gate_and_bundle_authentication(monkeypatch, tmp_path, bundle):
    inv, profile = metadata(monkeypatch)
    inventory_path, profile_path = tmp_path/'inventory.json', tmp_path/'profile.json'
    inventory_path.write_text(json.dumps(inv)); profile_path.write_text(json.dumps(profile))
    previous_root, project = tmp_path/'old', tmp_path/'project'
    previous_root.mkdir(); project.mkdir()
    data_root = tmp_path/d.RELEASE/'jetclass2'; data_root.mkdir(parents=True)
    source = artifact('SOURCE', commit='a'*40, files={'frozen_science.py': 'b'*64})
    env = artifact('NUMERICAL_ENVIRONMENT')
    previous = artifact('DEV_STUDY', root=str(previous_root), source=source, numerical_environment=env,
                        review=provisional_compatibility(inventory_hash='c'*64))
    old_study = dev.write(previous_root, 'study_spec.json', previous, 'DEV_STUDY')
    donor = artifact('BDZ_JOINT_STAGE', stage='joint_compare', study=file_ref(old_study))
    donor_path = dev.write(previous_root, 'stage_spec.json', donor, 'BDZ_JOINT_STAGE')
    monkeypatch.setattr(dev, 'source_snapshot', lambda *a, **kw: source)
    monkeypatch.setattr(c, 'numerical_environment', lambda: env)
    # Isolate the new campaign; the full original JOINT ancestry is regression-tested separately.
    monkeypatch.setattr(c, 'freeze', lambda parent: deepcopy(bundle))
    root = tmp_path/'new'
    spec = c.create(parent_spec=donor_path, inventory=inventory_path, profile=profile_path, data_root=data_root,
        project_dir=project, source_commit='a'*40, root=root)
    assert spec['stage'] == 'generation_gate' and len(spec['tasks']) == 1
    study = c.validate_stage(spec)
    assert study['particle_roles'] == ['train'] and study['site']['partition'] == 'tier3'
    assert not (dev.stage_dir(spec)/'submission_ledger.json').exists()
    with pytest.raises(FileNotFoundError): c.advance(dev.stage_dir(spec)/'stage_spec.json')
    (root/'frozen_bundle.json').write_text('{}')
    with pytest.raises(ValueError): c.validate_stage(spec, source=False)


def test_submission_exact_plan_idempotence_and_no_implicit_jobs(monkeypatch, tmp_path):
    study = dict(root=str(tmp_path), project_dir='/project', site=c.site('debug'))
    spec = artifact('GEN_STAGE', root=str(tmp_path), name='generation_gate_r1', stage='generation_gate', tasks=c.tasks('generation_gate'))
    plan = dev.command_plan(spec, study)
    dev.write(tmp_path, 'stages/generation_gate_r1/command_plan.json', plan, 'DEV_PLAN')
    monkeypatch.setattr(submission, 'validate_stage', lambda *a: study)
    monkeypatch.setattr(submission, 'site_checks', lambda *a: None)
    monkeypatch.setattr(submission, 'check_other_stages', lambda *a: None)
    calls = []
    def scheduler(argv):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout='12345\n', stderr='')
    monkeypatch.setattr(submission, 'scheduler', scheduler)
    assert submission.submit(spec, execute=False) == plan and not calls
    with pytest.raises(PermissionError): submission.submit(spec, execute=True)
    for _ in range(2):
        ledger = submission.submit(spec, execute=True, reviewed_plan_hash=plan['content_hash'],
            authorization_phrase=dev.PHRASES['generation_gate'])
    assert len(calls) == 1 and ledger['jobs'] == {'jg_gate': '12345'}


def test_storage_cap_counts_partial_outputs(monkeypatch, tmp_path):
    (tmp_path/'partial').write_bytes(b'x'*100)
    monkeypatch.setattr(storage, 'TOTAL_CAP', 101)
    with pytest.raises(OSError, match='exceeds'):
        storage.publish_bytes(tmp_path, 'block', b'xx', remaining_bytes=0)


def test_gate_worker_runs_trace_variant_and_does_not_auto_submit(monkeypatch, tmp_path):
    spec = artifact('GEN_STAGE', root=str(tmp_path), name='generation_gate_r1', stage='generation_gate', tasks=c.tasks('generation_gate'))
    study = dict(bundle='bundle', numerical_environment={'content_hash': 'e'*64})
    monkeypatch.setattr(c, 'validate_stage', lambda *a, **k: study)
    monkeypatch.setattr(dev_worker, 'allocation', lambda *a: {'job_id': '12345'})
    monkeypatch.setattr(submission, 'scheduler_identity', lambda *a: {})
    monkeypatch.setattr(w, 'numerical_environment', lambda: study['numerical_environment'])
    monkeypatch.setattr(w, 'checked_file', lambda x: x)
    monkeypatch.setattr(w, 'load_json', lambda x: {})
    monkeypatch.setattr(w, 'Measurement', LocalMeasurement)
    @contextmanager
    def stream(*a, **kw):
        assert kw['limit'] == 64
        yield iter(sample(64))
    monkeypatch.setattr(d, 'stream', stream)
    traces = []
    def process(source, bundle, **kw):
        traces.append(kw['trace'])
        assert len(list(source)) == 64
        return dict(jets=64, particles=128, ordered_identities='a'*64, physical_digest='b'*64,
            generation_key_digest='c'*64, flags={}, blocks=[], output_bytes=1000, processing_seconds=.1)
    monkeypatch.setattr(e, 'process', process)
    receipt = w.run(spec, 'jg_gate')
    assert traces == [False, True, False, False]
    row = dev.product(spec, 'jg_gate', 'result')
    assert row['parity'] and row['jets'] == 64 and row['projection'] == c.projection(row)
    assert receipt['owner'] == 'jg_gate'


@pytest.mark.parametrize('exit_status', [0, 7])
def test_helper_dry_by_default_exact_review_stdin_and_syntax(exit_status):
    path = Path(__file__).parents[1]/'scripts/queue_cms2jc2_generation_benchmark.sh'
    text = path.read_text()
    assert 'EXECUTE=0' in text and '--reviewed-plan-hash' in text
    assert not any(word in text for word in ('scancel', 'scontrol update', 'git push', 'frozen_confirm'))
    bash = Path(os.environ.get('ProgramFiles', 'C:/Program Files'))/'Git/bin/bash.exe'
    if not bash.exists(): pytest.skip('Bash unavailable')
    result = subprocess.run([str(bash), '-n', str(path)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    function = 'phase() {'+text.split('phase() {', 1)[1].split('\nif [ ! -e', 1)[0]
    script = 'set -euo pipefail\n'+function+'\nsleep() { command sleep 0.01; }\n'
    script += f'''result=0
phase test bash -c 'read -r line; printf "%s\\n" "$line"; exit {exit_status}' <<'INPUT' || result=$?
saved stdin
INPUT
printf 'result=%s\\n' "$result"
'''
    result = subprocess.run([str(bash), '--noprofile', '--norc', '-s'], input=script,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'saved stdin' in result.stdout and f'result={exit_status}' in result.stdout
