"""Portable engineering capability, no relaxed historical scientific contracts."""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from hlt_classification.cms2jc2_response import (
    generation_portable_campaign as c, generation_portable_data as d,
    generation_portable_worker as w, generation_benchmark_engine as e,
    generation_benchmark_campaign as gen, generation_benchmark_data as gd,
    dev_campaign as dev, dev_submission as submission, dev_worker, storage)
from hlt_classification.cms2jc2_response.contracts import artifact, load_json, sha256_file, with_content_hash
from hlt_classification.cms2jc2_response.dev_data import file_ref
from test_cms2jc2_generation_benchmark import bundle, sample, native_threads
from test_cms2jc2_bounded import LocalMeasurement


@pytest.mark.parametrize('maximum,cap,count', [(36, 52, 5), (72, 124, 7), (144, 144, 9)])
def test_profiles_cpu_only_and_bounded(maximum, cap, count):
    study = dict(project_dir='/project', site=c.site('tigris'), max_workers=maximum)
    spec = artifact('PORT_STAGE', root='/root', name='port_screen_r1', stage='port_screen',
                    tasks=c.tasks('port_screen', maximum))
    plan = dev.command_plan(spec, study)
    assert plan['cpu_upper_bound'] == cap and len(plan['commands']) == count and plan['gpus'] == 0
    for row, task in zip(plan['commands'], spec['tasks']):
        assert '--partition=tigris' in row['argv']
        assert not any('gpu' in a or a.startswith('--qos') for a in row['argv'])
        assert str(Path('/project')/c.WORKER) in row['argv']
        if task['task_id'] == 'jp_c144_r0':
            assert task['depends_on'] == ['jp_c016_r1', 'jp_c036_r1', 'jp_c072_r1']
    assert gen.site('debug')['qos'] == 'qos_tier3'
    assert max(t['cpus'] for t in gen.tasks('generation_screen')) == 16


@pytest.mark.parametrize('value', [True, 16, 37, 256])
def test_unregistered_maximum_rejected(value):
    with pytest.raises(ValueError): c.tasks('port_screen', value)


def test_new_stage_has_no_production_or_automatic_advance():
    with pytest.raises(PermissionError): c.tasks('production')
    with pytest.raises(ValueError): c.site('tigris,debug')


def physical():
    pair = sample(1)[0]
    return e.pack([(pair.identity, pair.offline, '')])


def test_cross_arch_tolerance_is_not_bitwise_parity():
    ref = physical(); actual = deepcopy(ref)
    actual['p4'][0, 0] += 1e-13
    result = d.compare_arrays(actual, ref)
    assert not result['exact_bytes'] and result['max_absolute_difference']['p4'] > 0
    assert d.compare_arrays(ref, ref)['exact_bytes']
    actual['p4'][0, 0] += .001
    with pytest.raises(ValueError, match='tolerance'): d.compare_arrays(actual, ref)


@pytest.mark.parametrize('field', ['valid', 'category', 'charge', 'jet_identity', 'offsets'])
def test_every_discrete_output_is_exact(field):
    ref = physical(); actual = deepcopy(ref)
    actual[field].flat[0] = not actual[field].flat[0] if field == 'valid' else actual[field].flat[0]+1
    with pytest.raises(ValueError, match='discrete'): d.compare_arrays(actual, ref)


def test_nonfinite_and_dtype_drift_refused():
    ref = physical(); actual = deepcopy(ref)
    actual['tracking'][0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'): d.compare_arrays(actual, ref)
    actual = deepcopy(ref); actual['p4'] = actual['p4'].astype('f4')
    with pytest.raises(ValueError, match='dtype'): d.compare_arrays(actual, ref)


@pytest.mark.parametrize('fault', ['offset', 'charge', 'invalid_tracking', 'spacelike', 'nonfinite'])
def test_vectorized_block_validation_retains_physical_invariants(tmp_path, fault):
    value = physical()
    if fault == 'offset': value['offsets'][-1] += 1
    elif fault == 'charge': value['charge'][0] = 0
    elif fault == 'invalid_tracking': value['valid'][0, 0] = False
    elif fault == 'spacelike': value['p4'][0, 3] = .01
    else: value['tracking'][0, 0] = float('nan')
    path = tmp_path/'bad.npz'; path.write_bytes(e.encode(value, 'stored'))
    with pytest.raises(ValueError): d.arrays(path)


@pytest.fixture
def portable(tmp_path, bundle):
    root = tmp_path/'packet'; root.mkdir()
    particles = sample(1)[0].offline
    ids = [hashlib.sha256(f'portable-{i}'.encode()).hexdigest() for i in range(10000)]
    blocks = {'inputs': [], 'reference': [], 'reference_gate': []}
    for prefix, count in [('inputs', 10000), ('reference', 10000), ('reference_gate', 64)]:
        for start in range(0, count, 1000):
            jets = ids[start:min(start+1000, count)]
            value = e.pack([(jet, particles, '') for jet in jets])
            path = root/prefix/f'{start//1000:05d}.npz'
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(e.encode(value, 'stored'))
            blocks[prefix].append(dict(relative=path.relative_to(root).as_posix(), sha256=sha256_file(path),
                                       bytes=path.stat().st_size, jets=len(jets)))
    source = artifact('SOURCE', commit='a'*40, files={'scientific.py': 'b'*64})
    env = artifact('NUMERICAL_ENVIRONMENT', machine='SPORC_TEST')
    def run(n, prefix):
        return dict(jets=n, particles=n*len(particles), ordered_identities=gd.identity_hash(ids[:n]),
            generation_key_digest='c'*64, physical_digest='d'*64, flags={}, blocks=blocks[prefix],
            output_bytes=sum(r['bytes'] for r in blocks[prefix]), processing_seconds=1.)
    packet = artifact('PORT_PACKET', jets=10000, role='train', candidate='JOINT', replica=0,
        native_hlt_access=False, production_qualified=False, bundle=bundle, source=source, environment=env,
        membership=artifact('GEN_MEMBERSHIP', role='train', jets=10000,
            ordered_identities=gd.identity_hash(ids), gate_ordered_identities=gd.identity_hash(ids[:64])),
        inputs=blocks['inputs'], reference=run(10000, 'reference'), gate_reference=run(64, 'reference_gate'))
    path = root/'manifest.json'; path.write_text(json.dumps(packet))
    return path, packet


def test_packet_requires_external_fingerprint_and_all_bytes(portable):
    path, packet = portable
    assert d.read_packet(path, sha256_file(path)) == packet
    with pytest.raises(ValueError, match='fingerprint'): d.read_packet(path, '0'*64)
    target = path.parent/packet['inputs'][0]['relative']
    target.write_bytes(b'bad')
    with pytest.raises(ValueError, match='bytes'): d.read_packet(path, sha256_file(path))


def test_packet_relocation_and_train_only_reader(portable, tmp_path):
    import shutil
    path, packet = portable
    moved = tmp_path/'relocated'; shutil.copytree(path.parent, moved)
    assert d.read_packet(moved/'manifest.json', sha256_file(path)) == packet
    pairs = list(d.pairs(packet, moved, limit=64))
    assert len(pairs) == 64 and all(p.hlt is None and p.diagnostic_class is None for p in pairs)
    assert pairs[0].offline.keys == tuple(f'part:{j}' for j in range(len(pairs[0].offline)))
    with pytest.raises(PermissionError): list(d.pairs(packet, moved, limit=65))
    wrong = deepcopy(packet); wrong['membership']['gate_ordered_identities'] = '0'*64
    with pytest.raises(ValueError, match='membership'): list(d.pairs(wrong, moved, limit=64))


@pytest.mark.parametrize('change', ['role', 'missing', 'alias'])
def test_packet_bad_registry_refused(portable, change):
    path, packet = portable
    bad = deepcopy(packet)
    if change == 'role': bad['role'] = 'final_test'
    elif change == 'missing': bad['inputs'].pop()
    else: bad['inputs'][0] = bad['inputs'][1]
    path.write_text(json.dumps(with_content_hash(bad)))
    with pytest.raises((ValueError, PermissionError)): d.read_packet(path, sha256_file(path))


def test_full_and_prefix_replay_check_keys_flags_and_coordinates(portable):
    path, packet = portable
    row = packet['reference']
    assert d.compare_run(row, path.parent, packet, path.parent)['exact_bytes']
    assert d.compare_run(packet['gate_reference'], path.parent, packet, path.parent, gate=True)['compatible']
    for name in ('generation_key_digest', 'flags', 'particles', 'ordered_identities'):
        bad = deepcopy(row); bad[name] = 'changed'
        with pytest.raises(ValueError, match='drift'): d.compare_run(bad, path.parent, packet, path.parent)


def test_create_tigris_keeps_own_environment_and_no_implicit_jobs(portable, monkeypatch, tmp_path):
    path, packet = portable
    project = tmp_path/'project'; project.mkdir()
    env = artifact('NUMERICAL_ENVIRONMENT', machine='TIGRIS_TEST')
    monkeypatch.setattr(dev, 'source_snapshot', lambda *a, **k: packet['source'])
    monkeypatch.setattr(c, 'numerical_environment', lambda: env)
    spec = c.create(project_dir=project, source_commit='a'*40, root=tmp_path/'study',
        packet=path, packet_sha256=sha256_file(path), partition='tigris', max_workers=72)
    study = c.validate_stage(spec)
    assert study['numerical_environment'] == env != packet['environment']
    assert not (dev.stage_dir(spec)/'submission_ledger.json').exists()
    with pytest.raises(FileNotFoundError): c.advance(dev.stage_dir(spec)/'stage_spec.json')
    bad = deepcopy(packet['source']); bad['files']['scientific.py'] = 'f'*64
    monkeypatch.setattr(dev, 'source_snapshot', lambda *a, **k: bad)
    with pytest.raises(ValueError, match='source'): c.validate_stage(spec)


def test_source_match_has_no_silent_compatibility_allowlist():
    a = dict(commit='a'*40, files={'science': 'b'*64})
    c.validate_source_match(a, a)
    with pytest.raises(ValueError): c.validate_source_match(a, dict(a, commit='c'*40))


def test_real_portable_reader_generator_serial_spawn_and_export_keys(portable, tmp_path):
    path, packet = portable
    pairs = list(d.pairs(packet, path.parent, limit=64))
    exported = tmp_path/'exported'; exported.mkdir()
    blocks = []
    assert list(d.export_stream(iter(pairs), exported, blocks)) == pairs
    assert len(blocks) == 1 and blocks[0]['jets'] == 64
    decoded = d.arrays(exported/'packet'/blocks[0]['relative'])
    assert len(decoded['jet_identity']) == 64
    results = []
    for name, workers, trace in [('reference', 1, False), ('trace', 1, True), ('spawn', 4, False)]:
        results.append(e.process(iter(pairs), packet['bundle'], root=tmp_path, relative=name,
            workers=workers, chunk=8, compression='stored', expected_count=64, trace=trace))
    assert e.parity(results)
    reference = deepcopy(packet)
    reference['gate_reference'] = results[0]
    assert d.compare_run(results[-1], tmp_path, reference, tmp_path, gate=True)['exact_bytes']


def test_export_creation_reuses_real_gate_without_old_root_writes(portable, monkeypatch, tmp_path):
    path, packet = portable
    old_root = tmp_path/'old'; old_root.mkdir()
    project = tmp_path/'project'; project.mkdir()
    original = dict(root=str(old_root), data_root=str(tmp_path/'raw'), source=packet['source'], numerical_environment=packet['environment'])
    parent = artifact('GEN_STAGE', stage='generation_gate')
    parent_path = old_root/'stage_spec.json'; parent_path.write_text(json.dumps(parent))
    before = parent_path.read_bytes()
    monkeypatch.setattr(gen, 'validate_stage', lambda *a, **k: original)
    monkeypatch.setattr(gen, 'accepted', lambda *a, **k: {'passed': True})
    monkeypatch.setattr(dev, 'source_snapshot', lambda *a, **k: packet['source'])
    monkeypatch.setattr(c, 'numerical_environment', lambda: packet['environment'])
    spec = c.create(project_dir=project, source_commit='a'*40, root=tmp_path/'export',
                    parent_spec=parent_path, partition='debug')
    assert spec['stage'] == 'port_export' and spec['tasks'] == c.tasks('port_export')
    assert c.validate_stage(spec)['site']['partition'] == 'debug'
    assert before == parent_path.read_bytes()
    with pytest.raises(PermissionError):
        c.create(project_dir=project, source_commit='a'*40, root=tmp_path/'raw'/'forbidden',
                 parent_spec=parent_path, partition='tier3')
    assert not (tmp_path/'raw'/'forbidden').exists()
    monkeypatch.setattr(gen, 'accepted', lambda *a, **k: (_ for _ in ()).throw(PermissionError('not complete')))
    with pytest.raises(PermissionError, match='complete'): c.validate_stage(spec)


def test_claimed_tigris_gate_and_full_report_lifecycle(portable, monkeypatch, tmp_path):
    path, packet = portable
    project = tmp_path/'project'; project.mkdir()
    env = artifact('NUMERICAL_ENVIRONMENT', machine='TIGRIS_TEST')
    monkeypatch.setattr(dev, 'source_snapshot', lambda *a, **k: packet['source'])
    monkeypatch.setattr(c, 'numerical_environment', lambda: env)
    monkeypatch.setattr(w, 'numerical_environment', lambda: env)
    monkeypatch.setattr(w, 'Measurement', LocalMeasurement)
    monkeypatch.setattr(dev_worker, 'allocation', lambda *a: {'job_id': '123'})
    monkeypatch.setattr(submission, 'scheduler_identity', lambda *a: {})
    spec = c.create(project_dir=project, source_commit='a'*40, root=tmp_path/'replay',
        packet=path, packet_sha256=sha256_file(path), partition='tigris')
    # Use real packet reads, output serialization/checksums, claims, source/stage
    # validators and receipts; replace only the expensive 10k science kernel.
    # The preceding test separately exercises actual Generator serial/spawn.
    def process(source, bundle, **kw):
        count = sum(1 for _ in source)
        assert count == kw['expected_count']
        ref = packet['gate_reference'] if count == 64 else packet['reference']
        row = deepcopy(ref); row['blocks'] = []
        for i, block in enumerate(ref['blocks']):
            target = storage.publish_bytes(kw['root'], f'{kw["relative"]}/{i:05d}.npz',
                (path.parent/block['relative']).read_bytes(), remaining_bytes=0)
            row['blocks'].append(dict(block, relative=target.relative_to(kw['root']).as_posix()))
        row['processing_seconds'] = 1e-6
        return row
    monkeypatch.setattr(e, 'process', process)
    receipt = dev_worker.run(spec, 'jp_gate')
    assert receipt['owner'] == 'jp_gate'
    result = w.read(spec)
    assert result['within_site_exact'] and result['compatible']
    assert result['projection']['resource_envelope_ok']
    with pytest.raises(FileExistsError): dev_worker.run(spec, 'jp_gate')
    screen = c.advance(dev.stage_dir(spec)/'stage_spec.json')
    with pytest.raises(FileNotFoundError): w.read(screen)
    for task in screen['tasks']:
        dev_worker.run(screen, task['task_id'])
    report = w.read(screen)
    assert len(report['settings']) == 2 and not report['production_qualified']
    assert all(s['exact_sporc_bytes'] for s in report['settings'])
    assert not report['raw_root_ingestion_measured']
    with pytest.raises(PermissionError, match='production'):
        c.advance(dev.stage_dir(screen)/'stage_spec.json')
    assert not (dev.stage_dir(screen)/'submission_ledger.json').exists()


def test_measurement_admission_no_assumed_speedup():
    row = dict(serial=dict(processing_seconds=1., output_bytes=10000),
               measurement=dict(sampled_peak_tree_rss_bytes=1000000))
    assert c.projection(row, 36)['resource_envelope_ok']
    assert c.resources(c.projection(row, 36)) == dict(hours=2, memory_gib=32)
    assert c.projection(row, 144)['memory_bytes'] > c.projection(row, 36)['memory_bytes']
    row['serial']['processing_seconds'] = 1000.
    assert not c.projection(row, 36)['resource_envelope_ok']
    row['serial']['processing_seconds'] = float('nan')
    with pytest.raises(ValueError): c.projection(row, 36)


def test_worker_claim_and_environment_before_input_reads(monkeypatch, tmp_path):
    spec = artifact('PORT_STAGE', root=str(tmp_path), name='port_gate_r1', stage='port_gate', tasks=c.tasks('port_gate'))
    study = dict(numerical_environment={'test': 'frozen'})
    monkeypatch.setattr(c, 'validate_stage', lambda *a, **k: study)
    monkeypatch.setattr(dev_worker, 'allocation', lambda *a: {'job_id': '123'})
    monkeypatch.setattr(submission, 'scheduler_identity', lambda *a: {})
    monkeypatch.setattr(w, 'numerical_environment', lambda: {'test': 'wrong'})
    monkeypatch.setattr(d, 'pairs', lambda *a, **k: pytest.fail('Read before environment gate'))
    with pytest.raises(ValueError, match='environment'): w.run(spec, 'jp_gate')
    assert not (dev.stage_dir(spec)/'claims').exists()


def test_export_worker_publishes_self_contained_authenticated_packet(portable, monkeypatch, tmp_path):
    path, packet = portable
    old_root = tmp_path/'original'; old_root.mkdir()
    project = tmp_path/'project'; project.mkdir()
    bundle_path = dev.write(old_root, 'bundle.json', packet['bundle'], 'GEN_BUNDLE')
    parent = artifact('GEN_STAGE', stage='generation_gate', root=str(path.parent))
    parent_path = old_root/'stage_spec.json'; parent_path.write_text(json.dumps(parent))
    original = dict(root=str(old_root), data_root=str(tmp_path/'raw'), source=packet['source'], numerical_environment=packet['environment'],
        bundle=file_ref(bundle_path), membership=packet['membership'], review={'synthetic': True})
    gate = artifact('GEN_GATE', serial=packet['gate_reference'])
    monkeypatch.setattr(gen, 'validate_stage', lambda *a, **k: original)
    monkeypatch.setattr(gen, 'accepted', lambda *a, **k: gate)
    monkeypatch.setattr(dev, 'source_snapshot', lambda *a, **k: packet['source'])
    monkeypatch.setattr(c, 'numerical_environment', lambda: packet['environment'])
    monkeypatch.setattr(w, 'numerical_environment', lambda: packet['environment'])
    monkeypatch.setattr(w, 'Measurement', LocalMeasurement)
    monkeypatch.setattr(dev_worker, 'allocation', lambda *a: {'job_id': '123'})
    monkeypatch.setattr(submission, 'scheduler_identity', lambda *a: {})
    monkeypatch.setattr(gd, 'validate_membership', lambda *a: None)
    monkeypatch.setattr(gd, '_iterate', lambda *a: d.pairs(packet, path.parent))
    def process(source, bundle, **kw):
        assert sum(1 for _ in source) == 10000
        result = deepcopy(packet['reference']); result['blocks'] = []
        for i, block in enumerate(packet['reference']['blocks']):
            target = storage.publish_bytes(kw['root'], f'{kw["relative"]}/{i:05d}.npz',
                (path.parent/block['relative']).read_bytes(), remaining_bytes=0)
            result['blocks'].append(dict(block, relative=target.relative_to(kw['root']).as_posix()))
        return result
    monkeypatch.setattr(e, 'process', process)
    spec = c.create(project_dir=project, source_commit='a'*40, root=tmp_path/'export',
                    parent_spec=parent_path, partition='tier3')
    receipt = w.run(spec, 'jp_export')
    result = w.read(spec)
    assert len(receipt['outputs']) == 23  # 10 inputs, 10 outputs, gate, manifest, report.
    imported = d.read_packet(result['packet']['path'], result['packet']['sha256'])
    assert imported['membership'] == packet['membership'] and imported['bundle'] == packet['bundle']
    assert imported['reference']['physical_digest'] == packet['reference']['physical_digest']
    assert not (dev.stage_dir(spec)/'submission_ledger.json').exists()


def test_portable_submission_dry_hash_idempotence_and_ambiguous_refusal(monkeypatch, tmp_path):
    study = dict(root=str(tmp_path), project_dir='/project', site=c.site('tigris'), max_workers=36)
    spec = artifact('PORT_STAGE', root=str(tmp_path), name='port_gate_r1', stage='port_gate', tasks=c.tasks('port_gate'))
    plan = dev.command_plan(spec, study)
    dev.write(tmp_path, 'stages/port_gate_r1/command_plan.json', plan, 'DEV_PLAN')
    monkeypatch.setattr(submission, 'validate_stage', lambda *a, **k: study)
    monkeypatch.setattr(submission, 'site_checks', lambda *a: None)
    monkeypatch.setattr(submission, 'check_other_stages', lambda *a: None)
    calls = []
    def scheduler(argv):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout='12345\n', stderr='')
    monkeypatch.setattr(submission, 'scheduler', scheduler)
    assert submission.submit(spec) == plan and not calls
    with pytest.raises(PermissionError): submission.submit(spec, execute=True)
    for _ in range(2):
        result = submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES['port_gate'],
                                   reviewed_plan_hash=plan['content_hash'])
    assert len(calls) == 1 and result['jobs']['jp_gate'] == '12345'
    (submission.journal(spec, 'jp_gate')/'receipt.json').unlink()
    with pytest.raises(RuntimeError, match='Unacknowledged'):
        submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES['port_gate'],
                          reviewed_plan_hash=plan['content_hash'])
    assert len(calls) == 1


def test_tigris_scheduler_does_not_require_sporc_qos_but_identity_stays_exact(monkeypatch, tmp_path):
    study = dict(project_dir='/project', site=c.site('tigris'), max_workers=36)
    spec = artifact('PORT_STAGE', root=str(tmp_path), name='port_gate_r1', stage='port_gate', tasks=c.tasks('port_gate'))
    plan = dev.command_plan(spec, study); row = plan['commands'][0]
    intent = artifact('DEV_SUBMIT_INTENT', parents={'stage': spec['content_hash'], 'plan': plan['content_hash']},
                      task_id='jp_gate', argv=row['argv'])
    dev.write(tmp_path, 'stages/port_gate_r1/submission/jp_gate/intent.json', intent, 'DEV_SUBMIT_INTENT')
    monkeypatch.setenv('USER', 'ryreu')
    fields = dict(JobId='123', JobState='RUNNING', Comment=f'c2jd:{spec["content_hash"]}:jp_gate',
        WorkDir='/project', Command=str(Path('/project')/c.WORKER), UserId='ryreu(42)', Partition='tigris',
        Account='reu-aisocial', QOS='site_default', NumCPUs='4', NumNodes='1', ReqTRES='cpu=4,mem=32G')
    monkeypatch.setattr(submission, 'scheduler', lambda *a: SimpleNamespace(returncode=0, stderr='',
        stdout=' '.join(f'{k}={v}' for k, v in fields.items())))
    assert submission.scheduler_identity(spec, study, 'jp_gate', '123')['QOS'] == 'site_default'
    fields['ReqTRES'] += ',gres/gpu=1'
    with pytest.raises(PermissionError): submission.scheduler_identity(spec, study, 'jp_gate', '123')


@pytest.mark.parametrize('status', [0, 7])
def test_helpers_syntax_heartbeat_stdin_and_exit(status):
    root = Path(__file__).parents[1]
    bash = Path(os.environ.get('ProgramFiles', 'C:/Program Files'))/'Git/bin/bash.exe'
    if not bash.exists(): pytest.skip('Bash unavailable')
    helper = root/'scripts/queue_cms2jc2_portable_benchmark.sh'
    for path in (helper, root/c.WORKER):
        result = subprocess.run([str(bash), '-n', str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    text = helper.read_text()
    assert 'EXECUTE=0' in text and '--reviewed-plan-hash' in text and 'scancel' not in text
    function = 'phase() {'+text.split('phase() {', 1)[1].split('\nphase "Check', 1)[0]
    script = 'set -euo pipefail\n'+function+'\nsleep() { command sleep 0.01; }\n'
    script += f'''result=0
phase test bash -c 'read -r line; printf "%s\\n" "$line"; exit {status}' <<'INPUT' || result=$?
kept stdin
INPUT
printf 'status=%s\\n' "$result"
'''
    result = subprocess.run([str(bash), '--noprofile', '--norc', '-s'], input=script,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and 'kept stdin' in result.stdout and f'status={status}' in result.stdout
