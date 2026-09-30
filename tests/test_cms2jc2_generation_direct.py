"""Direct Tigris registration, old-reference reuse, roles, replay and submission."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from hlt_classification.cms2jc2_response import (
    generation_direct_campaign as c, generation_direct_worker as w,
    generation_benchmark_campaign as gen, generation_benchmark_data as data,
    generation_benchmark_engine as engine, generation_portable_data as portable,
    dev_campaign as dev, dev_submission as submission, dev_worker, storage)
from hlt_classification.cms2jc2_response.contracts import artifact, load_json, with_content_hash, sha256_file
from hlt_classification.cms2jc2_response.dev_data import file_ref
from hlt_classification.cms2jc2_response.readers import Pair
from hlt_classification.jetclass2_delphes.contracts import row_identity
from test_cms2jc2_generation_benchmark import bundle, metadata, sample, native_threads
from test_cms2jc2_bounded import LocalMeasurement


@pytest.fixture
def donor(tmp_path, monkeypatch, bundle):
    old = tmp_path/'original'; old.mkdir()
    project = tmp_path/'project'; project.mkdir()
    inv, profile = metadata(monkeypatch)
    membership = data.build(inv, profile)
    inv_path = old/'inventory.json'; inv_path.write_text(json.dumps(inv))
    prof_path = old/'profile.json'; prof_path.write_text(json.dumps(profile))
    model = dev.write(old, 'frozen_bundle.json', bundle, 'GEN_BUNDLE')
    source = artifact('SOURCE', commit='a'*40, files={'science.py': 'f'*64,
        'src/hlt_classification/cms2jc2_response/dev_campaign.py': '1'*64})
    original = artifact('GEN_STUDY', parents={'source': source['content_hash'],
        'protocol': gen.protocol()['content_hash'], 'membership': membership['content_hash']},
        root=str(old), project_dir=str(tmp_path/'oldproject'), source=source, protocol=gen.protocol(),
        membership=membership, inventory=file_ref(inv_path), profile=file_ref(prof_path),
        bundle=file_ref(model), data_root=str(tmp_path/data.RELEASE/'jetclass2'), review={'fixture': True},
        numerical_environment=artifact('NUMERICAL_ENVIRONMENT', machine='SPORC'),
        site=gen.site('tier3'), particle_roles=['train'], native_hlt_access=False, production_qualified=False)
    dev.write(old, 'study_spec.json', original, 'GEN_STUDY')
    parent = artifact('GEN_STAGE', parents={'study': original['content_hash'], 'protocol': gen.protocol()['content_hash']},
        root=str(old), stage='generation_gate', name='generation_gate_r1', tasks=gen.tasks('generation_gate'),
        study=file_ref(old/'study_spec.json'), parent_spec=None, protocol=gen.protocol(), scientific_qualification=False)
    parent_path = dev.write(old, 'stages/generation_gate_r1/stage_spec.json', parent, 'GEN_STAGE')
    particle = sample(1)[0].offline
    pairs = [Pair(row_identity(inv['content_hash'], row['path'], row['tree_key'], entry),
                  row['sha256'], particle, None) for row in membership['files'] for entry in row['entries']]
    reference = engine.process(iter(pairs[:64]), bundle, root=old, relative='reference',
                               workers=1, chunk=8, compression='stored', expected_count=64)
    fields = dict(jets=64, parity=True, serial=reference, replay_runs=[reference]*3,
                  measurement={'sampled_peak_tree_rss_bytes': 1000000}, input_read_seconds=.01)
    gate = artifact('GEN_GATE', parents=gen.parents(parent), projection=gen.projection(fields), **fields)
    path = dev_worker.publish_result(parent, 'jg_gate', gate, 'GEN_GATE')
    dev_worker.publish_outputs(parent, 'jg_gate', {'result': path, 'block': old/reference['blocks'][0]['relative']})
    calls = []
    def validate_parent(spec, **kwargs):
        assert spec == parent and kwargs == {'source': False}
        calls.append('full_ancestry')
        return original
    monkeypatch.setattr(gen, 'validate_stage', validate_parent)
    current = with_content_hash(dict(source, commit='b'*40, files={**source['files'],
        'src/hlt_classification/cms2jc2_response/dev_campaign.py': '2'*64}))
    env = artifact('NUMERICAL_ENVIRONMENT', machine='TIGRIS')
    monkeypatch.setattr(dev, 'source_snapshot', lambda *a, **k: current)
    monkeypatch.setattr(c, 'numerical_environment', lambda: env)
    monkeypatch.setattr(w, 'numerical_environment', lambda: env)
    monkeypatch.setattr(w, 'Measurement', LocalMeasurement)
    monkeypatch.setattr(dev_worker, 'allocation', lambda *a: {'job_id': '123'})
    monkeypatch.setattr(submission, 'scheduler_identity', lambda *a: {})
    reads = []
    def iterate(study, *, limit=None):
        assert study == original
        reads.append(limit)
        yield from pairs[:limit]
    monkeypatch.setattr(data, '_iterate', iterate)
    return dict(parent=parent, parent_path=parent_path, original=original, project=project,
                root=tmp_path/'direct', source=current, env=env, pairs=pairs,
                reference=reference, calls=calls, reads=reads, bundle=bundle)


def create(donor):
    return c.create(parent_spec=donor['parent_path'], project_dir=donor['project'],
                    source_commit='b'*40, root=donor['root'])


def test_create_metadata_only_and_import_preserves_old_environment(donor):
    spec = create(donor)
    study = c.validate_stage(spec)
    original, _, bundle = c.inputs(study)
    assert study['numerical_environment'] == donor['env'] != original['numerical_environment']
    assert bundle == donor['bundle'] and original == donor['original']
    assert donor['calls'] == ['full_ancestry'] and donor['reads'] == []
    assert spec['stage'] == 'tigris_direct_gate'
    assert not (dev.stage_dir(spec)/'submission_ledger.json').exists()
    with pytest.raises(FileNotFoundError): c.advance(dev.stage_dir(spec)/'stage_spec.json')
    with pytest.raises(FileExistsError): create(donor)


@pytest.mark.parametrize('target', ['science.py', 'generation_benchmark_engine.py', 'generation_benchmark_data.py'])
def test_only_existing_execution_surface_may_change(target):
    key = target if target == 'science.py' else 'src/hlt_classification/cms2jc2_response/'+target
    with pytest.raises(ValueError, match='scientific source'):
        c.source_match({'files': {key: 'a'*64}}, {'files': {key: 'b'*64}})


@pytest.mark.parametrize('target', ['bundle', 'inventory', 'profile', 'original_study', 'reference'])
def test_import_corruption_fails_closed(donor, target):
    spec = create(donor)
    if target == 'original_study':
        path = Path(donor['parent']['study']['path'])
    elif target == 'reference':
        path = Path(donor['original']['root'])/donor['reference']['blocks'][0]['relative']
    else:
        path = Path(donor['original'][target]['path'])
    path.write_bytes(b'corrupt')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        c.validate_stage(spec)


@pytest.mark.parametrize('stage', ['production', 'port_export', 'generation_screen'])
def test_no_production_export_or_wrong_stage(stage):
    with pytest.raises(PermissionError): c.tasks(stage)


def test_cpu_plan_has_six_replays_dependency_lanes_and_no_gpu():
    study = dict(project_dir='/project', site=c.port.site('tigris'))
    spec = artifact('TG_STAGE', root='/root', name='tigris_direct_screen_r1', stage='tigris_direct_screen',
                    tasks=c.tasks('tigris_direct_screen', hours=5, memory_gib=64))
    plan = dev.command_plan(spec, study)
    assert plan['cpu_upper_bound'] == 124 and plan['gpus'] == 0
    assert len(plan['commands']) == 7
    for row, task in zip(plan['commands'], spec['tasks']):
        assert '--partition=tigris' in row['argv']
        assert not any('gpu' in a or a.startswith('--qos') for a in row['argv'])
        assert str(Path('/project')/c.WORKER) in row['argv']
        if task['action'] == 'jt_run':
            assert task['cpus'] == task['params']['workers'] in (16, 36, 72)
            assert task['hours'] == 5 and task['memory_gib'] == 64
            assert len(task['depends_on']) == task['params']['repeat']
    assert len(spec['tasks'][-1]['depends_on']) == 6
    assert c.tasks('tigris_direct_gate')[0]['cpus'] == 4


def test_projection_includes_root_time_and_rechecks_free_space(tmp_path, monkeypatch):
    row = dict(serial={'processing_seconds': 1., 'output_bytes': 10000}, input_read_seconds=2.,
               measurement={'sampled_peak_tree_rss_bytes': 1e6})
    projection = c.projection(row)
    assert projection['seconds'] == 3600+2*3*10000/64
    assert projection['resource_envelope_ok']
    c.storage_check(tmp_path, projection)
    monkeypatch.setattr(c.shutil, 'disk_usage', lambda p: SimpleNamespace(free=1))
    with pytest.raises(OSError, match='nothing deleted'): c.storage_check(tmp_path, projection)
    row['input_read_seconds'] = float('nan')
    with pytest.raises(ValueError, match='input timing'): c.projection(row)


def test_root_and_stage_drift_refused(donor):
    with pytest.raises(PermissionError, match='nest'):
        c.create(parent_spec=donor['parent_path'], project_dir=donor['project'], source_commit='b'*40,
                 root=Path(donor['original']['root'])/'nested')
    spec = create(donor)
    wrong = deepcopy(spec)
    wrong['tasks'][0]['cpus'] = 72
    with pytest.raises(ValueError, match='registration'):
        c.validate_stage(with_content_hash(wrong))
    wrong = deepcopy(spec)
    wrong['parent_spec'] = file_ref(donor['parent_path'])
    with pytest.raises(PermissionError, match='Unexpected'):
        c.validate_stage(with_content_hash(wrong))


def test_cross_site_reference_is_only_64_and_exact_discrete(donor):
    reference = donor['reference']
    original = donor['original']
    assert w.compare_reference(reference, original['root'], original, {'serial': reference})['exact_bytes']
    for key in ('jets', 'flags', 'generation_key_digest'):
        wrong = deepcopy(reference)
        wrong[key] = 10000 if key == 'jets' else 'changed'
        with pytest.raises(ValueError):
            w.compare_reference(wrong, original['root'], original, {'serial': reference})


def test_tigris_scheduler_identity_default_qos_and_exact_wrapper(monkeypatch, tmp_path):
    study = dict(project_dir='/project', site=c.port.site('tigris'))
    spec = artifact('TG_STAGE', root=str(tmp_path), name='tigris_direct_gate_r1',
                    stage='tigris_direct_gate', tasks=c.tasks('tigris_direct_gate'))
    plan = c.command_plan(spec, study)
    intent = artifact('DEV_SUBMIT_INTENT', parents={'stage': spec['content_hash'], 'plan': plan['content_hash']},
                      task_id='jt_gate', argv=plan['commands'][0]['argv'])
    dev.write(tmp_path, 'stages/tigris_direct_gate_r1/submission/jt_gate/intent.json', intent, 'DEV_SUBMIT_INTENT')
    monkeypatch.setenv('USER', 'ryreu')
    fields = dict(JobId='123', JobState='RUNNING', Comment=f'c2jd:{spec["content_hash"]}:jt_gate',
        WorkDir='/project', Command=str(Path('/project')/c.WORKER), UserId='ryreu(42)', Partition='tigris',
        Account='reu-aisocial', QOS='site_default', NumCPUs='4', NumNodes='1', ReqTRES='cpu=4,mem=32G')
    monkeypatch.setattr(submission, 'scheduler', lambda *a: SimpleNamespace(returncode=0, stderr='',
        stdout=' '.join(f'{k}={v}' for k, v in fields.items())))
    assert submission.scheduler_identity(spec, study, 'jt_gate', '123')['QOS'] == 'site_default'
    fields['ReqTRES'] += ',gres/gpu=1'
    with pytest.raises(PermissionError): submission.scheduler_identity(spec, study, 'jt_gate', '123')


def test_allocation_environment_and_claim_before_reads(donor, monkeypatch):
    spec = create(donor)
    study = c.validate_stage(spec)
    with pytest.raises(FileNotFoundError):
        with w.stream(spec, study, 'jt_gate', limit=64) as source: list(source)
    assert not donor['reads']
    monkeypatch.setattr(w, 'numerical_environment', lambda: {})
    with pytest.raises(ValueError, match='environment'): w.run(spec, 'jt_gate')
    assert not donor['reads'] and not (dev.stage_dir(spec)/'claims').exists()
    for limit in (None, 65):
        with pytest.raises(PermissionError):
            with w.stream(spec, study, 'jt_gate', limit=limit) as source: list(source)


def test_real_serial_trace_spawn_gate_and_prefix(donor):
    spec = create(donor)
    dev_worker.run(spec, 'jt_gate')
    row = w.read(spec)
    assert row['compatible'] and row['within_site_exact'] and row['cross_site_jets'] == 64
    assert all(v['exact_bytes'] for v in row['comparisons'])
    assert donor['reads'] == [64] and donor['calls'] == ['full_ancestry']
    prefix = w.prefix_digest(row['serial'], spec['root'])
    assert all(prefix[k] == row['serial'][k] for k in prefix)
    with pytest.raises(FileExistsError): w.run(spec, 'jt_gate')


def test_10k_screen_flow_checks_all_runs_without_cross_site_overclaim(donor, monkeypatch):
    # Keep source/claim/membership/reader/serialization/receipt/report checks
    # real, replacing just 60k expensive response evaluations with identity.
    # The preceding test exercises the actual frozen Generator and spawn pool.
    def parallel(source, model, workers, chunk, trace=False):
        for pairs in engine.chunks(source, chunk):
            rows = [(pair.identity, pair.offline, 'a'*64) for pair in pairs]
            yield dict(rows=rows, cpu_seconds=.01, worker_wall_seconds=.01, flags={})
    monkeypatch.setattr(engine, 'parallel', parallel)
    # Replace donor reference with this deterministic fixture kernel, retaining
    # real publication, content hashes, gate contract and output receipts.
    old = Path(donor['original']['root'])
    ref = engine.process(iter(donor['pairs'][:64]), donor['bundle'], root=old, relative='fixture_reference',
                         workers=1, chunk=8, compression='stored', expected_count=64)
    fields = dict(jets=64, parity=True, serial=ref, replay_runs=[ref]*3,
                  input_read_seconds=.01, measurement={'sampled_peak_tree_rss_bytes': 1000000})
    gate = artifact('GEN_GATE', parents=gen.parents(donor['parent']), projection=gen.projection(fields), **fields)
    report_path = old/'reports/generation_gate_r1/jg_gate.json'
    report_path.write_text(json.dumps(gate))
    receipt_path = dev.stage_dir(donor['parent'])/'receipts/jg_gate.json'
    receipt = load_json(receipt_path)
    receipt['outputs']['result']['sha256'] = sha256_file(report_path)
    receipt['outputs']['block'] = dict(relative=ref['blocks'][0]['relative'], sha256=ref['blocks'][0]['sha256'])
    receipt_path.write_text(json.dumps(with_content_hash(receipt)))
    spec = create(donor)
    w.run(spec, 'jt_gate')
    screen = c.advance(dev.stage_dir(spec)/'stage_spec.json')
    with pytest.raises(FileNotFoundError): w.read(screen)
    for task in screen['tasks']:
        w.run(screen, task['task_id'])
    report = w.read(screen)
    assert [r['workers'] for r in report['settings']] == [16, 36, 72]
    assert report['cross_site_jets'] == 64 and not report['full_run_cross_site_compared']
    assert report['raw_root_ingestion_measured'] and not report['production_qualified']
    assert donor['reads'] == [64]+[None]*6
    assert donor['calls'] == ['full_ancestry']
    assert not (dev.stage_dir(screen)/'submission_ledger.json').exists()
    with pytest.raises(PermissionError, match='production'): c.advance(dev.stage_dir(screen)/'stage_spec.json')
    # Removing a single output prevents results, rather than selecting survivors.
    output = dev.product(screen, 'jt_c072_r1', 'result')['run']['blocks'][0]
    (Path(screen['root'])/output['relative']).write_bytes(b'bad')
    with pytest.raises(ValueError, match='Corrupt'): w.read(screen)


def test_submission_exact_review_idempotence_ambiguity(donor, monkeypatch):
    spec = create(donor)
    monkeypatch.setattr(submission, 'site_checks', lambda *a: None)
    calls = []
    def scheduler(argv):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout='12345\n', stderr='')
    monkeypatch.setattr(submission, 'scheduler', scheduler)
    plan = submission.submit(spec)
    assert not calls
    with pytest.raises(PermissionError): submission.submit(spec, execute=True)
    for _ in range(2):
        submission.submit(spec, execute=True, reviewed_plan_hash=plan['content_hash'],
                          authorization_phrase=dev.PHRASES[spec['stage']])
    assert len(calls) == 1 and '--partition=tigris' in calls[0]
    (submission.journal(spec, 'jt_gate')/'receipt.json').unlink()
    with pytest.raises(RuntimeError, match='Unacknowledged'):
        submission.submit(spec, execute=True, reviewed_plan_hash=plan['content_hash'],
                          authorization_phrase=dev.PHRASES[spec['stage']])
    assert len(calls) == 1


@pytest.mark.parametrize('status', [0, 7])
def test_helper_dry_default_syntax_and_heartbeat_exit(status):
    root = Path(__file__).parents[1]
    bash = Path(os.environ.get('ProgramFiles', 'C:/Program Files'))/'Git/bin/bash.exe'
    if not bash.exists(): pytest.skip('Bash unavailable')
    helper = root/'scripts/queue_cms2jc2_direct_tigris.sh'
    result = subprocess.run([str(bash), '-n', str(helper)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    text = helper.read_text()
    assert 'EXECUTE=0' in text and '--reviewed-plan-hash' in text and 'scancel' not in text
    assert 'atlas_kd_tigris' in text and 'atlas_kd_sporc' not in text
    function = 'phase() {'+text.split('phase() {', 1)[1].split('\nif [ ! -e', 1)[0]
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
