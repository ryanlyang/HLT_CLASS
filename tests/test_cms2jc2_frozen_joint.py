"""Frozen confirmation: isolation, file-level evidence and finite stopping."""
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from hlt_classification.cms2jc2_response import (
    frozen_joint_campaign as campaign, frozen_joint_data as data,
    frozen_joint_worker as worker, frozen_joint_metrics as metrics, frozen_joint_queue as queue,
    dev_campaign as dev, dev_data, dev_submission as submission,
    bdz_joint_campaign as joint, bdz_joint_worker as jw, bdz_joint_metrics as jm,
    bdz_audit_campaign as ac, bdz_audit_worker as aw, bdz_campaign as bdz, bdz_worker as old)
from hlt_classification.cms2jc2_response.contracts import artifact, load_json, with_content_hash, sha256_file
from hlt_classification.cms2jc2_response.dev_diagnostics import fit_ranges, NAMES
from hlt_classification.cms2jc2_response.splits import pack_entries
from hlt_classification.cms2jc2_response.c_diagnostic_worker import historical_replay_equal
from test_cms2jc2_bdz_joint import joint_mapping
from test_cms2jc2_bdz_audit import particles
from test_cms2jc2_bdz_tuning import mapping, completed_bdz
from test_cms2jc2_bounded import bounded_root, LocalMeasurement
from test_cms2jc2_b_tracking import b_root
from test_cms2jc2_frozen_c import frozen_root, free_disk


def fake_metadata(monkeypatch):
    files = []
    for i in range(7):
        files.append(dict(path=f'{i}.root', sha256=f'{i+1:064x}', tree_key='Events;1', raw_entries=11,
            selected_entries=7, entry_mask=pack_entries([0, 2, 3, 4, 6, 8, 10], 11), source='A' if i%2 else 'B',
            original_role='train', response_role='response_fit' if i < 2 else 'response_confirm'))
    inventory = artifact('CMS_INVENTORY', files=files)
    roles = artifact('ROLES', files=files, counts={'response_confirm': 35})
    monkeypatch.setattr(data, 'metadata', lambda study: (inventory, roles))
    monkeypatch.setattr(data, 'validate_inventory', lambda *a: None)
    monkeypatch.setattr(data, 'validate_roles', lambda *a: None)
    monkeypatch.setattr(data, 'MIN_JETS', 30)
    monkeypatch.setattr(data, 'SHARD_JETS', 3)
    return inventory, roles


def test_metadata_membership_all_rows_uneven_file_local_shards(monkeypatch):
    inventory, roles = fake_metadata(monkeypatch)
    value = data.build({})
    assert value['jets'] == 35 and len(value['files']) == 5 and len(value['shards']) == 15
    assert all(s['jets'] <= 3 for s in value['shards'])
    assert [s['jets'] for s in value['shards'][:3]] == [3, 3, 1]
    assert len({s['ordered_identities'] for s in value['shards']}) == 15
    assert all(s['path'] not in ('0.root', '1.root') for s in value['shards'])
    data.validate_membership(value, {})
    changed = deepcopy(value); changed['shards'][0]['start'] = 1
    with pytest.raises(PermissionError): data.validate_membership(with_content_hash(changed), {})
    monkeypatch.setattr(data, 'MIN_JETS', 36)
    with pytest.raises(ValueError, match='all reserved'): data.build({})


@pytest.mark.parametrize('fault', ['role', 'source', 'overlap', 'count'])
def test_membership_refuses_wrong_sources_roles_counts(monkeypatch, fault):
    _, roles = fake_metadata(monkeypatch)
    if fault == 'role': roles['files'][2]['original_role'] = 'test'
    if fault == 'source': roles['files'][2]['source'] = 'C'
    if fault == 'overlap': roles['files'][2]['sha256'] = roles['files'][0]['sha256']
    if fault == 'count': roles['files'][2]['selected_entries'] = 6
    with pytest.raises((ValueError, PermissionError)): data.build({})


def test_no_access_without_separate_lock_and_worker_claim(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign, 'validate_stage', lambda *a, **kw: {})
    spec = dict(stage='frozen_confirm', root=str(tmp_path), name='confirm', content_hash='a'*64,
        untouched_asserted=True, membership=dict(content_hash='b'*64, shards=[{}]),
        reuse=dict(content_hash='c'*64), protocol=dict(content_hash='d'*64))
    opened = []
    monkeypatch.setattr(data, 'authenticated_open', lambda *a: opened.append(a))
    with pytest.raises(FileNotFoundError): next(data.iter_confirmation(spec, {}, shard=0))
    dev.write(tmp_path, 'stages/confirm/confirmation_access.json', data.access_value(spec), 'FROZEN_ACCESS')
    with pytest.raises(FileNotFoundError): next(data.iter_confirmation(spec, {}, shard=0))
    claim = artifact('DEV_CLAIM', parents={'stage': 'a'*64}, task_id='wrong')
    dev.write(tmp_path, 'stages/confirm/claims/jf_eval_0000/claim.json', claim, 'DEV_CLAIM')
    with pytest.raises(PermissionError, match='Exact'): next(data.iter_confirmation(spec, {}, shard=0))
    assert not opened
    with pytest.raises(PermissionError, match='Reader study'):
        next(data.iter_confirmation(spec, {'imported': {'cms_root': 'different'}}, shard=0))
    assert not opened
    with pytest.raises(PermissionError): data.access_value(dict(spec, stage='frozen_gate'))


@pytest.mark.parametrize('fault', ['imported', 'review', 'numerical_environment', 'scientific_source'])
def test_reuse_rejects_changed_frozen_interface_before_input_loading(monkeypatch, fault):
    previous = dict(imported={}, review={}, numerical_environment={},
                    source={'files': {'frozen_scientific.py': 'a'*64}}, root='old')
    study = deepcopy(previous)
    if fault == 'scientific_source': study['source']['files']['frozen_scientific.py'] = 'b'*64
    else: study[fault] = {'changed': True}
    monkeypatch.setattr(joint, 'read', lambda s: {'selected': 'JOINT', 'guard_failures': {'JOINT': []}})
    monkeypatch.setattr(campaign, 'checked_file', lambda ref: ref)
    monkeypatch.setattr(campaign, 'load_json', lambda path: previous)
    with pytest.raises(ValueError, match='Frozen'):
        campaign.reuse({'contract': joint.CONTRACT, 'stage': 'joint_compare', 'study': 'old'}, study)


def test_cpu_lanes_partition_and_no_gpu():
    membership = dict(shards=[{}]*21)
    ts = campaign.tasks('frozen_confirm', membership)
    assert len(ts) == 22 and ts[8]['depends_on'] == ['jf_eval_0000']
    assert ts[20]['depends_on'] == ['jf_eval_0012']
    assert all(t['cpus'] == 16 and t['hours'] == 4 and t['memory_gib'] == 64 for t in ts[:-1])
    assert ts[-1]['depends_on'] == [r['task_id'] for r in ts[:-1]]
    for partition in ('debug', 'tier3'):
        spec = dict(stage='frozen_confirm', name='frozen_confirm_r1', root='root', content_hash='a'*64,
                    contract=campaign.CONTRACT, tasks=ts)
        plan = dev.command_plan(spec, dict(project_dir='project', site=dict(partition=partition, account='reu-aisocial', qos='qos_tier3')))
        assert plan['gpus'] == 0 and plan['cpu_upper_bound'] == 128
        assert all('--partition='+partition in r['argv'] for r in plan['commands'])
        assert all(not any('gpu' in a for a in r['argv']) for r in plan['commands'])


@pytest.mark.parametrize('scenario,choice', [('debug', 'debug'), ('tier3', 'tier3'), ('tie', 'tier3'), ('unknown', None), ('reject', 'debug')])
def test_test_only_partition_probe(monkeypatch, scenario, choice):
    calls = []
    def run(argv):
        calls.append(argv)
        partition = next(a.split('=')[1] for a in argv if a.startswith('--partition='))
        late = (partition != scenario) if scenario in ('debug', 'tier3') else False
        if scenario == 'reject' and partition == 'tier3': return SimpleNamespace(returncode=1, stdout='', stderr='rejected')
        return SimpleNamespace(returncode=0, stdout='' if scenario == 'unknown' else
            f'sbatch: Job 1 to start at 2026-09-{30 if late else 29}T12:00:00 a using 16 processors', stderr='')
    monkeypatch.setattr(queue, 'scheduler', run)
    result = queue.probe()
    assert result['recommended_partition'] == choice and not result['jobs_submitted']
    assert len(calls) == 6 and all('--test-only' in a and '--wrap=true' in a for a in calls)
    assert 'Single-job' in queue.render(result)


def synthetic_result():
    p = particles()
    pairs = [SimpleNamespace(identity=f'j{i}', source_group=f'file{i}', offline=p, hlt=p) for i in range(4)]
    ranges = fit_ranges(pairs, 'a'*64)
    return worker.chunk(pairs, {}, mapping(), joint_mapping(), ranges, gen=lambda *a, **kw: (p, dict(flags={})))


def test_new_kernel_exact_frozen_diagnostics_and_no_p4_change():
    p = particles()
    pairs = [SimpleNamespace(identity=f'j{i}', source_group='f', offline=p, hlt=p) for i in range(3)]
    ranges = fit_ranges(pairs, 'a'*64)
    args = pairs, {}, mapping(), joint_mapping(), ranges
    gen = lambda *a, **kw: (p, dict(flags={}))
    new = worker.chunk(*args, gen=gen)
    previous = jw.chunk(*args, 'evaluate', gen=gen)
    for name in metrics.CANDIDATES:
        assert historical_replay_equal(new['by_file']['f'][name], previous['by_file']['f'][name])
        for replica in range(3):
            side = f'{name}/proxy{replica}'
            assert historical_replay_equal(new['audit']['f'][side], previous['audit']['f'][side])
    for field in ('jet_multiplicity', 'jet_pt_ratio', 'particle_pt'):
        cells = new['by_file']['f']
        assert cells['JOINT']['cells']['proxy0/all']['variables'][field] == cells['B_DZ']['cells']['proxy0/all']['variables'][field]


def test_distribution_plots_use_frozen_bins_with_visible_tail_accounting():
    result = synthetic_result()
    p = particles()
    ranges = fit_ranges([SimpleNamespace(identity='r', offline=p, hlt=p)], 'a'*64)
    figures = list(worker.distributions(metrics.summaries(result['by_file']), ranges))
    assert len(figures) == 4
    assert all(b'<svg' in blob and b'under' in blob and b'over' in blob for _, blob in figures)


def test_checks_width_failure_is_not_hidden_and_bootstrap_is_by_file():
    result = synthetic_result()
    for file, values in result['by_file'].items():
        for name in metrics.CANDIDATES:
            for side in metrics.SIDES:
                cell = values[name]['cells'][side+'/all']['variables']['jet_pt_ratio']
                sd = .05 if side == 'real' else .133
                cell.update(count=100, covered_jets=100, sum=100., sumsq=100*(1+sd**2))
    decision = metrics.assess(result['by_file'], result['audit'])
    check = decision['checks']['sd_ratio/jet_pt_ratio']
    assert check['value'] == pytest.approx(2.66) and check['status'] == 'rejected'
    assert check['contributing_files'] == 4 and check['interval_95'] == pytest.approx([2.66, 2.66])
    assert decision['status'] == 'rejected' and not decision['reselection']
    assert metrics.assess(result['by_file'], result['audit']) == decision
    short = {f: values for f, values in list(result['by_file'].items())[:3]}
    sd = metrics.assess(short, {f: result['audit'][f] for f in short})['checks']['sd_ratio/jet_pt_ratio']
    assert sd['interval_95'] is None and sd['status'] == 'rejected'


@pytest.mark.parametrize('point,ci,groups,status', [
    (.1, [.08,.12],4,'supported'), (.2,[.1,.3],4,'inconclusive'), (.4,[.3,.5],4,'rejected'),
    (.1,None,3,'inconclusive'), (.4,None,4,'inconclusive'), (None,None,4,'inconclusive')])
def test_no_claim_of_support_without_interval(point, ci, groups, status):
    assert metrics._decide(point, ci, [0.,.25], groups) == status


@pytest.mark.parametrize('fault', ['nan', 'time', 'memory', 'size', 'disk'])
def test_measured_resource_gate(fault):
    measured = dict(wall_seconds=10., sampled_peak_tree_rss_bytes=1024)
    if fault == 'nan': measured['wall_seconds'] = float('nan')
    if fault == 'time': measured['wall_seconds'] = 100000
    if fault == 'memory': measured['sampled_peak_tree_rss_bytes'] = 64*1024**3
    members = dict(shards=[dict(jets=5000)], files=[{}], jets=5000)
    if fault == 'disk': members['shards'] *= 100000
    if fault in ('nan', 'size'):
        with pytest.raises(ValueError): campaign.projection(measured, 0 if fault == 'size' else 10000, 64, members)
    else:
        assert not campaign.projection(measured, 10000, 64, members)['resource_envelope_ok']


def test_storage_projection_preserves_publication_headroom():
    measured = dict(wall_seconds=10., sampled_peak_tree_rss_bytes=1024)
    members = dict(shards=[dict(jets=5000)]*4, files=[{}], jets=20000)
    result = campaign.projection(measured, 10000, 64, members)
    assert result['resource_envelope_ok']
    assert result['required_free_bytes'] == 2*result['projected_total_bytes']+5*1024**3
    assert result['projected_report_bytes'] == 4*10000*6


@pytest.mark.parametrize('exit_status', [0, 7])
def test_helper_dry_by_default_and_syntax(exit_status):
    path = Path(__file__).parents[1]/'scripts/queue_cms2jc2_frozen_joint.sh'
    text = path.read_text()
    assert 'EXECUTE=0' in text and '--assert-untouched' in text
    assert 'recommended_partition' in text and 'FROZEN_PARTITION' in text
    assert not any(x in text for x in ('scancel', 'scontrol update', 'git push'))
    bash = Path(os.environ.get('ProgramFiles', 'C:/Program Files'))/'Git/bin/bash.exe'
    if not bash.exists(): pytest.skip('Bash unavailable')
    checked = subprocess.run([str(bash), '-n', str(path)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    function = 'run_phase() {'+text.split('run_phase() {', 1)[1].split('\nif [ ! -e', 1)[0]
    script = 'set -euo pipefail\n'+function+'\nsleep() { command sleep 0.01; }\n'
    script += f'''result=0
run_phase test bash -c 'read -r line; printf "%s\\n" "$line"; exit {exit_status}' <<'INPUT' || result=$?
saved stdin
INPUT
printf 'result=%s\\n' "$result"
'''
    checked = subprocess.run([str(bash), '--noprofile', '--norc', '-s'], input=script,
                             capture_output=True, text=True, timeout=30)
    assert checked.returncode == 0, checked.stderr
    assert 'saved stdin' in checked.stdout and f'result={exit_status}' in checked.stdout


def test_full_confirmation_real_root(completed_bdz, tmp_path, monkeypatch):
    """Tiny ROOT fixtures and synthetic winner; never real qualification evidence."""
    parent, ctx, put, _ = completed_bdz
    for module in (joint, jw, ac, aw): monkeypatch.setattr(module, 'COUNTS', dev_data.COUNTS)
    for module in (old, aw, jw, worker): monkeypatch.setattr(module, 'Measurement', LocalMeasurement)
    for module in (old, jw): monkeypatch.setattr(module, 'plot_pages', lambda *a, **kw: [])
    monkeypatch.setattr(jm, 'figures', lambda *a, **kw: [])
    bz = bdz.create(parent_spec=dev.stage_dir(parent)/'stage_spec.json', project_dir=tmp_path/'bp',
                    source_commit='c'*40, root=tmp_path/'bz')
    put(bz, 'bz_acceptance', dict(result=old.acceptance(ctx, bz)))
    put(bz, 'bz_calibrate', dict(result=old.calibrate(ctx, bz, dict(cpus=1))))
    bc = bdz.advance(dev.stage_dir(bz)/'stage_spec.json')
    for t in bc['tasks'][:4]: put(bc, t['task_id'], dict(result=old.evaluate(ctx, bc, dict(t, cpus=1))))
    put(bc, 'bz_select', dict(result=old.report(ctx, bc)))
    au = ac.create(parent_spec=dev.stage_dir(bc)/'stage_spec.json', project_dir=tmp_path/'ap',
                    source_commit='d'*40, root=tmp_path/'audit')
    put(au, 'ba_acceptance', dict(result=aw.acceptance(ctx, au)))
    for t in au['tasks'][1:5]: put(au, t['task_id'], dict(result=aw.evaluate(ctx, au, dict(t, cpus=1))))
    put(au, 'ba_report', dict(result=aw.report(ctx, au)))
    jg = joint.create(parent_spec=dev.stage_dir(au)/'stage_spec.json', project_dir=tmp_path/'jp',
                      source_commit='e'*40, root=tmp_path/'joint')
    put(jg, 'bj_acceptance', dict(result=jw.acceptance(ctx, jg)))
    put(jg, 'bj_calibrate', dict(result=jw.calibrate(ctx, jg, dict(cpus=1))))
    jc = joint.advance(dev.stage_dir(jg)/'stage_spec.json')
    for t in jc['tasks'][:4]: put(jc, t['task_id'], dict(result=jw.evaluate(ctx, jc, dict(t, cpus=1))))
    selected = jw.report(ctx, jc)
    # Explicit synthetic consistent winner to exercise freezing; old score
    # arithmetic is tested independently, not asserted as a real CMS outcome.
    common = deepcopy(selected['scores']['B_DZ'])
    selected['scores'] = {n: deepcopy(common) for n in jw.maps.CANDIDATES}
    selected['scores']['JOINT']['score'] *= .5
    selected['selected'], selected['guard_failures'] = jm.choose(selected['scores'])
    assert selected['selected'] == 'JOINT'
    put(jc, 'bj_select', dict(result=with_content_hash(selected)))
    preserved = {p: sha256_file(p) for p in Path(tmp_path/'joint').rglob('*') if p.is_file()}
    monkeypatch.setattr(data, 'MIN_JETS', 4)
    monkeypatch.setattr(data, 'SHARD_JETS', 2)
    gate = campaign.create(parent_spec=dev.stage_dir(jc)/'stage_spec.json', project_dir=tmp_path/'fp',
        source_commit='f'*40, root=tmp_path/'frozen', partition='tier3', assert_untouched=True)
    assert gate['membership']['jets'] >= 4
    with pytest.raises(FileNotFoundError): campaign.advance(dev.stage_dir(gate)/'stage_spec.json')
    with pytest.raises(PermissionError): campaign.create(parent_spec=dev.stage_dir(jc)/'stage_spec.json',
        project_dir=tmp_path/'fp', source_commit='f'*40, root=tmp_path/'no', partition='tier3')
    calls = []
    def scheduler(argv):
        calls.append(argv)
        if argv[0] == 'sacct':
            return SimpleNamespace(returncode=0, stdout='\n'.join(f'{j}|COMPLETED|0:0|' for j in argv[argv.index('-j')+1].split(',')), stderr='')
        return SimpleNamespace(returncode=0, stdout='PartitionName=tier3 State=UP' if argv[0] == 'scontrol' else str(9000+len(calls))+'\n', stderr='')
    monkeypatch.setattr(submission, 'scheduler', scheduler)
    plan = submission.submit(gate)
    assert not calls
    with pytest.raises(PermissionError): submission.submit(gate, execute=True)
    submission.submit(gate, execute=True, authorization_phrase=dev.PHRASES[gate['stage']], reviewed_plan_hash=plan['content_hash'])
    accepted = worker.acceptance(ctx, gate)
    put(gate, 'jf_acceptance', dict(result=accepted))
    assert accepted['frozen_worker_replay'] and not accepted['confirmation_accessed']
    confirm = campaign.advance(dev.stage_dir(gate)/'stage_spec.json')
    plan = submission.submit(confirm)
    ledger = submission.submit(confirm, execute=True, authorization_phrase=dev.PHRASES[confirm['stage']], reviewed_plan_hash=plan['content_hash'])
    previous_calls = len(calls)
    assert submission.submit(confirm, execute=True, authorization_phrase=dev.PHRASES[confirm['stage']], reviewed_plan_hash=plan['content_hash']) == ledger
    assert len(calls) == previous_calls
    live = [a for a in calls if a[0] == 'sbatch' and '--test-only' not in a]
    assert len(live) == len(confirm['tasks'])+1
    for t in confirm['tasks'][:-1]:
        claim = artifact('DEV_CLAIM', parents={'stage': confirm['content_hash']}, task_id=t['task_id'], allocation={'synthetic': True})
        dev.write(confirm['root'], f"stages/{confirm['name']}/claims/{t['task_id']}/claim.json", claim, 'DEV_CLAIM')
        row = worker.evaluate(ctx, confirm, dict(t, cpus=2 if t['params']['shard'] == 0 else 1))
        put(confirm, t['task_id'], dict(result=row))
    report = worker.report(ctx, confirm)
    put(confirm, 'jf_report', dict(result=report))
    assert report['decision']['status'] in ('rejected', 'inconclusive')
    assert not report['production_qualified'] and not report['transfer_authorized'] and report['confirmation_accessed']
    assert campaign.read(confirm) == report
    assert 'CMS mean +/- SD' in campaign.render(confirm, statistics=True)
    with pytest.raises(PermissionError, match='Stop after'): campaign.advance(dev.stage_dir(confirm)/'stage_spec.json')
    changed = deepcopy(confirm); changed['tasks'][0]['cpus'] = 36
    with pytest.raises(ValueError, match='registration'): campaign.validate_stage(with_content_hash(changed))
    assert preserved == {p: sha256_file(p) for p in preserved}
    with pytest.raises(PermissionError):
        with dev_data.sample_stream(ctx, 'response_confirm') as stream: next(stream)
