"""JSON-order regression, saved-fit recovery, exact DAG and fail-closed reuse."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import ast
import os
import random
import re
import shutil
import subprocess
import sys
import numpy as np
import pytest
import torch
from test_luka_embedding_aux import tiny, threads
from test_luka_fullstat_splithead import fixture
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_fullstat.recipe import learning_rate
from hlt_classification.luka_fullstat_splithead import contracts as c, training, worker, banks, submission, campaign
from hlt_classification.luka_fullstat_splithead import recovery as r, recovery_source as rs


def report_case(stage):
    """Real validator with valid positive losses that expose sorted-sum drift."""
    rec = c.recipe(stage); rng = random.Random(42)
    metrics = evaluate_probabilities(np.arange(22) % 11, np.full((22, 11), 1/11, np.float32))
    history = []
    for epoch in range(1, 21):
        kd = {t: rng.uniform(.001, .02) for t in c.teachers(stage)}
        embed = {t: rng.uniform(.01, .1) for t in c.teachers(stage)}
        row = dict(pass_number=epoch, update=epoch, learning_rate=learning_rate(epoch),
            validation=metrics, ce_loss=.36, weighted_ce_loss=.18, logit_losses=kd, embedding_losses=embed,
            weighted_logit_losses={t: rec['logit_weights'][t]*kd[t] for t in kd},
            weighted_embedding_losses={t: rec['weights'][t]*embed[t] for t in embed},
            ce_coefficient=.5, logit_total=.5, embedding_total=5., train_seconds=1., validation_seconds=1.)
        row['total_loss'] = row['weighted_ce_loss']+sum(row['weighted_logit_losses'].values())+sum(row['weighted_embedding_losses'].values())
        history.append(row)
    node = dict(node_id=stage)
    spec = dict(nodes={stage: node}, counts=dict(train=22, validation=22), anchor=dict(evidence=dict(prepared='p')))
    report = c.artifact('TRAINING_REPORT', parents=dict(recipe=rec['content_hash'], prepared='p', bank='b'),
        node=node, scientific_fit=True, technical=False, passes=20, validation_history=history,
        selected_pass=1, validation=metrics, selected_state='native_backbone_only',
        selected_weights_restored=True, initialization=rec['initialization'])
    return spec, report


@pytest.mark.parametrize('stage', c.STAGES)
def test_exact_loss_validation_survives_sorted_json_and_reversed_mappings(stage):
    spec, report = report_case(stage)
    training.validate_report(report, spec, stage, 'b')
    restored = json.loads(json.dumps(report, sort_keys=True))
    assert restored == report
    assert any(x['total_loss'] != x['weighted_ce_loss']+sum(x['weighted_logit_losses'].values())+
        sum(x['weighted_embedding_losses'].values()) for x in restored['validation_history'])
    training.validate_report(restored, spec, stage, 'b')
    for row in restored['validation_history']:
        for key in ('weighted_embedding_losses', 'weighted_logit_losses'):
            row[key] = dict(reversed(list(row[key].items())))
    training.validate_report(restored, spec, stage, 'b')
    # Even a one-ULP tampering remains rejected: no broad isclose tolerance.
    restored['validation_history'][0]['total_loss'] = float(np.nextafter(restored['validation_history'][0]['total_loss'], np.inf))
    with pytest.raises(ValueError, match='Epoch loss'):
        training.validate_report(with_content_hash(restored), spec, stage, 'b')


def saved_fit(path, monkeypatch, device='cpu'):
    old, caches = fixture(path, export=True)
    root = Path(old['root'])
    monkeypatch.setattr(worker, 'memory_snapshot', lambda _: dict(peak_rss_bytes=2**30, gpu_peak_bytes=2**28))
    monkeypatch.setattr(r, 'memory_snapshot', worker.memory_snapshot)
    worker.export_target(old, 'M0HLT', 'kd', caches['train']['D000'], device=device)
    tr, val = caches['train']['D066'], caches['validation']['D066']
    bank = banks.publish(old, 'DE066', tr)
    attempt = root/'train_DE066_attempt'; attempt.mkdir()
    start = worker.startup(old, 'DE066', tr, val, bank, device=device)
    write_immutable_json(attempt/'startup.json', start)
    real_train = training.train
    def full_population_roundoff(*args, **kwargs):
        report, state = real_train(*args, **kwargs)
        # Fault injection: tiny one-batch FP32 means often sum exactly. Supply
        # valid FP64 epoch means exhibiting the production JSON-order failure.
        # Actual model training, selected weights, probabilities and metrics stay
        # real and unchanged; only diagnostic loss telemetry is synthesized.
        _, case = report_case('DE066')
        rows = json.loads(json.dumps(case, sort_keys=True))['validation_history']
        ix = next(i for i, x in enumerate(rows) if x['total_loss'] != x['weighted_ce_loss']+
            sum(x['weighted_logit_losses'].values())+sum(x['weighted_embedding_losses'].values()))
        for k in ('ce_loss','weighted_ce_loss','logit_losses','embedding_losses',
                  'weighted_logit_losses','weighted_embedding_losses','total_loss'):
            report['validation_history'][0][k] = deepcopy(case['validation_history'][ix][k])
        return with_content_hash(report), state
    with monkeypatch.context() as patch:
        patch.setattr(training, 'train', full_population_roundoff)
        worker.fit(root/'train_DE066', old, 'DE066', tr, val, bank, device=device)
    jobs = dict(train_DE066=r.FAILED_JOB, train_DE033='7910170', train_DE000='7910171')
    states = {task: [job, 'lukafss_'+task, 'rlyang', old['execution_site']['account'],
        'FAILED' if task=='train_DE066' else 'CANCELLED', '1:0' if task=='train_DE066' else '0:0',
        '07:32:01' if task=='train_DE066' else '00:00:00'] for task, job in jobs.items()}
    monkeypatch.setattr(r, 'accounting', lambda s, j: states)
    monkeypatch.setattr(rs, 'original', lambda p: (old, jobs))
    monkeypatch.setattr(rs, 'current', lambda *a: dict(content_hash='a'*64))
    # Upstream synthetic fixture uses /parent rather than a real old Oscar tree.
    monkeypatch.setattr(campaign, 'protected', lambda *a: None)
    # This is a synthetic-population miniature, not a claim of full-data acceptance.
    real_policy = r.policy
    monkeypatch.setattr(r, 'policy', lambda original, evidence:
        dict(real_policy(original, evidence), counts=old['counts']))
    monkeypatch.setattr(worker, 'validation_arrays', lambda _: dict(identities=val.identities, labels=val.labels))
    monkeypatch.setattr(worker, 'metadata', lambda _: (dict(content_hash=tr.foundation_sha256,
        summaries=old['reuse']['summaries']), None, None))
    monkeypatch.setattr(worker, 'data', lambda spec, role, coordinate, **kw: caches[role][coordinate])
    monkeypatch.setattr(worker, 'execution', lambda *a: os.environ['SLURM_JOB_ID'])
    return old, jobs, caches


@pytest.fixture
def failed(tmp_path, monkeypatch):
    tiny(monkeypatch)
    return saved_fit(tmp_path, monkeypatch)


def create_recovery(tmp_path, old):
    return r.create(tmp_path/'recover', Path(old['root'])/'campaign_spec.json',
        tmp_path/'newproject', 'a'*40, quota_gib=16)


def test_audit_and_spec_retain_old_fit_no_preflight_no_retraining(failed, tmp_path):
    old, jobs, _ = failed
    before = {p: file_ref(p) for p in Path(old['root']).rglob('*') if p.is_file()}
    s = create_recovery(tmp_path, old)
    assert r.check(s) == old
    assert s['new_scientific_fits'] == 2 and s['reused_scientific_fits'] == 1
    assert s['recipes'] == old['recipes'] and s['nodes'] == old['nodes']
    assert s['counts'] == old['counts'] and s['preflight_jobs'] == 0
    assert s['external_refs']['DE066']['checkpoint']['path'] == str(Path(old['root'])/'train_DE066/selected.pt')
    assert s['recovery']['evidence']['reordered_epochs']
    assert before == {p: file_ref(p) for p in before}
    assert not (Path(old['root'])/'recovery_claim_v1').exists()
    assert worker.completed(s, 'DE066') is None


@pytest.mark.parametrize('relative', ['train_DE066/selected.pt', 'train_DE066/training_report.json',
    'train_DE066/validation_predictions.npz', 'train_DE066/receipt.json',
    'bundles/DE066.json', 'train_DE066_attempt/startup.json', 'logit_targets/M0HLT/probabilities.npz'])
def test_corrupt_saved_evidence_is_not_reusable(failed, relative):
    old, jobs, _ = failed
    p = Path(old['root'])/relative
    p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        r.audit(old, jobs)


@pytest.mark.parametrize('relative', ['targets/DE066', 'logit_targets/DE066', 'train_DE033',
    'train_DE033_attempt', 'train_DE000', 'train_DE000_attempt', 'train_DE066/task_receipt.json'])
def test_partial_or_started_followup_blocks_recovery(failed, relative):
    old, jobs, _ = failed
    (Path(old['root'])/relative).mkdir(parents=True)
    with pytest.raises(ValueError, match='Unexpected'):
        r.audit(old, jobs)


def test_exact_recovery_dag_and_cross_root_duplicate_claim(failed, tmp_path, monkeypatch):
    old, _, _ = failed
    s = create_recovery(tmp_path, old)
    monkeypatch.setattr(submission, 'check_site', lambda _: None)
    calls = []
    def fake(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=str(8100000+len(calls))+'\n', stderr='')
    monkeypatch.setattr(subprocess, 'run', fake)
    plan = submission.submit(s)
    assert plan['scientific_fits']==2 and plan['preflight_jobs']==0
    assert [x['task_id'] for x in plan['commands']] == ['recover_DE066', 'train_DE033', 'train_DE000']
    with pytest.raises(PermissionError):
        submission.submit(s, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    submission.submit(s, execute=True, plan_hash=plan['content_hash'], authorization=r.AUTHORIZATION)
    assert len(calls)==3
    assert '--job-name=lukafssr_recover_DE066' in calls[0]
    assert '--dependency=afterok:8100001' in calls[1]
    assert '--dependency=afterok:8100001:8100002' in calls[2]
    assert all('7910169' not in ' '.join(x) for x in calls)
    r.verify_claim(s)
    with pytest.raises(FileExistsError):
        submission.submit(s, execute=True, plan_hash=plan['content_hash'], authorization=r.AUTHORIZATION)
    with pytest.raises(FileExistsError):
        r.claim(dict(s, root=str(tmp_path/'different')), plan)


@pytest.mark.parametrize('change', ['owner', 'account', 'name', 'running', 'started', 'exit', 'duplicate', 'absent'])
def test_scheduler_evidence_fails_closed(monkeypatch, change):
    old = dict(execution_site=dict(account='default'))
    jobs = dict(train_DE066=r.FAILED_JOB, train_DE033='2', train_DE000='3')
    rows = [[r.FAILED_JOB,'lukafss_train_DE066','rlyang','default','FAILED','1:0','07:32:01'],
        ['2','lukafss_train_DE033','rlyang','default','CANCELLED','0:0','00:00:00'],
        ['3','lukafss_train_DE000','rlyang','default','CANCELLED','0:0','00:00:00']]
    if change in ('owner','account','name'):
        rows[1][dict(owner=2,account=3,name=1)[change]]='wrong'
    elif change=='running': rows[1][4]='RUNNING'
    elif change=='started': rows[1][6]='00:00:01'
    elif change=='exit': rows[0][5]='2:0'
    elif change=='duplicate': rows.append(rows[0])
    elif change=='absent': rows.pop()
    monkeypatch.setattr(r.getpass,'getuser',lambda:'rlyang')
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='\n'.join('|'.join(x) for x in rows)))
    with pytest.raises(ValueError): r.accounting(old,jobs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual installed-Weaver GPU required')
def test_real_native_export_only_recovery_then_two_full20_workers(tmp_path, monkeypatch):
    old, jobs, caches = saved_fit(tmp_path, monkeypatch, device='cuda')
    originals = {p: file_ref(p) for p in Path(old['root']).rglob('*') if p.is_file()}
    s = create_recovery(tmp_path, old)
    r.claim(s, submission.render(s))
    ids = dict(recover_DE066='8001',train_DE033='8002',train_DE000='8003')
    monkeypatch.setattr(submission,'accepted_jobs',lambda _: ids)
    monkeypatch.setattr(submission,'successful',lambda *a: None)
    original_fit, original_train = worker.fit, training.train
    def forbidden(*a, **kw): pytest.fail('Recovery MUST NOT train any model')
    monkeypatch.setattr(worker,'fit',forbidden)
    monkeypatch.setattr(training,'train',forbidden)
    monkeypatch.setenv('SLURM_JOB_ID','8001')
    done = worker.run(s,'recover_DE066')
    assert done == worker.completed(s,'DE066')
    assert done['training_repeated'] is False
    assert not (Path(s['root'])/'train_DE066/selected.pt').exists()
    assert originals == {p: file_ref(p) for p in originals}
    with pytest.raises(FileExistsError): worker.run(s,'recover_DE066')
    monkeypatch.setattr(worker,'fit',original_fit)
    monkeypatch.setattr(training,'train',original_train)
    for stage in ('DE033','DE000'):
        monkeypatch.setenv('SLURM_JOB_ID',ids['train_'+stage])
        worker.run(s,'train_'+stage)
        assert worker.completed(s,stage)
    assert worker.results(s)['pending']==[]
    assert originals == {p: file_ref(p) for p in originals}
    p=Path(s['root'])/'targets/DE066/targets.npz'
    p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): worker.completed(s,'DE000')


def test_source_boundary(monkeypatch, tmp_path):
    assert rs.compatible(rs.MODIFIED | rs.ADDED, rs.ADDED)
    assert not rs.compatible(rs.MODIFIED | {'src/changed_model.py'}, set())
    assert not rs.compatible(rs.ADDED, set())
    compile(rs.REPLAY,'replay','exec')
    project = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(rs,'_source',lambda *a:dict(content_hash='s'))
    def git(args, **kw):
        return SimpleNamespace(stdout='\n'.join(rs.REQUIRED if 'ls-files' in args else
            rs.ADDED if '--diff-filter=A' in args else rs.MODIFIED | rs.ADDED))
    monkeypatch.setattr(subprocess,'run',git)
    assert rs.current(project,'a'*40)==dict(content_hash='s')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='src/unreviewed.py'))
    with pytest.raises(ValueError, match='reviewed repair boundary'):
        rs.current(project, 'a'*40)


@pytest.mark.parametrize('failure', ['subprocess', 'missing', 'duplicate', 'changed_spec', 'changed_jobs'])
def test_original_pinned_replay_fails_closed(tmp_path, monkeypatch, failure):
    project = tmp_path/'oldsource'; project.mkdir()
    old = c.artifact('SPEC', root=str(tmp_path), project_dir=str(project),
        source_commit=rs.FAILED_COMMIT, counts=dict(c.COUNTS))
    path = tmp_path/'campaign_spec.json'; write_immutable_json(path, old)
    replay = dict(spec=deepcopy(old), jobs=dict(train_DE066=r.FAILED_JOB, train_DE033='2', train_DE000='3'))
    if failure == 'changed_spec': replay['spec']['root'] = 'wrong'
    if failure == 'changed_jobs': replay['jobs'].pop('train_DE000')
    line = 'FAILED_SPLITHEAD='+json.dumps(replay)+'\n'
    def run(args, **kwargs):
        assert kwargs['cwd'] == project.resolve()
        assert kwargs['env']['PYTHONPATH'] == str(project/'src')
        assert kwargs['env']['PYTHONNOUSERSITE'] == '1'
        assert kwargs['env']['PYTHONDONTWRITEBYTECODE'] == '1'
        assert args[-1] == str(path.resolve())
        return SimpleNamespace(returncode=1 if failure == 'subprocess' else 0,
            stdout='' if failure == 'missing' else line*2 if failure == 'duplicate' else line,
            stderr='synthetic replay failure')
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(ValueError): rs.original(path)


@pytest.mark.parametrize('task', ['recover_DE066', 'train_DE033', 'train_DE000'])
def test_recovery_ancestor_requires_successful_exact_accounting(monkeypatch, task):
    spec = dict(contract=r.RECOVERY_CONTRACT, execution_site=dict(account='default'))
    monkeypatch.setattr(submission, 'accepted_jobs', lambda _: {task: '8001'})
    monkeypatch.setattr(submission.getpass, 'getuser', lambda: 'rlyang')
    row = ['8001', 'lukafssr_'+task, 'rlyang', 'default', 'COMPLETED', '0:0']
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='|'.join(row)))
    submission.successful(spec, task, '8001')
    for index, bad in [(1, 'lukafss_'+task), (2, 'other'), (3, 'other'), (4, 'RUNNING'), (5, '1:0')]:
        good = row[index]; row[index] = bad
        with pytest.raises(ValueError): submission.successful(spec, task, '8001')
        row[index] = good
    with pytest.raises(ValueError): submission.successful(spec, task, '8002')


def test_original_execution_roots_are_protected(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'protected', lambda *a: None)
    old = dict(root=str(tmp_path/'old'), project_dir=str(tmp_path/'oldsource'), reuse={})
    for target in (tmp_path, tmp_path/'old', tmp_path/'old'/'child', tmp_path/'oldsource'):
        with pytest.raises(PermissionError): r.protected(target, old, tmp_path/'newsource')
    r.protected(tmp_path/'safe', old, tmp_path/'newsource')


def test_training_objective_and_model_are_byte_semantically_unchanged():
    project=Path(__file__).resolve().parents[1]
    prefix='src/hlt_classification/luka_fullstat_splithead/'
    for name in ('training.py','contracts.py','model.py'):
        old=subprocess.run(['git','show',rs.FAILED_COMMIT+':'+prefix+name],cwd=project,
            check=True,capture_output=True,text=True).stdout
        current=(project/prefix/name).read_text()
        if name!='training.py':
            assert ast.dump(ast.parse(old))==ast.dump(ast.parse(current))
        else:
            before={n.name:ast.dump(n) for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)}
            after={n.name:ast.dump(n) for n in ast.parse(current).body if isinstance(n,ast.FunctionDef)}
            for function in ('objective','optimizer','batch_loss','finite_gradients','train'):
                assert before[function]==after[function]


def test_wrappers_docs_and_cli():
    project=Path(__file__).resolve().parents[1]
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if not bash.is_file(): bash=shutil.which('bash')
    if not bash: pytest.skip('Bash unavailable')
    for file in ('scripts/queue_luka_fullstat_splithead_recovery.sh','sbatch/run_luka_fullstat_splithead_recovery.sh'):
        subprocess.run([str(bash),'-n',file],cwd=project,check=True,capture_output=True)
    text=(project/'docs/LUKA_FULLSTAT_SPLITHEAD_RECOVERY_RUNBOOK.md').read_text()
    for block in re.findall(r'^```bash\n(.*?)^```',text,re.M|re.S):
        subprocess.run([str(bash),'-n'],input=block,text=True,check=True,capture_output=True)
    subprocess.run([sys.executable,'-B','scripts/luka_fullstat_splithead_recovery.py','--help'],
        cwd=project,check=True,capture_output=True)
    for file in rs.REQUIRED:
        text=(project/file).read_text()
        assert all(line==line.rstrip() for line in text.splitlines()),file
        if file.endswith('.md'):
            for link in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)',text):
                assert (project/file).parent.joinpath(link).is_file(),(file,link)
