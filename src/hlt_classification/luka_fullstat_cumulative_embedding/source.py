"""Authenticate completed four-teacher evidence in its original pinned source."""
from pathlib import Path
import getpass
import json
import os
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullsim.foundation import _source
from .contracts import artifact, COUNTS

DONOR = 'b21c2d4e01211a31e57d879c5a86ca90aab2ca0b'
BASELINE = '7e8dd105af1ead66b7ec120d7a391ca8310b5401'
MODULES = ('__init__', 'contracts', 'source', 'campaign', 'banks', 'model', 'training', 'worker', 'submission')
ADDITIONS = {f'src/hlt_classification/luka_fullstat_cumulative_embedding/{n}.py' for n in MODULES}
ENTRYPOINTS = {'scripts/luka_fullstat_cumulative_embedding.py',
    'scripts/queue_luka_fullstat_cumulative_embedding.sh', 'sbatch/run_luka_fullstat_cumulative_embedding.sh'}
REQUIRED = ADDITIONS | ENTRYPOINTS | {'tests/test_luka_fullstat_cumulative_embedding.py',
    'docs/plans/LUKA_FULLSTAT_CUMULATIVE_EMBEDDING_PLAN.md',
    'docs/contracts/LUKA_FULLSTAT_CUMULATIVE_EMBEDDING.md',
    'docs/LUKA_FULLSTAT_CUMULATIVE_EMBEDDING_RUNBOOK.md'}

REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_multiteacher import campaign,worker,submission
s=load_json(sys.argv[1]); campaign.check(s); result=worker.results(s)
p=load_json(s['anchor']['spec']['path'])
refs={n:s['anchor']['evidence']['models'][n] for n in ('M0HLT','OFFLINE')}
refs['U100_EMBED_W5']=s['reference']
refs['MULTI4_EMBED_W5']={k:file_ref(Path(s['root'])/'train_multi4'/f) for k,f in
 [('checkpoint','selected.pt'),('report','training_report.json'),
 ('predictions','validation_predictions.npz'),('receipt','task_receipt.json')]}
print('CUMULATIVE_REUSE='+json.dumps(dict(anchor=s['anchor'],references=refs,
 offline_refs=s['teachers']['OFFLINE']['refs'],offline_bank=s['reuse_banks']['OFFLINE'],
 base_node=s['reuse']['base_node'],execution_site=s['execution_site'],
 environment=s['reuse']['environment'],gpu=s['reuse']['gpu'],summaries=s['reuse']['summaries'],
 original_preflight=p['preflight'],original_train_minutes=p['train_minutes'],
 multi_setup=file_ref(Path(s['root'])/'setup/receipt.json'),
 multi_train_minutes=s['train_minutes'],reference_result=result['content_hash'],
 jobs=submission.accepted_jobs(s),protected_roots=s['reuse']['protected_roots']+
 [s['root'],s['project_dir']]),sort_keys=True,allow_nan=False))
'''


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    subprocess.run(['git', '-C', str(project), 'merge-base', '--is-ancestor', commit, 'origin/main'],
                   check=True, capture_output=True, text=True)
    tracked = set(subprocess.run(['git', '-C', str(project), 'ls-files'],
        check=True, capture_output=True, text=True).stdout.splitlines())
    changed = set(subprocess.run(['git', '-C', str(project), 'diff', '--name-only',
        BASELINE, commit, '--', 'src', 'scripts', 'sbatch'],
        check=True, capture_output=True, text=True).stdout.splitlines())
    if (not REQUIRED <= tracked or not changed <= ADDITIONS | ENTRYPOINTS
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_fullstat_cumulative_embedding/source.py'):
        raise ValueError('Commit additive cumulative files without changing existing runtime')
    return snapshot


def authenticate(path, project):
    path = Path(path).resolve(strict=True)
    s = load_json(path)
    original = Path(s['project_dir'])
    if (s['source_commit'] != DONOR or s['counts'] != COUNTS
            or path != Path(s['root'])/'campaign_spec.json'):
        raise ValueError('Expected completed original FullStat four-teacher specification')
    names = subprocess.run(['git', '-C', str(original), 'ls-files', 'src', 'sbatch', 'scripts'],
        check=True, capture_output=True, text=True).stdout.splitlines()
    hashes = {}
    for name in names:
        if not name.endswith(('.py', '.sh')):
            continue
        before, after = original/name, Path(project)/name
        if not after.is_file() or sha256_file(before) != sha256_file(after):
            raise ValueError('Changed donor runtime: '+name)
        hashes[name] = sha256_file(after)
    if not hashes:
        raise ValueError('Empty donor runtime inventory')
    env = dict(os.environ, PYTHONPATH=str(original/'src'), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating completed four-teacher reference and original Oscar acceptance; no new preflight.', flush=True)
    run = subprocess.run([sys.executable, '-B', '-s', '-c', REPLAY, str(path)],
        cwd=original, env=env, capture_output=True, text=True)
    if run.returncode:
        raise ValueError('Original-source authentication failed:\n'+run.stderr)
    lines = [line.split('=', 1)[1] for line in run.stdout.splitlines() if line.startswith('CUMULATIVE_REUSE=')]
    if len(lines) != 1:
        raise ValueError('Missing exact original-source evidence')
    evidence = json.loads(lines[0])
    # A publication alone is not proof that the donor process completed.
    job = evidence['jobs']['train_multi4']
    if not str(job).isdigit() or int(job) <= 0:
        raise ValueError('Invalid donor job ID')
    result = subprocess.run(['sacct', '-j', job, '-X', '-n', '-P', '-o',
        'JobIDRaw,JobName%80,User,Account,State,ExitCode'], check=True, capture_output=True, text=True)
    rows = [line.strip().split('|')[:6] for line in result.stdout.splitlines() if line.strip()]
    expected = [job, 'lukafsm_train_multi4', getpass.getuser(), evidence['execution_site']['account'], 'COMPLETED', '0:0']
    if [r for r in rows if r[0] == job] != [expected]:
        raise ValueError('Donor four-teacher job must be COMPLETED/0:0 with exact identity')
    return artifact('REUSE', parent_spec=file_ref(path), source_commit=DONOR,
        counts=dict(COUNTS), compatible_source=hashes, **evidence)
