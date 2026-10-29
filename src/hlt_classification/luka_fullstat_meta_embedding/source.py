"""Replay completed U100-w5 and its original teacher in immutable source."""
from pathlib import Path
import json
import os
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.jetclass2_delphes.model import installed_environment
from .contracts import artifact, COUNTS
from hlt_classification.luka_fullstat_multiteacher.source import UNRELATED_SPORC

DONOR = '7ea8b0c693734a26ced3820354e047f4af26c979'
BASELINE = '7a6dc268800b0788e0e89d7ec56db37583397946'
RUNTIME = '/oscar/scratch/rlyang/hlt_classification/runtime_repair_v1/atlas_kd_oscar'
PARENT_CONTRACTS = ('LUKA_FULLSTAT_U100_EMBEDDING_SPEC/v1',)
MODULES = ('__init__', 'contracts', 'source', 'campaign', 'model', 'training', 'worker', 'submission', 'execution', 'launch')
ADDITIONS = {f'src/hlt_classification/luka_fullstat_meta_embedding/{n}.py' for n in MODULES}
ENTRYPOINTS = {'scripts/luka_fullstat_meta_embedding.py',
    'scripts/queue_luka_fullstat_meta_embedding.sh', 'sbatch/run_luka_fullstat_meta_embedding.sh',
    'scripts/launch_luka_fullstat_meta_embedding.sh'}
# All parent runtime bytes must match its completed pinned source as well.
REQUIRED = ADDITIONS | ENTRYPOINTS | {'tests/test_luka_fullstat_meta_embedding.py',
    'tests/test_luka_fullstat_meta_embedding_launch.py',
    'docs/plans/LUKA_FULLSTAT_META_EMBEDDING_PLAN.md',
    'docs/contracts/LUKA_FULLSTAT_META_EMBEDDING.md',
    'docs/LUKA_FULLSTAT_META_EMBEDDING_RUNBOOK.md'}

def compatible_changes(changed, added):
    new = ADDITIONS | ENTRYPOINTS
    return changed <= new and added == changed


def runtime(expected):
    if (Path(sys.prefix).resolve() != Path(RUNTIME).resolve(strict=True)
            or os.environ.get('CONDA_PREFIX') != RUNTIME):
        raise PermissionError('Use the verified runtime_repair_v1 environment')
    if installed_environment() != expected:
        raise ValueError('Replacement environment fingerprint differs')


REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_fullstat_u100_embedding import campaign,worker,submission
from hlt_classification.luka_fullsim.science_campaign import preflight_accounting
s=load_json(sys.argv[1]); campaign.check(s); result=worker.results(s)
jobs=submission.accepted_jobs(s); preflight_accounting(jobs['train_w5'])
p=load_json(s['anchor']['spec']['path'])
teacher=next(n for n in p['nodes'] if n['node_id']=='COARSE_U100')
if teacher['coordinate']!='U100' or teacher['context_coordinate'] is not None:
    raise ValueError('Expected original regular COARSE_U100')
reference={k:file_ref(Path(s['root'])/'train_w5'/f) for k,f in
    [('checkpoint','selected.pt'),('report','training_report.json'),
     ('predictions','validation_predictions.npz'),('receipt','task_receipt.json')]}
bank=file_ref(Path(s['root'])/'targets/manifest.json')
m=load_json(bank['path']); validate_file_ref(m['payload'],root=Path(bank['path']).parent)
e=dict(anchor=s['anchor'],reference=reference,reference_result=result['content_hash'],
    bank_manifest=bank,bank_hash=m['content_hash'],teacher_refs=s['teacher_refs'],teacher_node=teacher,
    base_node=s['node'],execution_site=s['execution_site'],environment=s['reuse']['environment'],
    gpu=s['reuse']['gpu'],summaries=s['reuse']['summaries'],jobs=jobs,
    protected_roots=[s['root'],s['project_dir'],p['root'],p['project_dir'],
        s['reuse']['old_root'],s['reuse']['old_project']])
print('FULLSTAT_META_REUSE='+json.dumps(e,sort_keys=True,allow_nan=False))
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
    added = set(subprocess.run(['git', '-C', str(project), 'diff', '--name-only', '--diff-filter=A',
        BASELINE, commit, '--', 'src', 'scripts', 'sbatch'],
        check=True, capture_output=True, text=True).stdout.splitlines())
    if (not REQUIRED <= tracked or not compatible_changes(changed, added)
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_fullstat_meta_embedding/source.py'):
        raise ValueError('Source changes outside additive FullStat meta-embedding namespace')
    return snapshot


def authenticate(path, project):
    path = Path(path).resolve(strict=True)
    s = load_json(path)
    original = Path(s['project_dir'])
    if (s['counts'] != COUNTS or s['contract'] not in PARENT_CONTRACTS
            or s['source_commit'] != DONOR
            or path != Path(s['root'])/'campaign_spec.json'):
        raise ValueError('Expected completed original FullStat U100-w5 specification')
    runtime(s['reuse']['environment'])
    subprocess.run(['git', '-C', str(original), 'merge-base', '--is-ancestor', DONOR, s['source_commit']],
                   check=True, capture_output=True, text=True)
    names = subprocess.run(['git', '-C', str(original), 'ls-files', 'src', 'sbatch', 'scripts'],
        check=True, capture_output=True, text=True).stdout.splitlines()
    hashes = {}
    for name in names:
        if not name.endswith(('.py', '.sh')) or name in UNRELATED_SPORC:
            continue
        before, after = original/name, Path(project)/name
        if not after.is_file() or sha256_file(before) != sha256_file(after):
            raise ValueError('Changed donor runtime: '+name)
        hashes[name] = sha256_file(after)
    if not hashes:
        raise ValueError('Empty donor runtime inventory')
    env = dict(os.environ, PYTHONPATH=str(original/'src'), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating completed U100-w5 and its original COARSE_U100 targets; no separate preflight.', flush=True)
    run = subprocess.run([sys.executable, '-B', '-s', '-c', REPLAY, str(path)],
        cwd=original, env=env, capture_output=True, text=True)
    if run.returncode:
        raise ValueError('Original-source authentication failed:\n'+run.stderr)
    lines = [line.split('=', 1)[1] for line in run.stdout.splitlines() if line.startswith('FULLSTAT_META_REUSE=')]
    if len(lines) != 1:
        raise ValueError('Missing exact original-source evidence')
    evidence = json.loads(lines[0])
    # Original worker validates the actual bank, fit, references and exact job bindings.
    return artifact('REUSE', parent_spec=file_ref(path), source_commit=s['source_commit'],
        counts=dict(COUNTS), compatible_source=hashes, **evidence)
