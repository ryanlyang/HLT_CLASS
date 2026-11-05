"""Separate source-bound controls, authenticated against the completed parent setup."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_fullsim.regular_submission import clean_environment
from hlt_classification.luka_u100_pair_hydra import inputs, site
from hlt_classification.luka_cross_message_hydra.execution import A6000_SITE
from hlt_classification.luka_u100_embedding.contracts import COUNTS
from .contracts import BASE, TASKS, artifact, validate, recipe, matched_task

MODULES = ('__init__','contracts','model','training','campaign','worker','submission','reporting')
RUNTIME = {'src/hlt_classification/luka_cross_message_ce_hydra/'+n+'.py' for n in MODULES} | {
    'scripts/luka_cross_message_ce_hydra.py', 'scripts/queue_luka_cross_message_ce_hydra.sh',
    'sbatch/run_luka_cross_message_ce_hydra.sh'}
REQUIRED = RUNTIME | {'tests/test_luka_cross_message_ce_hydra.py',
    'docs/plans/LUKA_CROSS_MESSAGE_CE_HYDRA_PLAN.md', 'docs/contracts/LUKA_CROSS_MESSAGE_CE_HYDRA.md',
    'docs/LUKA_CROSS_MESSAGE_CE_HYDRA_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve(); snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git','-C',str(project),*args], check=True, capture_output=True, text=True).stdout.splitlines()
    git('merge-base','--is-ancestor',BASE,commit)
    changes = git('diff','--name-status','--no-renames',BASE,commit,'--','src','scripts','sbatch')
    if (not REQUIRED <= set(git('ls-files'))
            or any(r[:2] != 'A\t' or r[2:] not in RUNTIME for r in changes)
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_cross_message_ce_hydra/campaign.py'):
        raise ValueError('Commit the additive CE-only package; preserve all donor runtime')
    return snapshot


def authenticate_parent(path):
    path = Path(path).resolve(strict=True); parent = load_json(path)
    if (parent.get('source_commit') != BASE or parent.get('gpu_profile') != 'a6000'
            or parent.get('schema_version') != 3 or parent.get('counts') != COUNTS
            or parent.get('execution_site') != A6000_SITE
            or Path(parent['environment_dir']).resolve() != Path(sys.prefix).resolve()):
        raise ValueError('Expected the completed original A6000 v3 parent in the same environment')
    # Authenticate in the original worktree: _source verifies the imported module's
    # location and must never be invoked against another project in this process.
    code = '''import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[2])/"src"))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_cross_message_hydra import campaign,worker,submission
s=load_json(sys.argv[1]); campaign.check(s,check_storage=False)
a=worker.completed_setup(s); r=worker.receipt(s,"setup")
print("CE_PARENT_PROOF="+json.dumps(dict(spec=s["content_hash"],receipt=r["content_hash"],
    job_id=r["job_id"],acceptance=a),sort_keys=True,allow_nan=False))
'''
    env = clean_environment(); env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                                         PYTHONPATH=str(Path(parent['project_dir'])/'src'))
    result = subprocess.run([sys.executable,'-B','-s','-c',code,str(path),parent['project_dir']],
        cwd=parent['project_dir'], env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Original parent authentication failed:\n'+result.stdout+result.stderr)
    rows = [line.removeprefix('CE_PARENT_PROOF=') for line in result.stdout.splitlines() if line.startswith('CE_PARENT_PROOF=')]
    if len(rows) != 1: raise ValueError('Missing unambiguous parent proof')
    proof = json.loads(rows[0]); acceptance = proof['acceptance']; root = Path(parent['root'])
    if (proof['spec'] != parent['content_hash'] or acceptance['parents'] != dict(spec=parent['content_hash'])
            or acceptance['gpu']['name'] != A6000_SITE['gpu_name']
            or acceptance['environment'] != parent['reference']['environment']):
        raise ValueError('Parent proof identity differs')
    evidence = artifact('PARENT_PROOF', parents=dict(spec=parent['content_hash'],
        acceptance=acceptance['content_hash'], receipt=proof['receipt']), setup_job=proof['job_id'],
        acceptance=acceptance, files=[file_ref(p) for p in (
            path,root/'setup/receipt.json',root/'setup/acceptance.json',root/'setup/portability.json')])
    return parent, evidence


def registration(evidence):
    acceptance = evidence['acceptance']
    minutes = {t: acceptance['minutes'][matched_task(t)]+10 for t in TASKS}
    if any(type(n) is not int or not 30 <= n <= 2880 for n in minutes.values()):
        raise ValueError('Measured parent control plus miniature allowance exceeds48h')
    return dict(recipe=recipe(), tasks=list(TASKS), counts=COUNTS, original50k_rows=50000,
        execution_site=A6000_SITE, gpu_profile='a6000', cpus=6, memory_mb=32000,
        minutes=minutes, storage_budget_bytes=2*2**30, fit_max_minutes=2880,
        dependencies='none', standalone_preflight=False, privileged_supervision=False,
        reference_retrained=False, stopping_policy='full30', automatic_retry=False)


def storage(spec):
    root = Path(spec['root']); used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    remaining = spec['storage_budget_bytes']-used
    if remaining <= 0 or shutil.disk_usage(root if root.exists() else root.parent).free < remaining+2**30:
        raise OSError('Need remaining two-GiB output allowance plus one GiB free headroom')


def create(root, parent_path, project, commit, quota_gib):
    if type(quota_gib) is not int or quota_gib < 2: raise ValueError('Reserve at least two GiB for new outputs')
    parent, evidence = authenticate_parent(parent_path)
    root = Path(root).absolute(); project = Path(project).resolve()
    protected = [Path(parent[k]).resolve() for k in ('root','bundle_root','delta_root','encoder_root','project_dir','environment_dir')]+[project]
    if (root.exists() or not root.parent.is_dir() or root.resolve() != root
            or any(p.is_symlink() for p in (root,*root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh separate CE-only campaign root')
    snapshot = current(project, commit); site.environment(parent['reference']['environment'])
    spec = artifact('SPEC', parents=dict(source=snapshot['content_hash'], parent=parent['content_hash'],
        qualification=evidence['content_hash']), **registration(evidence), root=str(root), project_dir=str(project),
        source_commit=commit, source_snapshot=snapshot, environment_dir=str(Path(sys.prefix).resolve()),
        parent_spec=file_ref(Path(parent_path).resolve()), qualification=evidence, available_quota_gib=quota_gib,
        **{k:parent[k] for k in ('bundle_root','reference','imported','anchor')})
    storage(spec); root.mkdir(); write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def check(spec, *, check_storage=True):
    validate_file_ref(spec['parent_spec'])
    parent, evidence = authenticate_parent(spec['parent_spec']['path'])
    validate(spec, 'SPEC', dict(source=spec['source_snapshot']['content_hash'],
        parent=parent['content_hash'], qualification=evidence['content_hash']))
    if (evidence != spec['qualification'] or any(spec.get(k) != v for k,v in registration(evidence).items())
            or any(spec[k] != parent[k] for k in ('bundle_root','reference','imported','anchor'))
            or spec['available_quota_gib'] < 2 or Path(sys.prefix).resolve() != Path(spec['environment_dir'])
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec
            or current(spec['project_dir'],spec['source_commit']) != spec['source_snapshot']):
        raise ValueError('Immutable CE-only registration/source differs')
    site.environment(spec['reference']['environment'])
    if check_storage: storage(spec)
    return parent


def load_hlt(spec, role):
    if role not in ('train','validation'): raise PermissionError('Final-test access is sealed')
    return inputs.load_view(spec, role, 'D000')
