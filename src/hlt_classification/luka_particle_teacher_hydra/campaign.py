"""Additive source and original-worktree authentication; fresh immutable roots."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_fullsim.regular_submission import clean_environment
from hlt_classification.luka_cross_message_ce_hydra.campaign import authenticate_parent
from hlt_classification.luka_cross_message_hydra.execution import A6000_SITE
from hlt_classification.luka_u100_pair_hydra import site
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle
from hlt_classification.luka_u100_embedding.contracts import COUNTS
from .contracts import BASE,TASKS,recipe,artifact,validate

MODULES=('__init__','contracts','model','training','campaign','inputs','resources','submission','worker','reporting')
RUNTIME={'src/hlt_classification/luka_particle_teacher_hydra/'+n+'.py' for n in MODULES}|{
    'scripts/luka_particle_teacher_hydra.py','scripts/queue_luka_particle_teacher_hydra.sh','sbatch/run_luka_particle_teacher_hydra.sh'}
REQUIRED=RUNTIME|{'tests/test_luka_particle_teacher_hydra.py','docs/plans/LUKA_PARTICLE_TEACHER_HYDRA_PLAN.md',
    'docs/contracts/LUKA_PARTICLE_TEACHER_HYDRA.md','docs/LUKA_PARTICLE_TEACHER_HYDRA_RUNBOOK.md',
    'docs/LUKA_PARTICLE_TEACHER_HYDRA_VALIDATION.md'}


def current(project,commit):
    project=Path(project).resolve(); snapshot=_source(project,commit)
    def git(*args):
        return subprocess.run(['git','-C',str(project),*args],check=True,capture_output=True,text=True).stdout.splitlines()
    git('merge-base','--is-ancestor',BASE,commit)
    changes=git('diff','--name-status','--no-renames',BASE,commit,'--','src','scripts','sbatch')
    if (not REQUIRED<=set(git('ls-files')) or any(r[:2]!='A\t' or r[2:] not in RUNTIME for r in changes)
            or Path(__file__).resolve()!=project/'src/hlt_classification/luka_particle_teacher_hydra/campaign.py'):
        raise ValueError('Commit only additive particle-teacher runtime; preserve all donor implementations')
    return snapshot


def teacher_proof(parent_path,parent):
    code='''import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[2])/"src"))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_cross_message_hydra import worker,training
s=load_json(sys.argv[1]); r=worker.receipt(s,"teacher_m16")
report,_=training.validate_fit(Path(s["root"])/"teacher_m16/fit",s,"teacher_m16")
print("PARTICLE_TEACHER_PROOF="+json.dumps(dict(receipt=r["content_hash"],job_id=r["job_id"],report=report["content_hash"])))
'''
    env=clean_environment(); env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(Path(parent['project_dir'])/'src'))
    result=subprocess.run([sys.executable,'-B','-s','-c',code,str(parent_path),parent['project_dir']],
        cwd=parent['project_dir'],env=env,capture_output=True,text=True)
    if result.returncode: raise RuntimeError('Original m16 teacher authentication failed:\n'+result.stdout+result.stderr)
    rows=[r.removeprefix('PARTICLE_TEACHER_PROOF=') for r in result.stdout.splitlines() if r.startswith('PARTICLE_TEACHER_PROOF=')]
    if len(rows)!=1: raise ValueError('Missing unique teacher proof')
    value=json.loads(rows[0]); root=Path(parent['root'])/'teacher_m16'
    return artifact('PARENT_TEACHER',parents=dict(spec=parent['content_hash'],receipt=value['receipt']),
        job_id=value['job_id'],report_hash=value['report'],files=[file_ref(root/p) for p in (
            'receipt.json','fit/receipt.json','fit/selected.pt','fit/training_report.json','fit/validation_predictions.npz')])


def registration():
    return dict(recipe=recipe(),tasks=list(TASKS),counts=COUNTS,original50k_rows=50000,
        execution_site=A6000_SITE,gpu_profile='a6000',cpus=6,memory_mb=32000,
        fit_max_minutes=2880,setup_minutes=120,release_minutes=60,report_minutes=60,
        cpu_site=dict(partition='compute',account='default',qos='normal',cpus=1,memory_mb=4000),
        resource_policy=dict(safety_multiplier=2.,fixed_overhead_seconds=900,memory_fraction=.85),
        storage_budget_bytes=4*2**30,reference_retrained=False,automatic_retry=False)


def storage(spec):
    root=Path(spec['root']); used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    remaining=spec['storage_budget_bytes']-used
    if remaining<=0 or shutil.disk_usage(root if root.exists() else root.parent).free<remaining+2**30:
        raise OSError('Need remaining four-GiB allowance plus one GiB free headroom; no deletion')


COPIED=('bundle_root','delta_root','delta_hash','encoder_root','encoder_hash','reference','imported','anchor')


def create(root,parent_path,project,commit,quota_gib):
    if type(quota_gib) is not int or quota_gib<4: raise ValueError('Reserve at least four GiB for outputs')
    parent_path=Path(parent_path).resolve(strict=True)
    parent,qualification=authenticate_parent(parent_path); teacher=teacher_proof(parent_path,parent)
    root=Path(root).absolute(); project=Path(project).resolve()
    protected=[Path(parent[k]).resolve() for k in ('root','project_dir','environment_dir','bundle_root','delta_root','encoder_root')]+[project]
    if (root.exists() or not root.parent.is_dir() or root.resolve()!=root or any(p.is_symlink() for p in (root,*root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh, separate particle-teacher campaign root')
    snapshot=current(project,commit); site.environment(parent['reference']['environment'])
    spec=artifact('SPEC',parents=dict(source=snapshot['content_hash'],parent=parent['content_hash'],
        qualification=qualification['content_hash'],teacher=teacher['content_hash']),**registration(),
        root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=snapshot,
        environment_dir=str(Path(sys.prefix).resolve()),parent_spec=file_ref(parent_path),qualification=qualification,
        teacher_proof=teacher,available_quota_gib=quota_gib,
        embedding_hash=Bundle(parent['bundle_root']).json('targets/u100/manifest.json')['content_hash'],
        **{k:parent[k] for k in COPIED})
    storage(spec); root.mkdir(); write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def check(spec,*,check_storage=True):
    validate_file_ref(spec['parent_spec']); parent,proof=authenticate_parent(spec['parent_spec']['path'])
    teacher=teacher_proof(spec['parent_spec']['path'],parent)
    validate(spec,'SPEC',dict(source=spec['source_snapshot']['content_hash'],parent=parent['content_hash'],
        qualification=proof['content_hash'],teacher=teacher['content_hash']))
    if (spec['qualification']!=proof or spec['teacher_proof']!=teacher
            or any(spec.get(k)!=v for k,v in registration().items()) or any(spec[k]!=parent[k] for k in COPIED)
            or spec['embedding_hash']!=Bundle(spec['bundle_root']).json('targets/u100/manifest.json')['content_hash']
            or spec['available_quota_gib']<4 or Path(sys.prefix).resolve()!=Path(spec['environment_dir'])
            or current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']
            or load_json(Path(spec['root'])/'campaign_spec.json')!=spec):
        raise ValueError('Immutable particle-teacher registration differs')
    site.environment(spec['reference']['environment'])
    if check_storage: storage(spec)
    return parent
