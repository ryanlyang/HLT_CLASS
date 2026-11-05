"""Fresh additive Group A registration, exact pushed source and qualified originals."""
from pathlib import Path
import shutil
import subprocess
import sys
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_u100_pair_hydra import site
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle
from hlt_classification.luka_cross_message_ce_hydra.campaign import authenticate_parent
from hlt_classification.luka_cross_message_hydra.execution import A6000_SITE
from hlt_classification.luka_u100_contrastive_hydra import inputs as contrast
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from .contracts import BASE,COUNTS,TASKS,ORDER,recipe,artifact,validate
from . import inputs,reuse

MODULES=('__init__','contracts','data','sampling','parallel','model','inputs','reuse','training','campaign','resources','submission','worker','reporting')
REQUIRED={f'src/hlt_classification/luka_next_supervision_hydra/{n}.py' for n in MODULES}|{
    'scripts/luka_next_supervision_hydra.py','scripts/queue_luka_next_supervision_hydra.sh',
    'sbatch/run_luka_next_supervision_hydra.sh','tests/test_luka_next_supervision_hydra.py',
    'docs/contracts/LUKA_NEXT_SUPERVISION_HYDRA.md','docs/LUKA_NEXT_SUPERVISION_GROUP_A_RUNBOOK.md'}


def current(project,commit):
    project=Path(project).resolve(); snapshot=_source(project,commit)
    def git(*args): return subprocess.run(['git','-C',str(project),*args],check=True,capture_output=True,text=True).stdout.splitlines()
    git('merge-base','--is-ancestor',BASE,commit)
    if not REQUIRED<=set(git('ls-files')) or Path(__file__).resolve()!=project/'src/hlt_classification/luka_next_supervision_hydra/campaign.py':
        raise ValueError('Commit the complete Group A implementation')
    donors=('jetclass2_delphes','luka_fullsim','luka_embedding_aux','luka_u100_embedding',
            'luka_u100_contrastive','luka_u100_pair_hydra','luka_cross_message_hydra')
    changes=git('diff','--name-only',BASE,commit,'--',*[f'src/hlt_classification/{n}' for n in donors])
    if changes: raise ValueError('Reviewed donor runtime changed; register a new compatibility version')
    return snapshot


def registration():
    return dict(recipe=recipe(),tasks=list(TASKS),global_order=list(ORDER),scope='group_A_only',counts=COUNTS,
        original50k_rows=50000,execution_site=A6000_SITE,gpu_profile='a6000',cpus=6,memory_mb=32000,
        setup_minutes=720,fit_max_minutes=2880,candidate_workers=4,resource_policy=dict(safety_multiplier=2.,fixed_overhead_seconds=900,memory_fraction=.85),
        storage_budget_bytes=4*2**30,science_dependencies='none',max_concurrent_fits=None,automatic_retry=False,
        reference_retrained=False,stopping_policy='full30',deferred_groups=['B','C'])


def storage(spec):
    root=Path(spec['root']); used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    remaining=spec['storage_budget_bytes']-used
    if remaining<=0 or shutil.disk_usage(root if root.exists() else root.parent).free<remaining+2**30:
        raise OSError('Need remaining four-GiB output allowance plus one GiB headroom')


COPIED=('bundle_root','reference','imported','anchor')


def create(root,parent_path,raw_root,raw_hash,delta_root,delta_hash,reuse_map,project,commit,quota_gib):
    parent,qualification=authenticate_parent(parent_path)
    project=Path(project).resolve(); root=Path(root).absolute()
    if type(quota_gib) is not int or quota_gib<4: raise ValueError('Reserve four GiB after raw transfer/environment')
    protected=[Path(parent[k]).resolve() for k in ('root','project_dir','environment_dir','bundle_root')]+[
        project,Path(raw_root).resolve(),Path(delta_root).resolve()]
    if (root.exists() or not root.parent.is_dir() or root.resolve()!=root or any(p.is_symlink() for p in (root,*root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)): raise ValueError('Use a fresh separate output root')
    source=current(project,commit); site.environment(parent['reference']['environment'])
    spec=dict(registration(),root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=source,
        environment_dir=str(Path(sys.prefix).resolve()),parent_spec=file_ref(Path(parent_path).resolve()),qualification=qualification,
        raw_root=str(Path(raw_root).resolve()),raw_hash=raw_hash,delta_root=str(Path(delta_root).resolve()),delta_hash=delta_hash,
        available_quota_gib=quota_gib,**{k:parent[k] for k in COPIED})
    b=Bundle(spec['bundle_root']); d=contrast.delta(spec); contrast.bind(spec,d)
    spec['embedding_hash']=b.json('targets/u100/manifest.json')['content_hash']
    spec['key_hash']=load_json(d.path('keys/manifest.json'))['content_hash']
    raw=inputs.raw(spec)
    if raw.manifest['original_m0']!=parent['anchor']['evidence']['models']['M0HLT']['checkpoint']:
        raise ValueError('Frozen HLT control checkpoint differs')
    if not {'C0','C1'}<=set(reuse_map) or set(reuse_map)-{'C0','C1','J5'}:
        raise ValueError('Supply completed same-seed C0/C1 specs; optional J5. Compatible references are not retrained')
    spec['reused']={row:reuse.inspect(row,path,spec['imported'],spec['anchor']['evidence']['prepared'],spec['embedding_hash'],spec['key_hash'] if row=='J5' else None) for row,path in reuse_map.items()}
    if 'J5' not in spec['reused']:
        dot=reuse.transferred_dot(spec)
        if dot is not None: spec['reused']['J5']=dot
    spec['remaining']=[t for t in TASKS if t not in spec['reused']]
    value=artifact('SPEC',parents=dict(source=source['content_hash'],parent=parent['content_hash'],raw=raw_hash,delta=delta_hash),**spec)
    storage(value); root.mkdir(); write_immutable_json(root/'campaign_spec.json',value)
    return value


def check(spec,*,check_storage=True):
    validate_file_ref(spec['parent_spec']); parent,qualification=authenticate_parent(spec['parent_spec']['path'])
    validate(spec,'SPEC',dict(source=spec['source_snapshot']['content_hash'],parent=parent['content_hash'],raw=spec['raw_hash'],delta=spec['delta_hash']))
    if (any(spec.get(k)!=v for k,v in registration().items()) or any(spec[k]!=parent[k] for k in COPIED)
            or spec['qualification']!=qualification or Path(sys.prefix).resolve()!=Path(spec['environment_dir'])
            or current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']
            or load_json(Path(spec['root'])/'campaign_spec.json')!=spec or spec['available_quota_gib']<4
            or not {'C0','C1'}<=set(spec['reused']) or set(spec['reused'])-{'C0','C1','J5'}
            or spec['remaining']!=[t for t in TASKS if t not in spec['reused']]): raise ValueError('Immutable Group A registration differs')
    b=Bundle(spec['bundle_root']); d=contrast.delta(spec); contrast.bind(spec,d); inputs.raw(spec)
    if (spec['embedding_hash']!=b.json('targets/u100/manifest.json')['content_hash']
            or spec['key_hash']!=load_json(d.path('keys/manifest.json'))['content_hash']): raise ValueError('Target bank changed')
    for proof in spec['reused'].values(): reuse.check(proof,imported=spec['imported'])
    site.environment(spec['reference']['environment'])
    if check_storage: storage(spec)
    return parent
