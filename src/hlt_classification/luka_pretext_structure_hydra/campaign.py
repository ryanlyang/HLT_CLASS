"""Pinned additive registration; historical artifacts validated in their own code."""
from pathlib import Path
import subprocess
import sys
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_next_supervision_hydra.campaign import storage
from hlt_classification.luka_selective_teacher_hydra import reuse
from hlt_classification.luka_selective_teacher_hydra.campaign import anchors
from hlt_classification.luka_cross_message_hydra.execution import A6000_SITE
from hlt_classification.luka_u100_pair_hydra import site
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle
from .contracts import BASE, COUNTS, TEACHERS, STUDENTS, FITS, artifact, validate, recipe

MODULES=('__init__','contracts','campaign','transfer','structure','corruption','normalization','target_banks',
         'inputs','embedding_banks','teacher_model','teacher_training','student_model','student_training',
         'resources','submission','worker','reporting')
RUNTIME={f'src/hlt_classification/luka_pretext_structure_hydra/{n}.py' for n in MODULES}|{
    'scripts/luka_pretext_structure_hydra.py','scripts/queue_luka_pretext_structure_hydra.sh','sbatch/run_luka_pretext_structure_hydra.sh'}
REQUIRED=RUNTIME|{'tests/test_luka_pretext_structure_hydra.py','docs/plans/LUKA_OFFLINE_PRETEXT_STRUCTURE_HYDRA_PLAN.md',
    'docs/contracts/LUKA_PRETEXT_STRUCTURE_HYDRA.md','docs/LUKA_PRETEXT_STRUCTURE_HYDRA_RUNBOOK.md','docs/LUKA_PRETEXT_STRUCTURE_HYDRA_VALIDATION.md'}
COPIED=('bundle_root','reference','imported','anchor','embedding_hash','qualification','offline_root','offline_hash')


def current(project,commit):
    project=Path(project).resolve(); snapshot=_source(project,commit)
    def git(*args):
        return subprocess.run(['git','-C',str(project),*args],check=True,capture_output=True,text=True).stdout.splitlines()
    git('merge-base','--is-ancestor',BASE,commit)
    changes=git('diff','--name-status','--no-renames',BASE,commit,'--','src','scripts','sbatch')
    if (not REQUIRED<=set(git('ls-files')) or any(r[:2]!='A\t' or r[2:] not in RUNTIME for r in changes)
            or Path(__file__).resolve()!=project/'src/hlt_classification/luka_pretext_structure_hydra/campaign.py'):
        raise ValueError('Commit the complete additive pretext campaign; preserve pinned donors')
    return snapshot


def registration():
    return dict(recipe=recipe(),tasks=list(FITS),counts=COUNTS,execution_site=A6000_SITE,gpu_profile='a6000',
        cpus=6,memory_mb=32000,setup_minutes=720,fit_max_minutes=2880,release_minutes=60,report_minutes=60,
        cpu_site=dict(partition='compute',account='default',qos='normal',cpus=1,memory_mb=4000),
        storage_budget_bytes=32*2**30,resource_policy=dict(safety_multiplier=2.,fixed_overhead_seconds=900,memory_fraction=.85),
        teacher_fits=8,student_fits=22,automatic_retry=False,original50k_rows=50000)


def nodes(old):
    original=Bundle(old['bundle_root']).json('reference/OFFLINE/report.json')['node']
    result={t:dict(original,node_id=t,coordinate='OFFLINE',deployable=False,teacher=None,
        initialization_seed=original['initialization_seed']+200000,sampler_seed=original['sampler_seed']+200000) for t in TEACHERS}
    result.update({t:dict(old['imported']['node'],node_id=t,coordinate='D000',deployable=True) for t in STUDENTS})
    return result


def raw_source(old):
    validate_file_ref(old['group_a_spec']); group=load_json(old['group_a_spec']['path'])
    from hlt_classification.luka_next_supervision_hydra.contracts import validate as validate_group
    validate_group(group,'SPEC')
    if group['bundle_root']!=old['bundle_root'] or group['anchor']!=old['anchor']:
        raise ValueError('Original raw parent differs')
    return dict(raw_root=group['raw_root'],raw_hash=group['raw_hash'])


def create(root,parent_path,validation_root,validation_hash,project,commit,quota_gib):
    from .transfer import sources
    project=Path(project).resolve(); root=Path(root).absolute()
    if type(quota_gib) is not int or quota_gib<34: raise ValueError('Reserve 32 GiB outputs and 2 GiB headroom after transfers')
    source=current(project,commit); old=reuse.original(parent_path); raw=raw_source(old)
    protected=[Path(old[k]).resolve() for k in ('root','project_dir','environment_dir','bundle_root','offline_root')]+[
        project,Path(raw['raw_root']).resolve(),Path(validation_root).resolve()]
    if (root.exists() or not root.parent.is_dir() or root.resolve()!=root or any(p.is_symlink() for p in (root,*root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh separate campaign output root')
    if Path(sys.prefix).resolve()!=Path(old['environment_dir']).resolve(): raise ValueError('Use pinned Brown environment')
    site.environment(old['reference']['environment'])
    value=artifact('SPEC',parents=dict(source=source['content_hash'],parent=old['content_hash'],recipe=recipe()['content_hash'],validation=validation_hash),
        **registration(),root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=source,
        environment_dir=str(Path(sys.prefix).resolve()),teacher_sweep_spec=file_ref(Path(parent_path).resolve()),
        reused=reuse.references(old),nodes=nodes(old),recovery_anchors=anchors(old),available_quota_gib=quota_gib,
        validation_root=str(Path(validation_root).resolve(strict=True)),validation_hash=validation_hash,
        **raw,**{k:old[k] for k in COPIED})
    sources(value); storage(value); root.mkdir(); write_immutable_json(root/'campaign_spec.json',value)
    return value


def check(spec,*,check_storage=True):
    from .transfer import sources
    validate_file_ref(spec['teacher_sweep_spec']); old=reuse.original(spec['teacher_sweep_spec']['path'])
    validate(spec,'SPEC',dict(source=spec['source_snapshot']['content_hash'],parent=old['content_hash'],
                             recipe=recipe()['content_hash'],validation=spec['validation_hash']))
    if (any(spec.get(k)!=v for k,v in registration().items()) or any(spec[k]!=old[k] for k in COPIED)
            or any(spec[k]!=v for k,v in raw_source(old).items()) or spec['nodes']!=nodes(old)
            or spec['reused']!=reuse.references(old) or spec['recovery_anchors']!=anchors(old)
            or spec['available_quota_gib']<34 or Path(sys.prefix).resolve()!=Path(spec['environment_dir'])
            or current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']
            or load_json(Path(spec['root'])/'campaign_spec.json')!=spec):
        raise ValueError('Immutable campaign/source/parent differs')
    site.environment(spec['reference']['environment']); sources(spec)
    if check_storage: storage(spec)
