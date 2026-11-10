"""Fresh immutable executions over the existing small-data teacher-sweep inputs."""
from pathlib import Path
import subprocess
import sys
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.luka_fullsim.foundation import _source
from .reuse import original, references
from hlt_classification.luka_next_supervision_hydra.campaign import storage
from hlt_classification.luka_cross_message_hydra.execution import A6000_SITE
from hlt_classification.luka_u100_pair_hydra import site
from .contracts import BASE, COUNTS, FITS, recipe, artifact, validate

MODULES=('__init__','contracts','campaign','inputs','training','resources','model','submission','worker','reporting','targets','losses','reuse')
RUNTIME={f'src/hlt_classification/luka_subclass_hydra/{n}.py' for n in MODULES}|{
    'scripts/luka_subclass_hydra.py','scripts/queue_luka_subclass_hydra.sh',
    'sbatch/run_luka_subclass_hydra.sh'}
REQUIRED=RUNTIME|{'tests/test_luka_subclass_hydra.py',
    'docs/plans/LUKA_SUBCLASS_HYDRA_PLAN.md','docs/contracts/LUKA_SUBCLASS_HYDRA.md',
    'docs/LUKA_SUBCLASS_HYDRA_RUNBOOK.md','docs/LUKA_SUBCLASS_HYDRA_VALIDATION.md'}
COPIED=('bundle_root','reference','imported','anchor','embedding_hash','qualification')
def current(project,commit):
    project=Path(project).resolve(); snapshot=_source(project,commit)
    def git(*args): return subprocess.run(['git','-C',str(project),*args],check=True,capture_output=True,text=True).stdout.splitlines()
    git('merge-base','--is-ancestor',BASE,commit)
    changes=git('diff','--name-status','--no-renames',BASE,commit,'--','src','scripts','sbatch')
    if (not REQUIRED<=set(git('ls-files')) or any(r[:2]!='A\t' or r[2:] not in RUNTIME for r in changes)
            or Path(__file__).resolve()!=project/'src/hlt_classification/luka_subclass_hydra/campaign.py'):
        raise ValueError('Commit the complete additive subclass; preserve pinned donor runtime')
    return snapshot


def registration():
    return dict(recipe=recipe(),tasks=list(FITS),remaining=list(FITS),counts=COUNTS,
        execution_site=A6000_SITE,gpu_profile='a6000',cpus=6,memory_mb=32000,
        setup_minutes=720,fit_max_minutes=2880,release_minutes=60,report_minutes=60,original50k_rows=50000,
        cpu_site=dict(partition='compute',account='default',qos='normal',cpus=1,memory_mb=4000),
        storage_budget_bytes=4*2**30,resource_policy=dict(safety_multiplier=2.,fixed_overhead_seconds=900,memory_fraction=.85),
        science_dependencies='none',automatic_retry=False,reference_retrained=False)


def create(root,teacher_sweep_spec,project,commit,quota_gib):
    old=original(teacher_sweep_spec); project=Path(project).resolve(); root=Path(root).absolute()
    if type(quota_gib) is not int or quota_gib<5: raise ValueError('Reserve four GiB outputs plus one GiB headroom')
    protected=[Path(old[k]).resolve() for k in ('root','project_dir','environment_dir','bundle_root')]+[
        project,Path(old['offline_root']).resolve()]
    if (root.exists() or not root.parent.is_dir() or root.resolve()!=root or any(p.is_symlink() for p in (root,*root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh separate subclass output root')
    source=current(project,commit); site.environment(old['reference']['environment'])
    value=artifact('SPEC',parents=dict(source=source['content_hash'],teacher_sweep=old['content_hash'],recipe=recipe()['content_hash']),
        **registration(),root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=source,
        environment_dir=str(Path(sys.prefix).resolve()),teacher_sweep_spec=file_ref(Path(teacher_sweep_spec).resolve()),
        available_quota_gib=quota_gib,control=old['control'],teacher_refs=references(old),
        nodes={t: dict(old['imported']['node'], node_id=t) for t in FITS},
        recovery_anchors=load_anchors(old),**{k:old[k] for k in COPIED})
    from .inputs import validate_sources
    validate_sources(value); storage(value); root.mkdir(); write_immutable_json(root/'campaign_spec.json',value)
    return value


def check(spec,*,check_storage=True):
    validate_file_ref(spec['teacher_sweep_spec']); old=original(spec['teacher_sweep_spec']['path'])
    validate(spec,'SPEC',dict(source=spec['source_snapshot']['content_hash'],teacher_sweep=old['content_hash'],recipe=recipe()['content_hash']))
    if (any(spec.get(k)!=v for k,v in registration().items()) or any(spec[k]!=old[k] for k in COPIED)
            or spec['control']!=old['control'] or spec['teacher_refs']!=references(old)
            or spec['nodes']!={t:dict(old['imported']['node'],node_id=t) for t in FITS}
            or spec['recovery_anchors']!=load_anchors(old)
            or spec['available_quota_gib']<5 or Path(sys.prefix).resolve()!=Path(spec['environment_dir'])
            or current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']
            or load_json(Path(spec['root'])/'campaign_spec.json')!=spec): raise ValueError('Immutable subclass registration differs')
    site.environment(spec['reference']['environment'])
    from .inputs import validate_sources
    validate_sources(spec)
    if check_storage: storage(spec)


def load_anchors(spec):
    from hlt_classification.luka_u100_pair_hydra.inputs import Bundle
    return Bundle(spec['bundle_root']).json('origin/campaign_spec.json')['recovery_anchors']
