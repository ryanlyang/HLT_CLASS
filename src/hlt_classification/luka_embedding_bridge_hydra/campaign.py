"""Immutable scientific registry, exact source and conservative byte inventory."""
from pathlib import Path
import math
import subprocess
import sys
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_subclass_hydra.reuse import original
from hlt_classification.luka_next_supervision_hydra.campaign import storage
from hlt_classification.luka_u100_pair_hydra import site
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle
from .contracts import REPAIR_BASE,COUNTS,artifact,validate,recipe
from .registry import FITS,PRODUCERS,dependencies
from .execution import configuration
from .transfer import fresh,Transfer
from . import native

MODULES=('__init__','registry','contracts','regions','response','losses','model','storage','transfer','native','inputs',
         'distributed','banks','training','campaign','execution','resources','qualification','worker','submission','reporting')
RUNTIME={f'src/hlt_classification/luka_embedding_bridge_hydra/{n}.py' for n in MODULES}|{
    'scripts/luka_embedding_bridge_hydra.py','scripts/queue_luka_embedding_bridge_hydra.sh','sbatch/run_luka_embedding_bridge_hydra.sh'}
REQUIRED=RUNTIME|{'tests/test_luka_embedding_bridge_hydra.py','tests/test_luka_embedding_bridge_native.py','docs/plans/LUKA_EMBEDDING_BRIDGE_RESPONSE_REGION_HYDRA_PLAN.md',
    'docs/contracts/LUKA_EMBEDDING_BRIDGE_HYDRA.md','docs/LUKA_EMBEDDING_BRIDGE_HYDRA_RUNBOOK.md','docs/LUKA_EMBEDDING_BRIDGE_HYDRA_VALIDATION.md'}
COPIED=('bundle_root','reference','imported','anchor','embedding_hash','qualification')


def current(project,commit):
    project=Path(project).resolve();source=_source(project,commit)
    def git(*args):return subprocess.run(['git','-C',str(project),*args],check=True,capture_output=True,text=True).stdout.splitlines()
    git('merge-base','--is-ancestor',REPAIR_BASE,commit)
    changes=git('diff','--name-status','--no-renames',REPAIR_BASE,commit,'--','src','scripts','sbatch')
    if (not REQUIRED<=set(git('ls-files')) or any(r[:2] not in ('A\t','M\t') or r[2:] not in RUNTIME for r in changes)
            or Path(__file__).resolve()!=project/'src/hlt_classification/luka_embedding_bridge_hydra/campaign.py'):
        raise ValueError('Commit the complete campaign repair; preserve donor runtime at the repair base')
    return source


def registration(profile='1080ti_2_fp32'):
    return dict(recipe=recipe(),tasks=list(FITS),remaining=list(FITS),counts=COUNTS,**configuration(profile),cpus=6,memory_mb=48000,
        setup_minutes=720,fit_max_minutes=2880,release_minutes=60,report_minutes=120,original50k_rows=50000,
        cpu_site=dict(partition='compute',account='default',qos='normal',cpus=1,memory_mb=4000),
        report_site=dict(partition='compute',account='default',qos='normal',cpus=2,memory_mb=8000),
        resource_policy=dict(safety_multiplier=2.,fixed_overhead_seconds=1800,memory_fraction=.85,gpu_fraction=.88),
        dependencies={t:dependencies(t) for t in FITS},automatic_retry=False,reference_retrained=True,
        native_input_policy=native.POLICY)


def inventory(transfer):
    """Account for every dense array, metadata bound, checkpoint and temporary output."""
    m=transfer.manifest;n=sum(COUNTS.values());h=sum(m['summaries'][r]['D000']['particles'] for r in COUNTS)
    o=sum(m['summaries'][r]['OFFLINE']['particles'] for r in COUNTS)
    # Each source row has 14 float64 fields; views have 17+4 float32 fields.
    physical=(h+o)*14*8+n*40+2*(n+2)*8
    views=(7*h+o)*21*4
    u100=n*(128*4+40)+4*128*8
    targets=(COUNTS['train']+4096)*((6+6+29)*128*4+(6+6+29)+8*48)
    # Full and stripped checkpoints for fits and two-epoch miniatures, plus
    # original teacher/replay and diagnostics. Explicit conservative cap per fit.
    checkpoints=2*len(FITS)*96*2**20
    predictions=len(FITS)*COUNTS['validation']*(40+9*11*4)
    technical=2**30;metadata=256*2**20;temporary=2**30
    items=dict(physical=physical,native_views=views,u100=u100,teacher_targets=targets,checkpoints=checkpoints,
        predictions=predictions,technical_targets=technical,metadata=metadata,temporary=temporary)
    return dict(items=items,storage_budget_bytes=sum(items.values()),minimum_quota_gib=math.ceil(sum(items.values())/2**30)+1,
        input_transfer_bytes=m['payload_bytes'],existing_inputs_excluded=True)


def load_anchors(old):return Bundle(old['bundle_root']).json('origin/campaign_spec.json')['recovery_anchors']


def historical_registration(path,*,parent):
    if path is None:return None
    from hlt_classification.luka_capacity_hydra.reuse import original as selective
    old=selective(path)
    if old['anchor']!=parent['anchor'] or old['bundle_root']!=parent['bundle_root']:raise ValueError('Historical selective inputs differ')
    return dict(spec=file_ref(Path(path).resolve()),spec_hash=old['content_hash'],
        files={t:{name:file_ref(Path(old['root'])/t/'fit'/name) for name in ('validation_predictions.npz','training_report.json')} for t in ('S2IT','S6')})


def check_historical(value,parent):
    if value is None:return
    from hlt_classification.luka_selective_teacher_hydra.contracts import validate as selective_validate
    validate_file_ref(value['spec']);old=load_json(value['spec']['path']);selective_validate(old,'SPEC')
    if old['content_hash']!=value['spec_hash'] or old['anchor']!=parent['anchor'] or old['bundle_root']!=parent['bundle_root'] or set(value['files'])!={'S2IT','S6'}:
        raise ValueError('Historical comparison lineage differs')
    for task,files in value['files'].items():
        if set(files)!={'validation_predictions.npz','training_report.json'}:raise ValueError('Historical comparison inventory differs')
        for name,ref in files.items():
            if Path(ref['path'])!=Path(old['root'])/task/'fit'/name:raise ValueError('Historical comparison ownership differs')
            validate_file_ref(ref)


def create(root,parent_path,transfer_root,transfer_hash,project,commit,quota_gib,*,native_root,native_hash,selective_spec=None):
    old=original(parent_path);project=Path(project).resolve();source=current(project,commit)
    b=Bundle(old['bundle_root']);t=Transfer(transfer_root,transfer_hash,b.json('origin/campaign_spec.json'))
    from .inputs import transferred
    parent_ref=file_ref(Path(parent_path).resolve())
    probe=dict(old,transfer_root=str(t.root),transfer_hash=transfer_hash,native_root=native_root,native_hash=native_hash,parent_spec=parent_ref)
    transferred(probe).verify_all();sources=native.Sources(probe);sources.verify_all();budget=inventory(t)
    if type(quota_gib)is not int or quota_gib<budget['minimum_quota_gib']:
        raise ValueError('Available output quota must be at least '+str(budget['minimum_quota_gib'])+' GiB after input transfer')
    root=fresh(root,[project,t.root,sources.delta.root,sources.offline.root,*[old[k] for k in ('root','project_dir','environment_dir','bundle_root')]])
    site.environment(old['reference']['environment'])
    value=artifact('SPEC',parents=dict(source=source['content_hash'],parent=old['content_hash'],transfer=transfer_hash,native=native_hash,recipe=recipe()['content_hash']),
        **registration(),root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=source,
        environment_dir=str(Path(sys.prefix).resolve()),parent_spec=parent_ref,
        native_root=str(sources.delta.root),native_hash=native_hash,
        transfer_root=str(t.root),transfer_hash=transfer_hash,available_quota_gib=quota_gib,storage_inventory=budget,
        storage_budget_bytes=budget['storage_budget_bytes'],control=old['control'],
        historical_selective=historical_registration(selective_spec,parent=old),
        nodes={task:dict(old['imported']['node'],node_id=task) for task in FITS},recovery_anchors=load_anchors(old),
        **{k:old[k] for k in COPIED})
    storage(value);root.mkdir();write_immutable_json(root/'campaign_spec.json',value)
    return value


def check(spec,*,check_storage=True):
    validate(spec,'SPEC')  # Explain v1 incompatibility before reading v2 fields.
    validate_file_ref(spec['parent_spec']);old=original(spec['parent_spec']['path'])
    validate(spec,'SPEC',dict(source=spec['source_snapshot']['content_hash'],parent=old['content_hash'],transfer=spec['transfer_hash'],native=spec['native_hash'],recipe=recipe()['content_hash']))
    from .inputs import transferred
    t=transferred(spec);native.Sources(spec);budget=inventory(t)
    if (any(spec.get(k)!=v for k,v in registration(spec['gpu_profile']).items()) or any(spec[k]!=old[k] for k in COPIED)
            or spec['control']!=old['control'] or spec['storage_inventory']!=budget or spec['storage_budget_bytes']!=budget['storage_budget_bytes']
            or spec['available_quota_gib']<budget['minimum_quota_gib'] or spec['recovery_anchors']!=load_anchors(old)
            or spec['nodes']!={task:dict(old['imported']['node'],node_id=task) for task in FITS}
            or Path(sys.prefix).resolve()!=Path(spec['environment_dir']) or current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']
            or load_json(Path(spec['root'])/'campaign_spec.json')!=spec):raise ValueError('Immutable campaign registration differs')
    site.environment(spec['reference']['environment'])
    check_historical(spec['historical_selective'],old)
    if check_storage:storage(spec)
