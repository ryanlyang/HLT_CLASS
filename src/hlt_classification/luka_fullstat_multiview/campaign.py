"""Two privileged teachers and one native HLT student; source-bound historical references."""
from pathlib import Path
import shutil
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site, validate_resources
from hlt_classification.luka_fullstat_multiteacher.campaign import protected
from hlt_classification.luka_fullstat_readout import campaign as readout_campaign
from hlt_classification.luka_fullstat_readout.contracts import validate as readout_validate, teacher_replay
from . import source, resources
from .contracts import artifact, validate, recipe, nodes, COUNTS, TASKS


def registration(reuse):
    validate(reuse,'REUSE'); u=reuse['u100']
    old=readout_campaign.registration(u)
    proof=reuse['startup']; readout_validate(proof,'STARTUP')
    if (reuse['source_commit']!=source.DONOR or reuse['counts']!=COUNTS
            or old['execution_site']!=execution_site('oscar_l40s')
            or proof['teacher_replay_hash']!=teacher_replay()['content_hash']
            or any(proof[k] is not True for k in ('teacher_readout_parity','installed_weaver_parity','finite_connected_gradients'))):
        raise ValueError('Wrong completed Oscar acceptance/population')
    refs=dict(old['references'],READOUT_CE=reuse['readout_reference'])
    validate_resources(old['execution_site'],6,192000,4)
    return dict(counts=dict(COUNTS),nodes=nodes(reuse),recipe=recipe(),
        anchor=u['anchor'],references=refs,bank_manifest=u['bank_manifest'],teacher_refs=u['teacher_refs'],
        teacher_replay=teacher_replay(),runtime_prefix=source.RUNTIME,execution_site=old['execution_site'],
        cpus=6,memory_mb=192000,minutes=resources.policy()['minutes'],runtime_policy=resources.policy(),
        storage_budget_bytes=20*2**30,filesystem_headroom_bytes=2*2**30,
        preflight_jobs=0,setup_jobs=0,scientific_fits=3,automatic_retry=False,
        teacher_order='afterany:teacher_CONCAT',historical_control='U100_W5')


def storage(spec):
    root=Path(spec['root']); existing=root if root.exists() else root.parent
    used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    remaining=spec['storage_budget_bytes']-used
    if remaining<=0 or shutil.disk_usage(existing).free<remaining+spec['filesystem_headroom_bytes']:
        raise OSError('Registered result allowance/headroom unavailable; nothing deleted')


def validate_spec(spec):
    validate(spec,'SPEC',dict(source=spec['source_snapshot']['content_hash'],reuse=spec['reuse']['content_hash']))
    if (any(spec.get(k)!=v for k,v in registration(spec['reuse']).items())
            or type(spec['available_quota_gib']) is not int or spec['available_quota_gib']<22):
        raise ValueError('Registered multiview science/resources differ')
    root=Path(spec['root'])
    if root.resolve()!=root or root.is_symlink() or not root.name.startswith('luka_fullstat_multiview_'):
        raise PermissionError('Dedicated unredirected multiview result root required')
    protected(root,dict(anchor=spec['anchor'],protected_roots=spec['reuse']['protected_roots']),spec['project_dir'])


def create(root,previous_spec,project,commit,*,quota_gib):
    root,project=Path(root).absolute(),Path(project).resolve()
    snapshot=source.current(project,commit); reuse=source.authenticate(previous_spec,project)
    spec=artifact('SPEC',parents=dict(source=snapshot['content_hash'],reuse=reuse['content_hash']),
        root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=snapshot,
        reuse=reuse,available_quota_gib=quota_gib,**registration(reuse))
    validate_spec(spec); storage(spec); root.mkdir(exist_ok=False)
    write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def check(spec,*,full_reuse=False):
    validate_spec(spec)
    if (load_json(Path(spec['root'])/'campaign_spec.json')!=spec
            or source.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']):
        raise ValueError('Pinned source or registration changed')
    source.check_registered_reuse(spec['reuse'],spec['project_dir'])
    if full_reuse and source.authenticate(spec['reuse']['parent_spec']['path'],spec['project_dir'])!=spec['reuse']:
        raise ValueError('Completed parent changed')
    storage(spec)
    return spec
