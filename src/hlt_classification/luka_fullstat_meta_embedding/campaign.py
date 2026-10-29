"""Immutable FullStat continuation registration with explicit resource admission."""
from pathlib import Path
import shutil
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.luka_fullstat_multiteacher.campaign import protected
from hlt_classification.jetclass2_delphes.execution import execution_site
from .contracts import artifact, validate, recipe, node, COUNTS, MINUTES
from . import source


def registration(reuse):
    validate(reuse, 'REUSE')
    if (reuse['counts'] != COUNTS or reuse['source_commit'] != source.DONOR
            or reuse['execution_site'] != execution_site('oscar_l40s')
            or 'L40S' not in reuse['gpu']['name']
            or reuse['teacher_node']['node_id'] != 'COARSE_U100'
            or reuse['teacher_node']['coordinate'] != 'U100'
            or reuse['teacher_node']['context_coordinate'] is not None
            or set(reuse['jobs']) != {'export_U100','train_w5'}):
        raise ValueError('Wrong completed FullStat U100 student/teacher or Oscar evidence')
    return dict(counts=dict(COUNTS), node=node(reuse['base_node']), recipe=recipe(),
        execution_site=reuse['execution_site'], cpus=6, memory_mb=180000,
        minutes=dict(MINUTES), storage_budget_bytes=8*2**30, automatic_retry=False,
        runtime_prefix=source.RUNTIME, anchor=reuse['anchor'],
        teacher_refs=reuse['teacher_refs'], bank_manifest=reuse['bank_manifest'],
        initial_checkpoint=reuse['reference']['checkpoint'],
        comparison='selected_U100_w5_fixed_vs_student_benefit_continuation',
        preflight_jobs=0, setup_jobs=1, scientific_fits=2)


def storage(spec):
    root=Path(spec['root']); existing=root if root.exists() else root.parent
    used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    remaining=spec['storage_budget_bytes']-used
    if remaining<=0 or shutil.disk_usage(existing).free<remaining:
        raise OSError('Registered 8 GiB allowance unavailable; nothing deleted')
    return used


def validate_spec(spec):
    validate(spec,'SPEC',dict(source=spec['source_snapshot']['content_hash'],reuse=spec['reuse']['content_hash']))
    if (any(spec.get(k)!=v for k,v in registration(spec['reuse']).items())
            or type(spec['available_quota_gib']) is not int or spec['available_quota_gib']<8):
        raise ValueError('Registered science/resource recipe differs')
    root=Path(spec['root'])
    if root.resolve()!=root or root.is_symlink() or not root.name.startswith('luka_fullstat_meta_'):
        raise PermissionError('Use a dedicated unredirected FullStat meta directory')
    protected(root,spec['reuse'],spec['project_dir'])


def create(root,previous_spec,project,commit,*,quota_gib):
    root,project=Path(root).absolute(),Path(project).resolve()
    active=source.current(project,commit); reuse=source.authenticate(previous_spec,project)
    spec=artifact('SPEC',parents=dict(source=active['content_hash'],reuse=reuse['content_hash']),
        root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=active,
        reuse=reuse,available_quota_gib=quota_gib,**registration(reuse))
    validate_spec(spec); storage(spec)
    root.mkdir(exist_ok=False); write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def check(spec):
    validate_spec(spec)
    if (load_json(Path(spec['root'])/'campaign_spec.json')!=spec
            or source.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']
            or source.authenticate(spec['reuse']['parent_spec']['path'],spec['project_dir'])!=spec['reuse']):
        raise ValueError('Pinned source/spec or original replay changed')
    validate_file_ref(spec['bank_manifest'])
    storage(spec)
    return spec
