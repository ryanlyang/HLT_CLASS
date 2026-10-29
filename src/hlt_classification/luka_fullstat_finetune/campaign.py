"""One short refinement in a new root; historical artifacts remain read-only."""
from pathlib import Path
import shutil
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_fullstat_multiteacher.campaign import protected
from hlt_classification.luka_fullstat_fine_tail.contracts import node as original_node
from . import source
from .contracts import artifact, validate, recipe, COUNTS


def registration(reuse):
    validate(reuse,'REUSE'); old = recipe()['objective']; node = reuse['base_node']
    if (reuse['counts'] != COUNTS or reuse['source_commit'] != source.DONOR
            or reuse['parent_recipe'] != old or node['node_id'] != 'DE000'
            or node['coordinate'] != 'D000' or node.get('context_coordinate') is not None
            or node['deployable'] is not True
            or node != original_node(node,'DE000')
            or set(reuse['imported_banks']) != set(old['teacher_order'])
            or set(reuse['teacher_refs']) != set(old['teacher_order'])
            or reuse['execution_site'] != execution_site('oscar_l40s') or 'L40S' not in reuse['gpu']['name']
            or set(reuse['jobs']) != {'train_DE022','train_DE011','train_DE000'}):
        raise ValueError('Wrong fine-tail initializer, objective, teachers or Oscar evidence')
    rows = sum(reuse['summaries'][r]['D000']['resident_bytes'] for r in ('train','validation'))
    if 2*rows + COUNTS['train']*(128*4*15+11*4*14) + 16*2**30 >= .85*180000*1024**2:
        raise MemoryError('Caches, seven banks and frozen-feature workspace exceed envelope')
    return dict(counts=dict(COUNTS),node=node,recipe=recipe(),anchor=reuse['anchor'],
        initial_checkpoint=reuse['reference']['checkpoint'],references=reuse['references'],
        runtime_prefix=source.RUNTIME,execution_site=reuse['execution_site'],cpus=6,memory_mb=180000,
        minutes={'finetune':1440},storage_budget_bytes=4*2**30,preflight_jobs=0,setup_jobs=0,
        scientific_fits=1,automatic_retry=False)


def storage(spec):
    root=Path(spec['root']); existing=root if root.exists() else root.parent
    used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    remaining=spec['storage_budget_bytes']-used
    if remaining<=0 or shutil.disk_usage(existing).free<remaining:
        raise OSError('Registered 4 GiB result allowance unavailable; nothing deleted')


def validate_spec(spec):
    validate(spec,'SPEC',dict(source=spec['source_snapshot']['content_hash'],reuse=spec['reuse']['content_hash']))
    if (any(spec.get(k)!=v for k,v in registration(spec['reuse']).items())
            or type(spec['available_quota_gib']) is not int or spec['available_quota_gib']<4):
        raise ValueError('Registered refinement or quota differs')
    root=Path(spec['root'])
    if root.resolve()!=root or root.is_symlink() or not root.name.startswith('luka_fullstat_finetune_'):
        raise PermissionError('Dedicated unredirected fine-tune result root required')
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
        raise ValueError('Pinned source, parent or registration changed')
    validate_file_ref(spec['initial_checkpoint']); storage(spec)
    return spec
