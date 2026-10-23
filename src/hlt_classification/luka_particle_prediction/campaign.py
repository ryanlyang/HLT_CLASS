"""Debug three-job registration; science remains gated on measured acceptance."""
from pathlib import Path
import shutil
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site
from .contracts import artifact, validate, recipe, node, tasks, ARMS, COUNTS
from . import source


def policy(parent):
    return dict(counts=COUNTS,original50k_rows=50000,recipe=recipe(),nodes={a:node(parent,a) for a in ARMS},
        tasks=tasks(),execution_site=execution_site('sporc_a100_debug'),cpus=6,memory_mb=90000,
        preflight_minutes=480,train_minutes=1440,storage_budget_bytes=2*2**30,
        scratch_base='/tmp',scratch_policy='verified_node_local_disk_unique_owned_per_job_no_home_fallback',automatic_retry=False)


def parents(spec):
    return dict(source=spec['source_snapshot']['content_hash'],reference=spec['reference']['content_hash'])


def create(root, reference_path, project, commit, *, available_quota_gib):
    root,project = Path(root).resolve(),Path(project).resolve()
    snapshot = source.current(project,commit); ref,parent = source.import_reference(reference_path)
    protected = [project,Path(parent['root']).parent,Path(parent['project_dir']),Path(reference_path).resolve().parent,
        Path(ref['anchor']['evidence']['prepared_root']),Path(ref['anchor']['evidence']['raw_container'])]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('Fresh root overlaps shared artifacts')
    if type(available_quota_gib) is not int or available_quota_gib < 2:
        raise ValueError('Reserve at least2GiB personally available persistent space')
    fields = dict(root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=snapshot,
        reference=ref,anchor=ref['anchor'],teacher=ref['teacher'],available_quota_gib=available_quota_gib,**policy(parent))
    spec = artifact('SPEC',parents=parents(fields),**fields)
    root.mkdir(parents=True,exist_ok=False); write_immutable_json(root/'campaign_spec.json',spec)
    storage(spec)
    return spec


def validate_spec(spec):
    validate(spec,'SPEC',parents(spec)); source.validate_reference(spec['reference'])
    parent = load_json(spec['anchor']['spec']['path'])
    if (any(spec.get(k) != v for k,v in policy(parent).items())
            or spec['anchor'] != spec['reference']['anchor'] or spec['teacher'] != spec['reference']['teacher']
            or type(spec['available_quota_gib']) is not int or spec['available_quota_gib'] < 2
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Two-arm debug registration differs')


def check(spec):
    validate_spec(spec)
    if source.current(spec['project_dir'],spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Source changed')
    ref,_ = source.import_reference(spec['reference']['spec']['path'])
    if ref != spec['reference']: raise ValueError('Historical reference changed')


def storage(spec):
    root = Path(spec['root']); used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if used >= spec['storage_budget_bytes'] or shutil.disk_usage(root).free < spec['storage_budget_bytes']-used:
        raise OSError('Persistent allowance unavailable; no deletion performed')
