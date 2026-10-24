"""One shared export and four fresh final-rung fits; preserve old campaigns."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_embedding_weight.worker import storage
from .contracts import artifact, validate, COUNTS, WEIGHTS, recipe, nodes, tasks
from . import source


def policy(parent):
    site = execution_site('sporc_a100_debug')
    if (parent['execution_site'] != site or parent['cpus'] != 6 or parent['memory_mb'] != 90000
            or parent['counts'] != COUNTS):
        raise ValueError('Expected matched debug A100/6CPU/90000MiB c50 execution')
    inherited = parent['task_minutes']['c50_DE000']
    if type(inherited) is not int or not 1 <= inherited <= 1440:
        raise ValueError('Inherited allowance exceeds debug24h')
    minutes = max(480, inherited)
    return dict(counts=COUNTS, original50k_rows=50000, nodes=nodes(parent), recipes={a: recipe(a) for a in WEIGHTS},
        tasks=tasks(), execution_site=site, cpus=6, memory_mb=90000, report_cpus=1, report_memory_mb=16384,
        task_minutes={t['task_id']: 120 if t['task_id'] == 'report' else minutes for t in tasks()},
        storage_budget_bytes=3*2**30, preflight_jobs=0, scientific_fits=4, technical_epochs=0,
        startup_steps_per_arm=3, startup_batch=128, inherited_resources_new_objective_unmeasured=True,
        automatic_retry=False, native_models_retained=True, target_retirement='after_four_fits_and_saved_report')


def create(root, parent_path, project, commit, *, available_quota_gib):
    root, project = Path(root).resolve(), Path(project).resolve()
    snapshot = source.current(project, commit)
    reuse, parent = source.import_parent(parent_path, project)
    protected = [project, Path(parent['project_dir']), Path(parent['root']),
        Path(parent['anchor']['evidence']['prepared_root']), Path(parent['anchor']['evidence']['raw_container'])]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('Fresh output overlaps protected source/data/parent')
    if type(available_quota_gib) is not int or available_quota_gib < 3:
        raise ValueError('Reserve at least 3 GiB personally available persistent storage')
    spec = artifact('SPEC', parents=dict(source=snapshot['content_hash'], reuse=reuse['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=snapshot,
        reuse=reuse, anchor=reuse['anchor'], available_quota_gib=available_quota_gib, **policy(parent))
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec); storage(spec)
    return spec


def validate_spec(spec, *, imports=True):
    validate(spec, 'SPEC', dict(source=spec['source_snapshot']['content_hash'], reuse=spec['reuse']['content_hash']))
    # Reporting after publication authenticates retained new artifacts without
    # requiring the historical teacher files or disposable target payloads.
    validate_file_ref(spec['reuse']['spec'])
    parent = load_json(spec['reuse']['spec']['path'])
    if imports:
        parent = source.validate_reuse(spec['reuse'])
    validate(spec['reuse'], 'REUSE', dict(spec=parent['content_hash']))
    if (any(spec.get(k) != v for k, v in policy(parent).items()) or spec['anchor'] != spec['reuse']['anchor']
            or type(spec['available_quota_gib']) is not int or spec['available_quota_gib'] < 3
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Registered follow-up policy differs')


def check(spec):
    validate_spec(spec)
    if source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Pinned source changed')
