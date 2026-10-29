"""Inherited Oscar acceptance, explicit quotas and immutable two-job registration."""
from pathlib import Path
import shutil
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.luka_fullstat_multiteacher.campaign import protected
from hlt_classification.luka_fullstat_attention.worker import metadata
from . import source, banks
from .contracts import artifact, validate, recipe, node, tasks, STAGES, COUNTS, EXTERNAL


def policy(reuse):
    from hlt_classification.luka_fullstat_splithead.campaign import policy as inherited_policy
    validate(reuse, 'REUSE')
    inherited = inherited_policy(reuse['accepted'])
    if (reuse['counts'] != COUNTS or set(reuse['external_refs']) != set(EXTERNAL)
            or set(reuse['imported_banks']) != set(EXTERNAL)
            or reuse['parent_contract'] not in source.PARENT_CONTRACTS or reuse['runtime_prefix'] != source.RUNTIME
            or set(reuse['jobs']) != {'finalize_DE033', 'train_DE000'}
            or len(set(reuse['jobs'].values())) != 2
            or any(not str(j).isdigit() or int(j) <= 0 for j in reuse['jobs'].values())):
        raise ValueError('Expected completed non-relational parent and exact four teachers')
    for key in ('anchor', 'execution_site'):
        if reuse[key] != inherited[key]:
            raise ValueError('Inherited execution differs')
    for key in ('environment', 'gpu', 'summaries', 'base_node'):
        if reuse[key] != reuse['accepted'][key]:
            raise ValueError('Inherited runtime evidence differs')
    for t in EXTERNAL:
        if (set(reuse['imported_banks'][t]) != {'embedding', 'kd'}
                or (t in inherited['external_refs'] and reuse['external_refs'][t] != inherited['external_refs'][t])):
            raise ValueError('Teacher inventory differs')
    validate_file_ref(reuse['completed_parent_receipt'])
    references = {k:reuse['references'][k] for k in ('M0HLT', 'OFFLINE', 'OLD_SPLITHEAD_DE000')}
    return dict(anchor=reuse['anchor'], references=references, runtime_prefix=source.RUNTIME,
        external_refs=reuse['external_refs'], imported_banks=reuse['imported_banks'], counts=dict(COUNTS),
        nodes={s:node(reuse['base_node'], s) for s in STAGES}, recipes={s:recipe(s) for s in STAGES},
        tasks=tasks(), execution_site=reuse['execution_site'], cpus=6, memory_mb=180000,
        target_memory_mb=90000, target_minutes=1440, task_minutes={s:1440 for s in STAGES},
        preflight_jobs=0, setup_jobs=1, new_scientific_fits=1, preparation_workers=6,
        automatic_dependency=True, automatic_retry=False, fresh_matched_control=False,
        startup=dict(stress_steps=3, technical_epochs=0, warmup_steps=4, benchmark_steps=32,
            validation_benchmark_steps=16, full_validation_pass=False, timing_margin=1.5, timing_allowance_minutes=120))


def separate_outputs(root, target_root, reuse, project):
    root, target_root = Path(root).resolve(), Path(target_root).resolve()
    if root.is_relative_to(target_root) or target_root.is_relative_to(root):
        raise PermissionError('Durable results and target bank must be separate roots')
    for p in (root, target_root):
        protected(p, reuse, project)
        if p.parent == p or not p.name.startswith('luka_fullstat_neighborhood_'):
            raise PermissionError('Use dedicated named neighborhood output directories')


def resource_budget(spec, manifest):
    rows, tokens = banks.support(manifest)
    if rows != spec['counts']['train'] or manifest['content_hash'] != spec['anchor']['evidence']['prepared']:
        raise ValueError('Prepared TRAIN population differs')
    need = banks.size(rows, tokens)
    resident = sum(manifest['summaries'][r]['D000']['resident_bytes'] for r in ('train', 'validation'))
    # Account for cache load copies, four compact banks and decoding/checksum scratch.
    host = 2*resident + tokens*144 + rows*(128+11)*4*4*2 + 16*2**30
    if host >= .85*spec['memory_mb']*1024**2:
        raise MemoryError('Resident neighborhoods plus native caches exceed inherited host envelope')
    preparation_host = tokens*144 + rows*96 + 8*2**30
    if preparation_host >= .85*spec['target_memory_mb']*1024**2:
        raise MemoryError('Neighborhood construction exceeds registered CPU memory envelope')
    return dict(rows=rows, tokens=tokens, target_required_bytes=need, conservative_host_bytes=host,
                conservative_preparation_host_bytes=preparation_host)


def storage(spec, *, before_targets=False):
    if type(spec['available_result_bytes']) is not int or spec['available_result_bytes'] < 8*2**30:
        raise OSError('Declare at least 8 GiB personal durable result headroom')
    parent = Path(spec['root']).parent
    if shutil.disk_usage(parent).free < 8*2**30:
        raise OSError('Insufficient result filesystem space')
    needed = spec['resources']['target_required_bytes']
    if type(spec['available_target_bytes']) is not int or spec['available_target_bytes'] < needed:
        raise OSError(f'Need {needed/2**30:.2f} GiB personal target-bank headroom')
    if before_targets and shutil.disk_usage(Path(spec['target_root']).parent).free < needed:
        raise OSError('Insufficient target filesystem space')


def create(root, previous_spec, project, commit, *, quota_gib, target_root, target_quota_gib):
    root, project, target_root = map(lambda p:Path(p).resolve(), (root, project, target_root))
    active = source.current(project, commit)
    reuse = source.authenticate(previous_spec, project)
    separate_outputs(root, target_root, reuse, project)
    if root.exists() or target_root.exists() or not root.parent.is_dir() or not target_root.parent.is_dir():
        raise FileExistsError('Fresh roots beneath existing authorized parents required')
    if any(type(x) is not int or x < 1 for x in (quota_gib, target_quota_gib)):
        raise ValueError('Declare separate integer personal quota headroom')
    fields = dict(root=str(root), target_root=str(target_root), project_dir=str(project), source_commit=commit,
        source_snapshot=active, reuse=reuse, available_result_bytes=quota_gib*2**30,
        available_target_bytes=target_quota_gib*2**30, **policy(reuse))
    fields['resources'] = resource_budget(fields, metadata(fields)[0])
    storage(fields, before_targets=True)
    spec = artifact('SPEC', parents=dict(source=active['content_hash'], reuse=reuse['content_hash']), **fields)
    root.mkdir(exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def check(spec):
    validate(spec, 'SPEC', dict(source=spec['source_snapshot']['content_hash'], reuse=spec['reuse']['content_hash']))
    if source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Execution source changed')
    reuse = source.authenticate(spec['reuse']['parent_spec']['path'], spec['project_dir'])
    if (reuse != spec['reuse'] or any(spec.get(k) != v for k, v in policy(reuse).items())
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec
            or resource_budget(spec, metadata(spec)[0]) != spec['resources']):
        raise ValueError('Source, recipe, reference or resource registration differs')
    separate_outputs(spec['root'], spec['target_root'], reuse, spec['project_dir'])
    storage(spec)
    return reuse
