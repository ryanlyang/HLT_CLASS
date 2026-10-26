"""Four fresh scientific jobs; bounded acceptance reuse, never a standalone preflight."""
from pathlib import Path
import shutil
import math
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.luka_fullstat_multiteacher.campaign import protected
from . import source
from .contracts import artifact, validate, recipe, node, tasks, STAGES, COUNTS


def policy(reuse):
    from hlt_classification.luka_fullstat_splithead.campaign import policy as inherited_policy
    validate(reuse, 'REUSE')
    inherited = inherited_policy(reuse['accepted'])
    if (reuse['counts'] != COUNTS or set(reuse['external_refs']) != {'OFFLINE', 'M0HLT'}
            or set(reuse['imported_banks']) != {'OFFLINE', 'M0HLT'}
            or reuse['parent_contract'] not in source.PARENT_CONTRACTS
            or reuse['runtime_prefix'] != source.RUNTIME
            or set(reuse['jobs']) != {'finalize_DE033', 'train_DE000'}
            or len(set(reuse['jobs'].values())) != 2
            or any(not str(j).isdigit() or int(j) <= 0 for j in reuse['jobs'].values())):
        raise ValueError('Expected the completed recovery ladder and original OFFLINE/M0HLT')
    for key in ('anchor', 'execution_site'):
        if reuse[key] != inherited[key]:
            raise ValueError('Inherited execution differs: '+key)
    for key in ('environment', 'gpu', 'summaries', 'base_node'):
        if reuse[key] != reuse['accepted'][key]:
            raise ValueError('Inherited runtime differs: '+key)
    for t in ('OFFLINE', 'M0HLT'):
        if (set(reuse['imported_banks'][t]) != {'embedding', 'kd'}
                or reuse['external_refs'][t] != inherited['external_refs'][t]):
            raise ValueError('Imported native teacher inventory differs')
    validate_file_ref(reuse['completed_parent_receipt'])
    # CPU caches and native banks plus export/moment/loading workspace. Reference
    # gathers are bounded at 2048x128 per teacher; never load reference particles.
    for co in STAGES.values():
        rows = sum(reuse['summaries'][role][co]['resident_bytes'] for role in ('train', 'validation'))
        if 2*rows + COUNTS['train']*(128*8*8+11*4*4*2) + 16*2**30 >= .85*180000*1024**2:
            raise MemoryError('Registered caches/banks exceed Oscar host envelope')
    minutes = max(1440, math.ceil(1.5*max(inherited['task_minutes'].values()))+60)
    if minutes > 2880:
        raise ValueError('Conservative request exceeds 48h; explicit new envelope required')
    return dict(anchor=reuse['anchor'], references=reuse['references'], runtime_prefix=source.RUNTIME,
        external_refs=reuse['external_refs'], imported_banks=reuse['imported_banks'],
        counts=dict(COUNTS), nodes={s: node(reuse['base_node'], s) for s in STAGES},
        recipes={s: recipe(s) for s in STAGES}, tasks=tasks(), execution_site=reuse['execution_site'],
        cpus=6, memory_mb=180000, train_minutes=minutes,
        task_minutes={s: minutes for s in STAGES}, preflight_jobs=0, setup_jobs=0,
        new_scientific_fits=4, automatic_dependency=True, automatic_retry=False,
        resource_policy='max_24h_or_1.5_inherited_plus60min_integrated_measurement_required',
        startup=dict(stress_steps=3, technical_epochs=0, warmup_steps=4,
            benchmark_steps=32, validation_benchmark_steps=16, full_validation_pass=False,
            timing_margin=1.5, timing_allowance_minutes=120))


def storage(spec):
    if spec['available_result_bytes'] < 16*2**30 or shutil.disk_usage(spec['root']).free < 16*2**30:
        raise OSError('Need 16 GiB personally available durable headroom; no deletion')


def create(root, previous_spec, project, commit, *, quota_gib):
    root, project = Path(root).resolve(), Path(project).resolve()
    active = source.current(project, commit)
    reuse = source.authenticate(previous_spec, project)
    protected(root, reuse, project)
    if type(quota_gib) is not int or quota_gib < 16:
        raise ValueError('Declare at least 16 GiB personal durable headroom')
    spec = artifact('SPEC', parents=dict(source=active['content_hash'], reuse=reuse['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=active,
        reuse=reuse, available_result_bytes=quota_gib*2**30, **policy(reuse))
    root.mkdir(parents=True, exist_ok=False)
    storage(spec)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def check(spec):
    validate(spec, 'SPEC', dict(source=spec['source_snapshot']['content_hash'], reuse=spec['reuse']['content_hash']))
    if source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Execution source changed')
    reuse = source.authenticate(spec['reuse']['parent_spec']['path'], spec['project_dir'])
    if (reuse != spec['reuse'] or any(spec.get(k) != v for k, v in policy(reuse).items())
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec
            or type(spec['available_result_bytes']) is not int or spec['available_result_bytes'] < 16*2**30):
        raise ValueError('Source, bank, recipe or reference registration changed')
    protected(spec['root'], reuse, spec['project_dir'])
    return reuse
