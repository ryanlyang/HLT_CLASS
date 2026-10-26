"""Three scientific jobs; bounded acceptance reuse, never a standalone preflight."""
from pathlib import Path
import shutil
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.luka_fullstat_multiteacher.campaign import protected
from . import source
from .contracts import artifact, validate, recipe, node, tasks, STAGES, COUNTS


def policy(reuse):
    from hlt_classification.luka_fullstat_hybrid_embedding.campaign import policy as inherited_policy
    validate(reuse, 'REUSE')
    inherited = inherited_policy(reuse['accepted'])
    if (reuse['counts'] != COUNTS or set(reuse['external_refs']) != {'OFFLINE', 'M0HLT', 'UE100'}
            or set(reuse['imported_banks']) != {'OFFLINE', 'M0HLT', 'UE100'}
            or set(reuse['jobs']) != {'train_UE000', 'train_UE100'}
            or any(not str(j).isdigit() or int(j) <= 0 for j in reuse['jobs'].values())):
        raise ValueError('Expected exact completed FullStat UE100/UE000 evidence')
    for key in ('anchor', 'execution_site'):
        if reuse[key] != inherited[key]:
            raise ValueError('Inherited execution binding differs: '+key)
    for key in ('environment', 'gpu', 'summaries', 'base_node'):
        if reuse[key] != reuse['accepted'][key]:
            raise ValueError('Inherited runtime binding differs: '+key)
    for t in ('OFFLINE', 'M0HLT', 'UE100'):
        expected = {'embedding'} if t == 'M0HLT' else {'embedding', 'kd'}
        if set(reuse['imported_banks'][t]) != expected:
            raise ValueError('Imported native bank inventory differs')
    # Include four probability banks and a loading copy explicitly. The inherited
    # bound also covers embedding banks, raw export and FP64 moments workspace.
    for co in STAGES.values():
        rows = sum(reuse['summaries'][role][co]['resident_bytes'] for role in ('train', 'validation'))
        if 2*rows + COUNTS['train']*(128*8*8+11*4*4*2) + 16*2**30 >= .85*180000*1024**2:
            raise MemoryError('Split-head caches/target banks exceed registered memory envelope')
    minutes = {}
    for stage in STAGES:
        old = reuse['parent_task_minutes'][stage]
        if type(old) is not int or old != inherited['task_minutes'][stage]:
            raise ValueError('Original resource request differs')
        minutes[stage] = old + (120 if stage == 'DE066' else 60)
    if max(minutes.values()) > 2880:
        raise ValueError('Inherited request plus allowance exceeds 48h')
    return dict(anchor=reuse['anchor'], references=reuse['references'],
        external_refs=reuse['external_refs'], imported_banks=reuse['imported_banks'],
        counts=dict(COUNTS), nodes={s: node(reuse['base_node'], s) for s in STAGES},
        recipes={s: recipe(s) for s in STAGES}, tasks=tasks(), execution_site=reuse['execution_site'],
        cpus=6, memory_mb=180000, train_minutes=max(minutes.values()), task_minutes=minutes,
        preflight_jobs=0, setup_jobs=0, new_scientific_fits=3, automatic_dependency=True,
        automatic_retry=False,
        resource_policy='inherited_FullStat_hybrid_plus60min_first_plus120min_unmeasured_split_heads',
        startup=dict(stress_steps=3, technical_epochs=0, benchmark_steps=0, full_validation_pass=False))


def storage(spec):
    minimum = 2 if spec.get('contract') in ('LUKA_FULLSTAT_SPLITHEAD_FINALIZE_SPEC/v1',
        'LUKA_FULLSTAT_SPLITHEAD_FINALIZE_RETRY_SPEC/v1') else 16
    if spec['available_result_bytes'] < minimum*2**30 or shutil.disk_usage(spec['root']).free < minimum*2**30:
        raise OSError(f'Need {minimum} GiB personally available durable headroom; no deletion')


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
    if spec.get('contract') in ('LUKA_FULLSTAT_SPLITHEAD_FINALIZE_SPEC/v1',
            'LUKA_FULLSTAT_SPLITHEAD_FINALIZE_RETRY_SPEC/v1'):
        from .finalize import check as check_finalization
        return check_finalization(spec)
    if spec.get('contract') == 'LUKA_FULLSTAT_SPLITHEAD_RECOVERY_SPEC/v1':
        from .recovery import check as check_recovery
        return check_recovery(spec)
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
