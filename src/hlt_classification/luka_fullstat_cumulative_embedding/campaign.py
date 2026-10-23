"""Four scientific jobs; bounded acceptance reuse, never a standalone preflight."""
from pathlib import Path
import shutil
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.luka_fullstat_multiteacher.campaign import protected
from . import source
from .contracts import artifact, validate, recipe, node, tasks, STAGES, COUNTS


def policy(reuse):
    validate(reuse, 'REUSE')
    if (reuse['counts'] != COUNTS or 'L40S' not in reuse['gpu']['name']
            or set(reuse['references']) != {'M0HLT', 'OFFLINE', 'U100_EMBED_W5', 'MULTI4_EMBED_W5'}
            or any(type(reuse[k]) is not int or not 1 <= reuse[k] <= 2880
                   for k in ('multi_train_minutes', 'original_train_minutes'))):
        raise ValueError('Inapplicable Oscar four-teacher/native resource evidence')
    preflight = load_json(validate_file_ref(reuse['original_preflight']))
    setup = load_json(validate_file_ref(reuse['multi_setup']))
    if (preflight['technical_checks_passed'] is not True or setup['technical_checks_passed'] is not True
            or preflight['recipe']['batch_size'] != 128
            or preflight['gpu'] != reuse['gpu'] or setup['gpu'] != reuse['gpu']
            or preflight['environment'] != reuse['environment'] or setup['environment'] != reuse['environment']
            or preflight['train_minutes'] != reuse['original_train_minutes']
            or not set(STAGES.values()) <= {m['coordinate'] for m in preflight['measurements']}):
        raise ValueError('Original measured coordinates/environment differ')
    minutes = max(reuse['multi_train_minutes'], reuse['original_train_minutes'])+120
    if minutes > 2880:
        raise ValueError('Inherited request plus export allowance exceeds 48h; explicit new envelope required')
    # Conservative resident bound: two copies of caches for load peaks, four
    # targets plus one raw export/FP64 moment workspace, and 16 GiB process reserve.
    for co in STAGES.values():
        rows = sum(reuse['summaries'][role][co]['resident_bytes'] for role in ('train', 'validation'))
        if 2*rows + COUNTS['train']*128*8*8 + 16*2**30 >= .85*180000*1024**2:
            raise MemoryError('Registered cache/bank resident envelope does not fit')
    return dict(anchor=reuse['anchor'], references=reuse['references'], offline_refs=reuse['offline_refs'],
        offline_bank=reuse['offline_bank'], counts=dict(COUNTS), nodes={s: node(reuse['base_node'], s) for s in STAGES},
        recipes={s: recipe(s) for s in STAGES}, tasks=tasks(), execution_site=reuse['execution_site'],
        cpus=6, memory_mb=180000, train_minutes=minutes, preflight_jobs=0, setup_jobs=0,
        new_scientific_fits=4, automatic_dependency=True, automatic_retry=False,
        resource_policy='inherited_original_and_multi4_max_request_plus120min_no_new_timing_claim',
        startup=dict(stress_steps=3, technical_epochs=0, benchmark_steps=0, full_validation_pass=False))


def storage(spec):
    if spec['available_result_bytes'] < 12*2**30 or shutil.disk_usage(spec['root']).free < 12*2**30:
        raise OSError('Need 12 GiB personally available durable headroom; no deletion')


def create(root, previous_spec, project, commit, *, quota_gib):
    root, project = Path(root).resolve(), Path(project).resolve()
    active = source.current(project, commit)
    reuse = source.authenticate(previous_spec, project)
    protected(root, reuse, project)
    if type(quota_gib) is not int or quota_gib < 12:
        raise ValueError('Declare at least 12 GiB personal durable headroom')
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
            or type(spec['available_result_bytes']) is not int or spec['available_result_bytes'] < 12*2**30):
        raise ValueError('Source, bank, recipe or reference registration changed')
    protected(spec['root'], reuse, spec['project_dir'])
    return reuse
