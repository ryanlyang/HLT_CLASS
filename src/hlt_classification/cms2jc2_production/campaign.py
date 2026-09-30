"""Immutable production population, frozen imports and measured stage admission."""
import math
from pathlib import Path
import subprocess

from . import population as pop, storage
from .contracts import artifact, validate, load_json, write, checked, ref, safe, GIB, sha256_file
from hlt_classification.cms2jc2_response import dev_campaign as dev
from hlt_classification.cms2jc2_response import generation_direct_campaign as direct
from hlt_classification.cms2jc2_response import frozen_joint_campaign as confirmation
from hlt_classification.cms2jc2_response.generation_benchmark_data import validate_data_root
from hlt_classification.cms2jc2_response.dev_data import checked_file
from hlt_classification.cms2jc2_response.provenance import numerical_environment

EXTRA = (
    'docs/plans/CMS2JC2_PROXY_DATASET_PRODUCTION_PLAN.md',
    'docs/contracts/CMS2JC2_PROXY_DATASET_PRODUCTION.md',
    'scripts/cms2jc2_proxy_dataset.py', 'scripts/queue_cms2jc2_proxy_dataset.sh',
    'sbatch/run_cms2jc2_proxy_dataset_cpu.sh',
)
ENVIRONMENT = '/home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris'


def source_snapshot(project, commit):
    project = Path(project)
    donor = dev.source_snapshot(project, commit)
    files = dict(donor['files'])
    extra = list(EXTRA)+[p.relative_to(project).as_posix()
        for p in (project/'src/hlt_classification/cms2jc2_production').glob('*.py')]
    for name in extra:
        subprocess.run(['git', '-C', str(project), 'ls-files', '--error-unmatch', name],
                       check=True, capture_output=True)
        files[name] = sha256_file(project/name)
    return artifact('SOURCE', commit=donor['commit'], files=files, executable=True)


def imported(study, name):
    path = safe(study['root'], study['imports'][name]['relative'])
    record = study['imports'][name]
    if path.stat().st_size != record['bytes'] or sha256_file(path) != record['sha256']:
        raise ValueError('Imported production provenance changed: '+name)
    return load_json(path)


def study_kind(study):
    return ('STUDY_REDUCED_CONFIRMATION' if study.get('contract') ==
        'CMS2JC2_PROXY_STUDY_REDUCED_CONFIRMATION/v1' else 'STUDY')


def confirmation_evidence(spec, *, allow_reduced=False, full_auth=False):
    if spec.get('stage') == 'frozen_confirm':
        row = confirmation.read(spec) if full_auth else dev.product(spec, 'jf_report', 'result')
        return row, confirmation.donor(spec)['content_hash']
    from hlt_classification.cms2jc2_response import reduced_confirmation as reduced
    if spec.get('contract') != reduced.CONTRACT or not allow_reduced:
        raise PermissionError('Reduced confirmation requires explicit --allow-reduced-confirmation')
    return reduced.read(spec), reduced.donor(spec)['content_hash']


def reduced_scope(row):
    return dict(kind='reduced_36_of_57', coverage=row['coverage'],
        subset_decision=row['decision']['status'], full_population_status=row['full_population_status'])


def validate_study(study, *, source=False):
    kind = study_kind(study)
    validate(study, kind, parents={'source': study['source']['content_hash'],
        'population': study['population']['content_hash'],
        'confirmation': study['reviewed_confirmation_hash']}, test=False)
    if (study['counts'] != pop.COUNTS or study['candidate'] != 'JOINT' or study['replica'] != 0
            or study['generation_workers'] != 36 or study['max_concurrent'] != 16
            or study['physics_production_qualified'] is not False
            or study['dataset_kind'] != 'controlled_CMS_calibrated_proxy'
            or study['storage']['persistent_attested'] is not True
            or study['storage']['budget_bytes'] > study['storage']['available_quota_bytes']):
        raise ValueError('Production scope differs')
    if kind == 'STUDY_REDUCED_CONFIRMATION':
        scope = study['confirmation_scope']
        if (study.get('allow_reduced_confirmation') is not True or scope['kind'] != 'reduced_36_of_57'
                or scope['coverage']['included_indices'] != list(range(36))
                or scope['coverage']['excluded_indices'] != list(range(36, 57))
                or scope['full_population_status'] != 'inconclusive_incomplete_population'
                or study['confirmation_status'] != scope['full_population_status']):
            raise ValueError('Reduced production evidence scope differs')
    elif study.get('allow_reduced_confirmation') or 'confirmation_scope' in study:
        raise ValueError('Reduced evidence cannot masquerade as original production study')
    validate(study['source'], 'SOURCE', test=False)
    validate(study['population'], 'POPULATION', test=False)
    if study['population']['counts'] != study['counts']:
        raise ValueError('Population counts differ')
    if source and source_snapshot(study['project_dir'], study['source']['commit']) != study['source']:
        raise ValueError('Production source differs')
    return study


def location(root, persistent_parent, others):
    parent = Path(persistent_parent).resolve(strict=True)
    root = Path(root).absolute()
    if root.exists() or root.parent.resolve(strict=True) != parent or root.is_symlink():
        raise ValueError('New dataset root must be a direct child of the declared persistent directory')
    if any(p in ('tmp', 'scratch', 'local_scratch') for p in parent.parts):
        raise PermissionError('Ephemeral scratch is not a persistent destination')
    for other in others:
        other = Path(other).resolve()
        if root.resolve().is_relative_to(other) or other.is_relative_to(root.resolve()):
            raise PermissionError('Production destination overlaps inputs/worktree/old campaign')
    return root.resolve(), parent


def create(*, gate_spec, confirmation_spec, confirmation_hash, profile, project_dir,
           source_commit, root, persistent_parent, budget_gib, available_quota_gib,
           acknowledge_proxy=False, persistent_attested=False, allow_reduced_confirmation=False):
    if not acknowledge_proxy or not persistent_attested:
        raise PermissionError('Review proxy limitations and attest persistent storage explicitly')
    if not (50 <= budget_gib <= available_quota_gib and math.isfinite(available_quota_gib)):
        raise ValueError('Need at least 50 GiB bounded output budget within operator-reported available quota')
    print('CMS2JC2-PRODUCTION phase=import_metadata no_particle_reads=true', flush=True)
    gate_ref, confirm_ref = ref(gate_spec), ref(confirmation_spec)
    gs, cs = load_json(checked(gate_ref)), load_json(checked(confirm_ref))
    if gs.get('stage') != 'tigris_direct_gate':
        raise PermissionError('Require completed direct Tigris gate')
    ts = load_json(checked_file(gs['study']))
    evidence = load_json(checked_file(ts['imported']))
    original = load_json(checked_file(evidence['original_study']))
    bundle = load_json(checked_file(original['bundle']))
    gr = dev.product(gs, 'jt_gate', 'result')
    cr, comparison_hash = confirmation_evidence(cs, allow_reduced=allow_reduced_confirmation)
    if (cr['content_hash'] != confirmation_hash or cr['selected'] != 'JOINT'
            or comparison_hash != bundle['parents']['comparison']
            or not gr['compatible'] or not gr['within_site_exact']):
        raise ValueError('Reviewed confirmation/gate/frozen mapping lineage differs')
    validate_data_root(original['data_root'])
    if not Path(original['data_root']).is_dir():
        raise FileNotFoundError('Authenticated dzfix ROOT directory is not visible on this site')
    project = Path(project_dir).resolve(strict=True)
    root, parent = location(root, persistent_parent,
        [project, original['data_root'], original['root'], ts['root'], cs['root']])
    storage.check_space(parent, int(budget_gib*GIB))
    inv = load_json(checked_file(original['inventory']))
    old_profile = load_json(checked_file(original['profile']))
    new_profile = load_json(profile)
    print('CMS2JC2-PRODUCTION phase=freeze_membership jets=2250000', flush=True)
    population = pop.build(inv, new_profile, old_profile)
    shard_rows = pop.shards(population)
    print('CMS2JC2-PRODUCTION phase=authenticate_source_and_environment', flush=True)
    source = source_snapshot(project, source_commit)
    direct.source_match(source, ts['source'])
    env = numerical_environment()
    if env != ts['numerical_environment']:
        raise ValueError('Create production with the pinned Tigris numerical environment')
    root.mkdir(exist_ok=False)
    values = dict(bundle=bundle, inventory=inv, profile=new_profile, donor_profile=old_profile,
                  confirmation_report=cr, tigris_gate=gr)
    imports = {}
    for name, value in values.items():
        relative = f'provenance/{name}.json'
        record = write(safe(root, relative), value)
        imports[name] = dict(relative=relative, sha256=record['sha256'], bytes=record['bytes'])
    reduced = cs.get('stage') == 'frozen_reduced'
    extra = dict(allow_reduced_confirmation=True, confirmation_scope=reduced_scope(cr)) if reduced else {}
    study = artifact('STUDY_REDUCED_CONFIRMATION' if reduced else 'STUDY',
        parents={'source': source['content_hash'],
        'population': population['content_hash'], 'confirmation': confirmation_hash},
        project_dir=str(project), root=str(root), source=source,
        numerical_environment=env, gate_spec=gate_ref, confirmation_spec=confirm_ref,
        reviewed_confirmation_hash=confirmation_hash,
        confirmation_status=cr['full_population_status'] if reduced else cr['decision']['status'], **extra,
        imports=imports, review=original['review'], data_root=original['data_root'],
        population=population, shards=shard_rows, counts=pop.COUNTS, candidate='JOINT', replica=0,
        generation_workers=36, max_concurrent=16, native_hlt_access=False,
        dataset_kind='controlled_CMS_calibrated_proxy', physics_production_qualified=False,
        storage=dict(persistent_parent=str(parent), persistent_attested=True,
            budget_bytes=int(budget_gib*GIB), available_quota_bytes=int(available_quota_gib*GIB),
            quota_source='operator_attestation_not_filesystem_free_space', backups_assumed=False))
    validate_study(study)
    write(root/'study_spec.json', study)
    return make_attempt(study, 'pilot', shard_rows[:2], memory_gib=64, hours=4,
                        bytes_per_jet=max(4096, gr['serial']['output_bytes']/64*8))


def preflight(study):
    """Full historical check once, under a scheduled allocation."""
    gs = load_json(checked(study['gate_spec']))
    ts = direct.validate_stage(gs, source=False)
    gr = direct.accepted(gs, ts)
    original, _, bundle = direct.inputs(ts)
    cs = load_json(checked(study['confirmation_spec']))
    reduced = study_kind(study) == 'STUDY_REDUCED_CONFIRMATION'
    cr, comparison_hash = confirmation_evidence(cs, allow_reduced=reduced, full_auth=True)
    if reduced and study['confirmation_scope'] != reduced_scope(cr):
        raise ValueError('Reviewed reduced evidence scope changed')
    if (cr != imported(study, 'confirmation_report') or gr != imported(study, 'tigris_gate')
            or bundle != imported(study, 'bundle')
            or cr['content_hash'] != study['reviewed_confirmation_hash']
            or comparison_hash != bundle['parents']['comparison']
            or original['review'] != study['review'] or original['data_root'] != study['data_root']):
        raise ValueError('Production import authentication differs')
    direct.source_match(study['source'], ts['source'])
    population = pop.build(imported(study, 'inventory'), imported(study, 'profile'), imported(study, 'donor_profile'))
    if population != study['population'] or pop.shards(population) != study['shards']:
        raise ValueError('Frozen production population/partition differs')
    return artifact('PREFLIGHT', parents={'study': study['content_hash']},
        imports=study['imports'], source=study['source']['content_hash'],
        environment=study['numerical_environment']['content_hash'],
        confirmation_status=study['confirmation_status'], passed=True)


def require_preflight(study):
    value = load_json(safe(study['root'], 'preflight.json'))
    validate(value, 'PREFLIGHT', parents={'study': study['content_hash']}, test=False)
    if (value['imports'] != study['imports'] or value['passed'] is not True
            or value['source'] != study['source']['content_hash']
            or value['environment'] != study['numerical_environment']['content_hash']):
        raise ValueError('Preflight receipt differs')
    return value


def resources(pilots, max_jets=pop.SHARD_JETS):
    if len(pilots) != 2 or any(p['serial_process_exact'] is not True for p in pilots):
        raise PermissionError('Two retained production pilots must pass exact replay')
    for row in pilots:
        m, a = row['measurement'], row['allocation']
        if (row['role'] != 'train' or row['jets'] < 64 or m.get('synthetic', False)
                or m.get('samples', 0) < 1 or a.get('partition') != 'tigris'
                or a.get('cpus') != 36 or a.get('gpus') != 0 or not a.get('job_id', '').isdigit()
                or any(not math.isfinite(row[k]) or row[k] <= 0
                       for k in ('processing_seconds', 'output_bytes'))
                or not math.isfinite(row['authentication_seconds']) or row['authentication_seconds'] < 0
                or not math.isfinite(m['sampled_peak_tree_rss_bytes']) or m['sampled_peak_tree_rss_bytes'] <= 0):
            raise PermissionError('Genuine measured Tigris production pilot evidence required')
    per_jet = max(p['processing_seconds']/p['jets'] for p in pilots)
    required_seconds = 3*per_jet*max_jets + 2*max(p['authentication_seconds'] for p in pilots)+600
    required_ram = 2*max(p['measurement']['sampled_peak_tree_rss_bytes'] for p in pilots)+4*GIB
    hours = max(1, math.ceil(required_seconds/3600))
    memory = next((g for g in (32, 64, 128) if g*GIB >= required_ram), None)
    if not math.isfinite(required_seconds) or hours > 8 or memory is None:
        raise PermissionError('Measured production envelope exceeds 8h/128GiB; review resources before bulk')
    return dict(memory_gib=memory, hours=hours,
        bytes_per_jet=4*max(p['output_bytes']/p['jets'] for p in pilots))


def test_lock(study):
    row = load_json(safe(study['root'], 'test_build_lock.json'))
    validate(row, 'TEST_BUILD_LOCK', parents={'study': study['content_hash'],
        'population': study['population']['content_hash'], 'source': study['source']['content_hash'],
        'bundle': imported(study, 'bundle')['content_hash']}, test=False)
    if row['evaluate'] is not False or row['materialize_only'] is not True:
        raise PermissionError('Final-test build authorization differs')
    return row


def admission(study, pilots):
    envelope = resources(pilots)
    per_jet = max(p['processing_seconds']/p['jets'] for p in pilots)
    return artifact('ADMISSION', parents={'study': study['content_hash'],
        **{p['shard_id']: p['content_hash'] for p in pilots}}, resources=envelope,
        measured_slowest_seconds_per_jet_at_36_workers=per_jet,
        ideal_full_population_seconds_at_16_tasks=sum(study['counts'].values())*per_jet/16,
        estimate_excludes_queue_contention_and_tail_risk=True,
        pilot_shards_retained=True, physics_production_qualified=False)


def require_admission(study, attempt):
    from .recovery import validate_shard_lineage
    pilots = [load_json(safe(study['root'], f'shards/{s["shard_id"]}.json'))
              for s in study['shards'] if s['pilot']]
    for row in pilots:
        validate_shard_lineage(study, row)
    saved = load_json(safe(study['root'], 'pilot_admission.json'))
    if saved != admission(study, pilots):
        raise ValueError('Measured pilot admission changed')
    if any(attempt['resources'][k] != v for k, v in saved['resources'].items()):
        raise ValueError('Queued resources do not match measured pilot admission')
    return saved


def make_attempt(study, kind, shards, *, memory_gib, hours, bytes_per_jet, execution_repair=None):
    from .submission import plan
    from .recovery import load_repair
    extra = {}
    if execution_repair is not None:
        load_repair(study, execution_repair)
        extra = dict(execution_repair=execution_repair)
    root = Path(study['root'])
    with storage.lock(root, 'attempt'):
        index = len(list(safe(root, 'attempts').glob('*/attempt_spec.json')))
        name = f'{kind}_{index:03d}'
        path = safe(root, f'attempts/{name}')
        if path.exists():
            raise FileExistsError('Incomplete attempt directory; inspect without deleting')
        value = artifact('ATTEMPT_EXECUTION_REPAIR' if extra else 'ATTEMPT',
            parents={'study': study['content_hash']}, name=name, kind=kind, **extra,
            study=ref(root/'study_spec.json'), shards=[s['shard_id'] for s in shards],
            resources=dict(cpus=36, memory_gib=memory_gib, hours=hours,
                concurrent=min(16, len(shards)), bytes_per_jet=bytes_per_jet),
            sealed_test_lock=None if kind == 'pilot' else ref(root/'test_build_lock.json'))
        path.mkdir(parents=True)
        write(path/'attempt_spec.json', value)
        write(path/'command_plan.json', plan(study, value))
    return value


def validate_attempt(value, study):
    from .recovery import load_repair
    repaired = value.get('contract') == 'CMS2JC2_PROXY_ATTEMPT_EXECUTION_REPAIR/v1'
    validate(value, 'ATTEMPT_EXECUTION_REPAIR' if repaired else 'ATTEMPT',
             parents={'study': study['content_hash']}, test=False)
    if repaired:
        load_repair(study, value['execution_repair'])
    elif 'execution_repair' in value:
        raise ValueError('Original attempt cannot carry a source repair')
    if load_json(checked(value['study'])) != study:
        raise ValueError('Attempt study differs')
    rows = {r['shard_id']: r for r in study['shards']}
    r = value['resources']
    if (value['kind'] not in ('pilot', 'bulk', 'recovery', 'finalize')
            or (not value['shards'] and value['kind'] != 'finalize')
            or (value['kind'] == 'finalize' and value['shards'])
            or len(set(value['shards'])) != len(value['shards'])
            or any(s not in rows for s in value['shards'])
            or r['cpus'] != 36 or r['concurrent'] != min(16, len(value['shards']))
            or r['memory_gib'] not in (32, 64, 128) or not 1 <= r['hours'] <= 8
            or not math.isfinite(r['bytes_per_jet']) or r['bytes_per_jet'] <= 0):
        raise ValueError('Unregistered production attempt')
    if value['kind'] == 'pilot':
        if value['sealed_test_lock'] is not None or any(not rows[s]['pilot'] for s in value['shards']):
            raise PermissionError('Pilot escaped TRAIN pilot shards')
    elif load_json(checked(value['sealed_test_lock'])) != test_lock(study):
        raise PermissionError('Attempt test build lock differs')


def advance(study, *, authorize_test=False, recovery=False, execution_repair=None):
    from .submission import require_terminal
    from .output import completed
    from .recovery import validate_execution
    validate_execution(study, execution_repair)
    extra = dict(execution_repair=execution_repair) if execution_repair is not None else {}
    require_terminal(study)
    if safe(study['root'], 'preflight.json').exists():
        require_preflight(study)
    elif recovery:
        return make_attempt(study, 'pilot', [s for s in study['shards'] if s['pilot']],
            memory_gib=64, hours=4, **extra,
            bytes_per_jet=max(4096, imported(study, 'tigris_gate')['serial']['output_bytes']/64*8))
    else:
        raise PermissionError('Preflight did not complete; use recovery')
    done = completed(study)
    pilots = [done[s['shard_id']] for s in study['shards'] if s['pilot'] and s['shard_id'] in done]
    if len(pilots) != 2:
        if not recovery:
            raise PermissionError('Retained pilot not complete; use recovery, not bulk')
        missing = [s for s in study['shards'] if s['pilot'] and s['shard_id'] not in done]
        return make_attempt(study, 'pilot', missing, memory_gib=64, hours=4, **extra,
            bytes_per_jet=max(4096, imported(study, 'tigris_gate')['serial']['output_bytes']/64*8))
    envelope = resources(pilots)
    if not authorize_test:
        raise PermissionError('Explicit sealed test materialization authorization required')
    write(safe(study['root'], 'pilot_admission.json'), admission(study, pilots))
    lock = artifact('TEST_BUILD_LOCK', parents={'study': study['content_hash'],
        'population': study['population']['content_hash'], 'source': study['source']['content_hash'],
        'bundle': imported(study, 'bundle')['content_hash']}, materialize_only=True, evaluate=False,
        frozen_candidate='JOINT', replica=0, physics_production_qualified=False)
    write(safe(study['root'], 'test_build_lock.json'), lock)
    missing = [s for s in study['shards'] if s['shard_id'] not in done]
    if not missing:
        if safe(study['root'], 'dataset_manifest.json').exists():
            raise ValueError('Complete manifest already exists; no new jobs needed')
        return make_attempt(study, 'finalize', [], **envelope, **extra)
    pending_bytes = sum(envelope['bytes_per_jet']*s['jets']+64*2**20 for s in missing)
    reserved = sum(load_json(p)['allowance'] for p in safe(study['root'], 'reservations').glob('*.json'))
    if reserved+pending_bytes+storage.METADATA_ALLOWANCE > study['storage']['budget_bytes']:
        raise OSError('Measured build does not fit frozen output budget; no bulk submission')
    storage.check_space(study['root'], int(pending_bytes))
    return make_attempt(study, 'recovery' if recovery else 'bulk', missing, **envelope, **extra)
