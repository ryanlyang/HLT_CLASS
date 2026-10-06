"""Freeze the proven recipe once; register one complete generation DAG."""
import math
from pathlib import Path
import re

from hlt_classification.literature_context import campaign as pilot, worker
from hlt_classification.literature_proxy_v3 import inputs
from hlt_classification.literature_context.kernel import recipe
from hlt_classification.literature_proxy_v3.contracts import read_reference
from hlt_classification.literature_proxy_v2.kernel import validate_calibration
from . import population as pop, storage
from .contracts import artifact, validate, write, load_json, checked, ref, sha256_file, safe, GIB

PROJECT = Path(__file__).resolve().parents[3]
ENVIRONMENT = '/home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris'
APPROVED_PILOT_COMMIT = '6cb5f32ab6e8d179f846f0cfeb4511ef1a0b6ab4'
APPROVED_PILOT_HASH = '5e4fcd1297606421c957644ecbd4d06ff1ff1f01dd44f7a59309b6d998270f60'
RESOURCES = dict(cpus=36, memory_gib=64, hours=2, concurrent=16,
                 partition='tigris', account='reu-aisocial', gpus=0)
EXTRA = ('scripts/jetclass2_literature_context_dataset.py',
    'scripts/queue_jetclass2_literature_context_dataset.sh',
    'sbatch/run_jetclass2_literature_context_dataset.sh',
    'docs/plans/JETCLASS2_LITERATURE_CONTEXT_PRODUCTION_PLAN.md',
    'docs/contracts/JETCLASS2_LITERATURE_CONTEXT_PRODUCTION.md')


def location(root, persistent_parent, others):
    parent = Path(persistent_parent).resolve(strict=True)
    root = Path(root).absolute()
    if root.exists() or root.parent.resolve(strict=True) != parent or root.is_symlink():
        raise ValueError('New dataset must be a direct child of the declared persistent directory')
    if any(p in ('tmp', 'scratch', 'local_scratch') for p in parent.parts):
        raise PermissionError('Ephemeral scratch is not a persistent destination')
    for other in others:
        other = Path(other).resolve()
        if root.resolve().is_relative_to(other) or other.is_relative_to(root.resolve()):
            raise PermissionError('Production output overlaps source data/worktree/parent campaign')
    return root.resolve(), parent


def source(project, commit, *, pushed=False):
    result = pilot.source(project, commit, pushed=pushed)
    project = Path(project)
    required = list(EXTRA)
    # Pin reused contracts/readers as well as our direct implementation.
    for folder in ('literature_context_production', 'literature_proxy_production', 'cms2jc2_production', 'cms2jc2_response', 'jetclass2_delphes'):
        required += sorted(p.relative_to(project).as_posix()
            for p in (project/'src/hlt_classification'/folder).glob('*.py'))
    pilot.git(project, 'ls-files', '--error-unmatch', '--', *required)
    result['file_sha256'].update({p: sha256_file(project/p) for p in required})
    return result


def authenticate_pilot(path):
    spec = load_json(path)
    pilot.validate(spec, 'SPEC')
    if spec['source']['commit'] != APPROVED_PILOT_COMMIT or spec['content_hash'] != APPROVED_PILOT_HASH:
        raise PermissionError('Only the explicitly approved context pilot may be frozen')
    parent, rates, records = inputs.authenticate(spec['parent_spec'], spec['parent_receipt'], spec['calibration_ref'])
    if (spec['recipe'] != recipe() or spec['calibration'] != rates or spec['inputs'] != records
            or spec['population'] != parent['population']):
        raise ValueError('Frozen context pilot lineage/recipe differs')
    report = worker.results(spec)
    v1 = read_reference(parent['parent_spec'])
    return spec, report, v1


def require_frozen_code(current, original):
    """Execution plumbing may change; the imported scientific implementation may not."""
    names = set(current['file_sha256']) | set(original['file_sha256'])
    for name in names:
        scientific = ((name.startswith('src/hlt_classification/literature_proxy')
                       or name.startswith('src/hlt_classification/literature_context/'))
                      and not name.startswith('src/hlt_classification/literature_proxy_production/'))
        scientific |= name in ('src/hlt_classification/cms_proxy_ladder/inputs.py',
                               'src/hlt_classification/cms_proxy_ladder/contracts.py',
                               'src/hlt_classification/jetclass2_delphes/inputs.py')
        if scientific or name == 'src/hlt_classification/cms2jc2_response/bridge.py':
            if current['file_sha256'].get(name) != original['file_sha256'].get(name):
                raise ValueError('Frozen pilot implementation changed: '+name)


def bundle(study):
    value = load_json(checked(study['bundle']))
    validate(value, 'BUNDLE', test=False)
    if (value['recipe'] != recipe() or value['input_contract'] != recipe()['inputs']
            or value['parents']['pilot'] != study['pilot_hash']
            or study['pilot_hash'] != APPROVED_PILOT_HASH
            or value['pilot_source']['commit'] != APPROVED_PILOT_COMMIT):
        raise ValueError('Frozen production recipe differs')
    validate_calibration(value['calibration'], value['calibration_parent'])
    require_frozen_code(value['source'], value['pilot_source'])
    return value


def validate_study(study, *, authenticate_source=False):
    validate(study, 'STUDY', parents=dict(pilot=study['pilot_hash'],
        population=study['population']['content_hash']), test=False)
    pop.validate_population(study['population'], 'POPULATION', test=False)
    pop.validate_shards(study['population'], study['shards'])
    value = bundle(study)
    if (study['resources'] != RESOURCES or study['counts'] != pop.COUNTS
            or study['population']['counts'] != pop.COUNTS
            or study['kind'] != 'controlled_context_synthetic_proxy'
            or study['input_contract'] != value['input_contract']
            or study['physics_production_qualified'] is not False
            or study['storage']['persistent_attested'] is not True
            or not 4*GIB <= study['storage']['budget_bytes'] <= study['storage']['available_quota_bytes']-storage.FREE_HEADROOM
            or Path(study['project_dir']).resolve() != PROJECT.resolve()):
        raise ValueError('Frozen production scope/resources/storage differ')
    if value['source'] != study['source']:
        raise ValueError('Bundle source differs')
    if authenticate_source and source(study['project_dir'], study['source']['commit']) != study['source']:
        raise ValueError('Production source bytes differ')
    return study


def test_lock(study):
    expected = artifact('TEST_BUILD_LOCK', parents=dict(study=study['content_hash'],
        population=study['population']['content_hash']), materialize=True, evaluate=False,
        purpose='sealed_deterministic_materialization_only')
    actual = load_json(safe(study['root'], 'test_build_lock.json'))
    if actual != expected:
        raise PermissionError('Test materialization lock differs')
    return actual


def create(*, project, commit, parent_spec, profile, root, persistent_parent,
           available_quota_gib, budget_gib=20, persistent_attested=False, acknowledge_synthetic=False):
    if not persistent_attested or not acknowledge_synthetic:
        raise PermissionError('Explicit persistent storage and synthetic-data acknowledgement required')
    if not (math.isfinite(available_quota_gib) and math.isfinite(budget_gib)
            and 4 <= budget_gib <= available_quota_gib-2):
        raise ValueError('Output budget needs available quota plus two GiB headroom')
    project = Path(project).resolve()
    if project != PROJECT.resolve():
        raise ValueError('Run in requested pinned project')
    src = source(project, commit, pushed=True)
    print('JC2-CTX-PROD phase=authenticate_pilot training_only=true', flush=True)
    spec, report, v1 = authenticate_pilot(parent_spec)
    require_frozen_code(src, spec['source'])
    root, parent = location(root, persistent_parent,
        [project, v1['data_root'], spec['root'], Path(spec['parent_spec']['path']).parent,
         Path(v1['root'])])
    storage.check_space(parent, int(budget_gib*GIB))
    inventory, donor = (read_reference(v1[k]) for k in ('inventory', 'profile'))
    profile_ref = ref(profile)
    population = pop.build(inventory, load_json(checked(profile_ref)), donor)
    frozen = artifact('BUNDLE', parents=dict(pilot=spec['content_hash']),
        pilot=ref(parent_spec), pilot_receipt=ref(Path(spec['root'])/'receipt.json'),
        pilot_report=ref(Path(spec['root'])/'report.json'), recipe=recipe(), input_contract=recipe()['inputs'],
        calibration=spec['calibration'], calibration_parent=read_reference(spec['parent_spec'])['content_hash'],
        environment={k: report['environment'][k] for k in ('python', 'machine', 'packages')},
        source=src, pilot_source=spec['source'])
    bundle_ref = write(root/'frozen_bundle.json', frozen)
    study = artifact('STUDY', parents=dict(pilot=spec['content_hash'], population=population['content_hash']),
        pilot_hash=spec['content_hash'], bundle=bundle_ref, source=src, input_contract=recipe()['inputs'],
        root=str(root), project_dir=str(project), data_root=v1['data_root'],
        inventory=v1['inventory'], donor_profile=v1['profile'], profile=profile_ref,
        population=population, shards=pop.shards(population), counts=dict(pop.COUNTS), resources=dict(RESOURCES),
        kind='controlled_context_synthetic_proxy', physics_production_qualified=False,
        storage=dict(persistent_attested=True, persistent_parent=str(parent),
            available_quota_bytes=int(available_quota_gib*GIB), budget_bytes=int(budget_gib*GIB)))
    write(root/'study_spec.json', study)
    write(root/'test_build_lock.json', artifact('TEST_BUILD_LOCK', parents=dict(study=study['content_hash'],
        population=population['content_hash']), materialize=True, evaluate=False,
        purpose='sealed_deterministic_materialization_only'))
    create_attempt(study, 'initial')
    return study


def validate_attempt(attempt, study):
    validate(attempt, 'ATTEMPT', parents={'study': study['content_hash']}, test=False)
    if (not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', attempt['name'])
            or attempt['resources'] != RESOURCES or attempt['kind'] not in ('initial', 'recovery')
            or load_json(checked(attempt['study'])) != study
            or len(attempt['shards']) != len(set(attempt['shards']))
            or any(s not in {r['shard_id'] for r in study['shards']} for s in attempt['shards'])):
        raise ValueError('Attempt registration differs')
    retained = attempt['retained_shards']
    if attempt['kind'] == 'initial' and retained:
        raise ValueError('Initial attempt cannot omit shards')
    known = {s['shard_id'] for s in study['shards']}
    if not set(retained) <= known:
        raise ValueError('Unknown retained shard')
    for shard_id, reference in retained.items():
        row = load_json(checked(reference))
        if row['parents']['study'] != study['content_hash'] or row['shard_id'] != shard_id:
            raise ValueError('Retained shard belongs to another population')
    if attempt['shards'] != [s['shard_id'] for s in study['shards'] if s['shard_id'] not in retained]:
        raise ValueError('Attempt must cover all and only missing shards')


def create_attempt(study, name, *, recovery=False):
    from . import output, submission
    validate_study(study)
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', name):
        raise ValueError('Invalid attempt name')
    root = Path(study['root'])
    with storage.lock(root, 'attempt_creation'):
        if recovery:
            submission.require_terminal(study)
            done = output.completed(study)
        else:
            if list((root/'attempts').glob('*/attempt_spec.json')):
                raise PermissionError('Use explicit recovery for an existing study')
            done = {}
        path = safe(root, f'attempts/{name}/attempt_spec.json')
        if path.parent.exists():
            raise FileExistsError('New attempt name required; nothing removed')
        attempt = artifact('ATTEMPT', parents={'study': study['content_hash']},
            name=name, kind='recovery' if recovery else 'initial', resources=dict(RESOURCES),
            retained_shards={s: ref(safe(root, f'shards/{s}.json')) for s in done},
            study=ref(root/'study_spec.json'), shards=[r['shard_id'] for r in study['shards'] if r['shard_id'] not in done])
        write(path, attempt)
        write(path.parent/'command_plan.json', submission.plan(study, attempt))
    return path


def require_preflight(study, attempt):
    result = load_json(safe(study['root'], f'attempts/{attempt["name"]}/preflight.json'))
    validate(result, 'PREFLIGHT', parents={'study': study['content_hash'], 'attempt': attempt['content_hash']}, test=False)
    if not result['exact_replay'] or not result['resource_envelope_ok']:
        raise PermissionError('Production preflight has not passed')
    return result
