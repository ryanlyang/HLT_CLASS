"""Recover the 29 unaffected miniatures/profiles from failed setup 165857.

Historical artifacts retain their original spec and paths. They are never
relabeled as new fits or scientific weights. TP and current-runtime parity
are requalified; an old acceptance is neither required nor fabricated.
"""
from pathlib import Path
import sys
import numpy as np
from hlt_classification.data.cache_contracts import load_json, validate_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from .contracts import artifact, validate, recipe
from .registry import FITS
from . import recovery, inputs, training, qualification

SOURCE = '27a8341210ca686b74d1c538a3ed258bb7fe947d'
REUSED = tuple(t for t in FITS if t != 'TP')


def source_spec(path):
    path = Path(path).resolve(strict=True); old = load_json(path)
    validate_content_hash(old, expected_contract='LUKA_EMBEDDING_BRIDGE_HYDRA_SPEC/v5', expected_schema_version=5)
    if (old['source_commit'] != SOURCE or path != Path(old['root'])/'campaign_spec.json'
            or old['recipe'] != recipe() or Path(sys.prefix).resolve() != Path(old['environment_dir'])):
        raise ValueError('Only the pinned completed v5 miniatures can be recovered')
    return old


def inspect(path):
    from . import resources
    print('EMBEDDING_BRIDGE qualification recovery: authenticating completed miniatures', flush=True)
    old = source_spec(path); job = recovery.terminal_jobs(old, source=SOURCE)
    root = Path(old['root'])/'setup'
    if any((root/name).exists() for name in ('receipt.json', 'acceptance.json', 'cpu_resources.json')):
        raise ValueError('Expected the completed-measurement, time-admission failure')
    diagnostic = load_json(root/'resource_diagnostic.json')
    minutes = resources.profile_minutes(old, diagnostic, historical=True)
    over = [t for t in FITS if minutes[t] > old['fit_max_minutes']]
    if over != ['TP']:
        raise ValueError('The historical resource failure is not TP alone')
    qualification.check_replay(old)
    ds = inputs.Dataset(old); ds.u100('train'); ds.u100('validation')
    preparation = load_json(root/'preparation.json')
    validate(preparation, 'PREPARATION', dict(spec=old['content_hash']))
    if preparation['data'] != ds.manifest['content_hash']:
        raise ValueError('Historical data completion differs')
    population = load_json(root/'mini_population.json')
    validate(population, 'MINI_POPULATION', dict(spec=old['content_hash']))
    selected = resources.selected_indices(ds, wide=True)
    if population['indices'] != {k:v.tolist() for k,v in selected.items()} or population['technical'] is not True:
        raise ValueError('Historical miniature rows differ')
    small = inputs.SubsetDataset(ds, selected)
    for task in FITS:
        target = resources.technical_target(old, task, small, root)
        report, _ = training.validate_fit(root/('mini_'+task), old, task, technical=True)
        if (report['parents'] != dict(spec=old['content_hash'], recipe=recipe()['content_hash'], **target.parents)
                or report['world_size'] != 2):
            raise ValueError('Historical miniature target lineage differs')
    # Pin every historical artifact used later, including banks and epoch files.
    # No duplicate payloads or historical model weights are copied into science.
    names = ['resource_diagnostic.json', 'preparation.json', 'mini_population.json',
             'parity.json', 'replay.json', 'replay_predictions.npz', 'u100_replay.json']
    paths = [root/n for n in names]
    for task in FITS:
        paths.extend(p for p in (root/('mini_'+task)).rglob('*') if p.is_file())
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Redirected historical setup member')
    refs = [file_ref(p, root=root) for p in sorted(paths)]
    return artifact('QUALIFICATION_REUSE', parents=dict(original_spec=old['content_hash'],
        diagnostic=diagnostic['content_hash']), source_spec=file_ref(path), source_commit=SOURCE,
        failed_setup_job=job, reused=list(REUSED), remeasure=['TP'], files=refs,
        method='historical_miniatures_and_conservative_profiles_original_lineage',
        scientific_weights_reused=False)


def check(value, spec, *, full=False):
    validate(value, 'QUALIFICATION_REUSE')
    old = source_spec(validate_file_ref(value['source_spec']))
    recovery.compatible(old, spec)
    if (value['source_commit'] != SOURCE or value['parents']['original_spec'] != old['content_hash']
            or value['reused'] != list(REUSED) or value['remeasure'] != ['TP']
            or value['method'] != 'historical_miniatures_and_conservative_profiles_original_lineage'
            or value['scientific_weights_reused'] is not False):
        raise ValueError('Qualification recovery policy differs')
    root = Path(old['root'])/'setup'
    names = [r['path'] for r in value['files']]
    if len(set(names)) != len(names) or 'resource_diagnostic.json' not in names:
        raise ValueError('Qualification recovery inventory differs')
    for ref in value['files']:
        from .storage import regular
        regular(root, ref['path'])
        if full or ref['path'] == 'resource_diagnostic.json':validate_file_ref(ref, root=root)
    if load_json(root/'resource_diagnostic.json')['content_hash'] != value['parents']['diagnostic']:
        raise ValueError('Recovered measurements changed')
    return old


def miniature(spec, task):
    """Location and owner of a technical fit, never a scientific fit."""
    value = spec.get('qualification_reuse')
    if value is not None and task in REUSED:
        owner = source_spec(validate_file_ref(value['source_spec']))
        return Path(owner['root'])/'setup'/('mini_'+task), owner
    return Path(spec['root'])/'setup'/('mini_'+task), spec


def profiles(spec, ds, devices):
    value = spec.get('qualification_reuse')
    if value is None:return {}
    old = check(value, spec, full=True)
    # Verify terminal jobs again inside the new setup allocation.
    if recovery.terminal_jobs(old, source=SOURCE) != value['failed_setup_job']:
        raise ValueError('Historical job ownership changed')
    diagnostic = load_json(Path(old['root'])/'setup/resource_diagnostic.json')
    from . import execution
    execution.check_devices(diagnostic['devices'],'1080ti_2_fp32')
    execution.check_devices(devices,execution.TP_PROFILE)
    original = inputs.Dataset(old)
    for role in ('train', 'validation'):
        for coord in old['recipe']['coordinates']:
            before = original.view(role, coord).store.manifest['arrays']
            after = ds.view(role, coord).store.manifest['arrays']
            if before != after:raise ValueError('Recovered model-input bytes differ')
        if original.physical(role).manifest['arrays'] != ds.physical(role).manifest['arrays']:
            raise ValueError('Recovered physical inputs differ')
    for key in ('identities', 'labels', 'targets', 'mean', 'scale'):
        if not np.array_equal(original.u100('train')[key], ds.u100('train')[key]):
            raise ValueError('Recovered original U100 targets differ')
    return {t:diagnostic['profiles'][t] for t in REUSED}


def profile_devices(spec):
    old=check(spec['qualification_reuse'],spec)
    return load_json(Path(old['root'])/'setup/resource_diagnostic.json')['devices']
