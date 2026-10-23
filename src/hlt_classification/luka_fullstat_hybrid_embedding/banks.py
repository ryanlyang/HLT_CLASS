"""TRAIN-only native targets; hybrid provenance without averaging latent axes."""
from pathlib import Path
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    atomic_publish_bytes, deterministic_npz_bytes, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_embedding_aux.model import head_input
from hlt_classification.luka_embedding_aux.targets import standardize
from hlt_classification.luka_fullstat_embedding_sweep import banks as offline_banks
from .contracts import (artifact, validate, target_recipe, TEACHERS, STAGES, teachers,
                        ancestors, kd_teacher, EMBEDDING_EXPORTS)


def parents(prepared, refs, teacher_id):
    return dict(recipe=target_recipe(teacher_id)['content_hash'], prepared=prepared,
                checkpoint=refs['checkpoint']['sha256'], report=refs['report']['sha256'])


def check_selected(teacher, refs, teacher_id):
    for key in ('checkpoint', 'report'):
        validate_file_ref(refs[key])
    report = load_json(refs['report']['path'])
    if teacher_id in STAGES:
        validate(report, 'TRAINING_REPORT')
        if (report['node']['node_id'] != teacher_id or not report['scientific_fit']
                or report['technical'] or report['selected_state'] != 'native_backbone_only'):
            raise ValueError('Only selected native scientific representations may supervise')
    elif teacher_id not in ('OFFLINE', 'M0HLT'):
        raise ValueError('Unregistered original teacher')
    # Original references are authenticated in their original pinned execution
    # by campaign.check; their exact report/checkpoint refs remain bound here.
    expected = torch.load(refs['checkpoint']['path'], map_location='cpu', weights_only=True)
    if (any(k.startswith(('predictor.', 'predictors.')) for k in expected)
            or expected.keys() != teacher.state_dict().keys() or any(
                not torch.equal(v.detach().cpu(), expected[k]) for k, v in teacher.state_dict().items())):
        raise ValueError('Teacher differs from selected native checkpoint')
    del expected


def export(directory, teacher, cache, *, refs, device, teacher_id):
    if (teacher_id not in EMBEDDING_EXPORTS or cache.role != 'train'
            or cache.coordinate_name != TEACHERS[teacher_id]):
        raise PermissionError('Only registered selected TRAIN embeddings may export')
    check_selected(teacher, refs, teacher_id)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    teacher.to(device).eval().requires_grad_(False)
    values = np.empty((len(cache), 128), np.float32)
    with torch.inference_mode(), torch.autocast(torch.device(device).type, enabled=False):
        for start in range(0, len(cache), 128):
            ix = np.arange(start, min(start+128, len(cache)))
            raw = cache.batch(ix)
            tensors = {k: torch.from_numpy(raw[k]).to(device) for k in ('features', 'vectors', 'mask')}
            logits, embedding = head_input(teacher, **tensors)
            if not torch.isfinite(embedding).all() or not torch.isfinite(logits).all():
                raise ValueError('Nonfinite native teacher output')
            values[ix] = embedding.float().cpu().numpy()
            if start % (128*256) == 0:
                print(f'FULLSTAT HYBRID export teacher={teacher_id} rows={start+len(ix)}/{len(cache)}', flush=True)
    targets, mean, scale = standardize(values)
    del values
    atomic_publish_bytes(directory/'targets.npz', deterministic_npz_bytes(dict(
        identities=cache.identities, labels=cache.labels, targets=targets, mean=mean, scale=scale)))
    for key in ('checkpoint', 'report'):
        validate_file_ref(refs[key])
    record = artifact('BANK', parents=parents(cache.foundation_sha256, refs, teacher_id),
        role='train', coordinate=TEACHERS[teacher_id], teacher=teacher_id, rows=len(cache), width=128,
        checkpoint=refs['checkpoint'], report=refs['report'],
        payload=file_ref(directory/'targets.npz', root=directory))
    write_immutable_json(directory/'manifest.json', record)
    return record


def refs_for(spec, teacher):
    if teacher == 'OFFLINE':
        return spec['offline_refs']
    if teacher == 'M0HLT':
        return spec['references']['M0HLT']
    if teacher not in tuple(STAGES)[:-1]:
        raise ValueError('Unknown/final-stage teacher')
    root = Path(spec['root'])/('train_'+teacher)
    return {k: file_ref(root/f) for k, f in (('checkpoint', 'selected.pt'), ('report', 'training_report.json'))}


def manifest_ref(spec, teacher):
    return spec['offline_bank'] if teacher == 'OFFLINE' else file_ref(
        Path(spec['root'])/'targets'/teacher/'manifest.json')


def arrays(ref, cache, *, refs, teacher_id):
    """Import OFFLINE v1 unchanged, explicitly extend consumers to registered rungs.

    Only identities/labels/prepared identity are used for the cross-view join;
    no old cache is disguised as a different coordinate.
    """
    path = validate_file_ref(ref)
    m = load_json(path)
    for key in ('checkpoint', 'report'):
        validate_file_ref(refs[key])
    if teacher_id == 'OFFLINE':
        offline_banks.validate(m, 'BANK', offline_banks.parents(cache.foundation_sha256, refs))
    else:
        validate(m, 'BANK', parents(cache.foundation_sha256, refs, teacher_id))
    if (cache.role != 'train' or cache.coordinate_name not in STAGES.values()
            or any(m.get(k) != v for k, v in dict(role='train', coordinate=TEACHERS[teacher_id],
                teacher=teacher_id, rows=len(cache), width=128,
                checkpoint=refs['checkpoint'], report=refs['report']).items())
            or m['payload']['path'] != 'targets.npz'):
        raise ValueError('Embedding bank semantics or consumer role differ')
    a = load_npz_arrays(validate_file_ref(m['payload'], root=path.parent))
    if (set(a) != {'identities', 'labels', 'targets', 'mean', 'scale'}
            or a['identities'].dtype != np.uint8 or a['identities'].shape != (len(cache), 32)
            or not np.array_equal(a['identities'], cache.identities)
            or len(np.unique(a['identities'], axis=0)) != len(cache)
            or a['labels'].dtype != np.int64 or not np.array_equal(a['labels'], cache.labels)
            or a['targets'].dtype != np.float32 or a['targets'].shape != (len(cache), 128)
            or any(a[k].dtype != np.float64 or a[k].shape != (128,) for k in ('mean', 'scale'))
            or np.any(a['scale'] < 1e-3)
            or any(not np.isfinite(a[k]).all() for k in ('targets', 'mean', 'scale'))):
        raise ValueError('Embedding normalization or exact ordered TRAIN join differs')
    for value in a.values():
        value.flags.writeable = False
    return m, a


def bundle_record(spec, stage):
    order = teachers(stage)
    refs = {t: manifest_ref(spec, t) for t in order}
    # All new ancestor task receipts must have been authenticated by the worker.
    prior = {t: load_json(Path(spec['root'])/('train_'+t)/'task_receipt.json')['content_hash'] for t in ancestors(stage)}
    from . import kd_banks
    kd_ref = kd_banks.manifest_ref(spec, kd_teacher(stage))
    return artifact('BUNDLE', parents=dict(spec=spec['content_hash'], **prior), stage=stage,
        role='train', prepared=spec['anchor']['evidence']['prepared'], rows=spec['counts']['train'],
        teacher_order=order, records=refs,
        kd_teacher=kd_teacher(stage), temperature=2., kd_bank=kd_ref,
        kd_bank_hash=load_json(validate_file_ref(kd_ref))['content_hash'],
        bank_hashes={t: load_json(validate_file_ref(refs[t]))['content_hash'] for t in order})


class Bundle:
    def __init__(self, spec, stage, train):
        if train.role != 'train' or train.coordinate_name != STAGES[stage]:
            raise PermissionError('Only the registered TRAIN rung consumes these targets')
        self.manifest = bundle_record(spec, stage)
        self.identities, self.labels = train.identities, train.labels
        self.targets = {}
        for t in teachers(stage):
            _, a = arrays(self.manifest['records'][t], train, refs=refs_for(spec, t), teacher_id=t)
            self.targets[t] = a['targets']
        from . import kd_banks
        _, a = kd_banks.arrays(self.manifest['kd_bank'], train,
            refs=refs_for(spec, kd_teacher(stage)), teacher_id=kd_teacher(stage))
        self.probabilities = a['probabilities']
        check_join(self, train, stage)


def check_join(bank, train, stage):
    validate(bank.manifest, 'BUNDLE')
    if (train.role != 'train' or train.coordinate_name != STAGES[stage]
            or bank.manifest['prepared'] != train.foundation_sha256
            or bank.manifest['stage'] != stage or bank.manifest['role'] != 'train'
            or bank.manifest['rows'] != len(train) or bank.manifest['teacher_order'] != teachers(stage)
            or bank.manifest['kd_teacher'] != kd_teacher(stage) or bank.manifest['temperature'] != 2.
            or set(bank.targets) != set(teachers(stage))
            or not np.array_equal(bank.identities, train.identities)
            or not np.array_equal(bank.labels, train.labels)):
        raise ValueError('Hybrid TRAIN join differs')
    from .kd_banks import validate_probabilities
    validate_probabilities(bank.probabilities, len(train))
    for t in teachers(stage):
        a = bank.targets[t]
        if a.dtype != np.float32 or a.shape != (len(train), 128) or not np.isfinite(a).all():
            raise ValueError('Invalid targets for '+t)


def publish(spec, stage, train):
    value = Bundle(spec, stage, train)
    path = Path(spec['root'])/'bundles'/(stage+'.json')
    path.parent.mkdir(exist_ok=True)
    write_immutable_json(path, value.manifest)
    return value
