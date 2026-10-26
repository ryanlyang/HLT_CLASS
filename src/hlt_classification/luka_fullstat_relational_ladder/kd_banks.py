"""Frozen native TRAIN probabilities at T=2; never embeddings or head predictions."""
from pathlib import Path
import numpy as np
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    atomic_publish_bytes, deterministic_npz_bytes, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.luka_fullstat_hybrid_embedding import kd_banks as inherited_kd
from .contracts import artifact, validate, kd_recipe, KD_EXPORTS, TEACHERS, STAGES
from .banks import check_selected


def parents(prepared, refs, teacher_id):
    return dict(recipe=kd_recipe(teacher_id)['content_hash'], prepared=prepared,
                checkpoint=refs['checkpoint']['sha256'], report=refs['report']['sha256'])


def validate_probabilities(p, rows):
    if (p.dtype != np.float32 or p.shape != (rows, 11) or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(1), 1., rtol=0, atol=2e-6)):
        raise ValueError('Invalid normalized temperature-two probability targets')


def manifest_ref(spec, teacher):
    if teacher in ('OFFLINE', 'M0HLT'):
        return spec['imported_banks'][teacher]['kd']
    if teacher not in KD_EXPORTS:
        raise ValueError('Unregistered KD export')
    return file_ref(Path(spec['root'])/'logit_targets'/teacher/'manifest.json')


def export(directory, teacher, cache, *, refs, device, teacher_id):
    if (teacher_id not in KD_EXPORTS or cache.role != 'train'
            or cache.coordinate_name != TEACHERS[teacher_id]):
        raise PermissionError('Only registered native TRAIN logits may export')
    check_selected(teacher, refs, teacher_id)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    teacher.to(device).eval().requires_grad_(False)
    # Identical temperature/precision path to the original FullStat logit reducers.
    p = predict(teacher, cache, device=device, temperature=2., batch_size=128)
    validate_probabilities(p, len(cache))
    atomic_publish_bytes(directory/'probabilities.npz', deterministic_npz_bytes(dict(
        identities=cache.identities, labels=cache.labels, probabilities=p)))
    for key in ('checkpoint', 'report'):
        validate_file_ref(refs[key])
    record = artifact('KD_BANK', parents=parents(cache.foundation_sha256, refs, teacher_id),
        role='train', coordinate=TEACHERS[teacher_id], teacher=teacher_id, rows=len(cache),
        temperature=2., class_count=11, checkpoint=refs['checkpoint'], report=refs['report'],
        payload=file_ref(directory/'probabilities.npz', root=directory))
    write_immutable_json(directory/'manifest.json', record)
    _, restored = arrays(file_ref(directory/'manifest.json'), cache, refs=refs,
                         teacher_id=teacher_id, exporting=True)
    if not np.array_equal(restored['probabilities'], p):
        raise ValueError('KD probability-bank roundtrip differs')
    return record


def arrays(ref, cache, *, refs, teacher_id, exporting=False):
    if teacher_id == 'OFFLINE':
        if exporting:
            raise ValueError('Imported teachers must not be re-exported')
        return inherited_kd.arrays(ref, cache, refs=refs, teacher_id=teacher_id)
    path = validate_file_ref(ref)
    m = load_json(path)
    for key in ('checkpoint', 'report'):
        validate_file_ref(refs[key])
    if teacher_id == 'M0HLT':
        if exporting:
            raise ValueError('Imported M0HLT must not be re-exported')
        from hlt_classification.luka_fullstat_splithead import kd_banks as original
        original.validate(m, 'KD_BANK', original.parents(cache.foundation_sha256, refs, teacher_id))
    else:
        validate(m, 'KD_BANK', parents(cache.foundation_sha256, refs, teacher_id))
    allowed = (TEACHERS[teacher_id],) if exporting else tuple(STAGES.values())
    expected = dict(role='train', coordinate=TEACHERS[teacher_id], teacher=teacher_id,
        rows=len(cache), temperature=2., class_count=11, checkpoint=refs['checkpoint'], report=refs['report'])
    if (cache.role != 'train' or cache.coordinate_name not in allowed
            or any(m.get(k) != v for k, v in expected.items())
            or m['payload']['path'] != 'probabilities.npz'):
        raise ValueError('KD teacher, temperature, role or coordinate differs')
    a = load_npz_arrays(validate_file_ref(m['payload'], root=path.parent))
    if (set(a) != {'identities', 'labels', 'probabilities'}
            or a['identities'].dtype != np.uint8 or a['identities'].shape != (len(cache), 32)
            or not np.array_equal(a['identities'], cache.identities)
            or len(np.unique(a['identities'], axis=0)) != len(cache)
            or a['labels'].dtype != np.int64 or not np.array_equal(a['labels'], cache.labels)):
        raise ValueError('KD exact ordered TRAIN identity/label join differs')
    validate_probabilities(a['probabilities'], len(cache))
    for value in a.values():
        value.flags.writeable = False
    return m, a
