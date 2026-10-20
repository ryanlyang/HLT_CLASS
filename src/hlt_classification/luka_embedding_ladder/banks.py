"""Compact temperature-two TRAIN targets live in RAM, never attention banks."""
from types import SimpleNamespace
import hashlib
import numpy as np
from hlt_classification.jetclass2_delphes.runner import predict
from .contracts import artifact, validate, VIEWS


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def export(model, cache, *, teacher, checkpoint, report, device, technical):
    if cache.role != 'train' or cache.coordinate_name not in VIEWS:
        raise PermissionError('Logit targets require registered TRAIN views')
    model.eval()
    q = predict(model, cache, device=device, temperature=2., batch_size=128)
    manifest = artifact('LOGITS', parents=dict(checkpoint=checkpoint, report=report,
        prepared=cache.foundation_sha256), teacher=teacher, coordinate=cache.coordinate_name,
        role='train', temperature=2., rows=len(cache), technical=technical,
        probabilities_sha256=digest(q), identities_sha256=digest(cache.identities), labels_sha256=digest(cache.labels))
    result = SimpleNamespace(manifest=manifest, probabilities=q, identities=cache.identities, labels=cache.labels)
    check(result, cache)
    return result


def check(bank, cache, node=None, *, technical=None):
    m, q = bank.manifest, bank.probabilities
    validate(m, 'LOGITS')
    if (cache.role != 'train' or m['role'] != 'train' or m['temperature'] != 2.
            or m['coordinate'] not in VIEWS or m['rows'] != len(cache)
            or m['parents']['prepared'] != cache.foundation_sha256
            or q.dtype != np.float32 or q.shape != (len(cache), 11)
            or not np.isfinite(q).all() or np.any(q < 0)
            or not np.allclose(q.sum(-1), 1., atol=2e-6, rtol=0)
            or not np.array_equal(bank.identities, cache.identities)
            or not np.array_equal(bank.labels, cache.labels)
            or m['probabilities_sha256'] != digest(q)
            or m['identities_sha256'] != digest(cache.identities) or m['labels_sha256'] != digest(cache.labels)):
        raise ValueError('Logit bank population, bytes or temperature differs')
    if node and (m['teacher'] != node['teacher'] or m['coordinate'] != node['teacher_coordinate']):
        raise ValueError('Not the immediately preceding teacher in this arm')
    if technical is not None and m['technical'] is not technical:
        raise ValueError('Technical/scientific teacher contamination')
