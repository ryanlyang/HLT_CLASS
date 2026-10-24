"""Reuse unchanged target semantics in newly owned banks; share across four fits."""
from pathlib import Path
from hlt_classification.luka_trial_hybrid import banks as donor
from .contracts import artifact, validate, TEACHERS


def directory(spec, teacher):
    if teacher not in TEACHERS:
        raise ValueError('Unregistered target teacher')
    return Path(spec['root'])/'targets'/'banks'/teacher


def record(spec, teacher):
    value = spec['reuse']['teachers'][teacher]
    donor.validate(value, 'TEACHER')
    coordinate = dict(OFFLINE='OFFLINE', UE100='U100', DE066='D066', DE033='D033')[teacher]
    if (value['teacher'] != teacher or value['coordinate'] != coordinate
            or value['scope'] != ('shared' if teacher == 'OFFLINE' else 'c50')
            or value['parents']['prepared'] != spec['anchor']['evidence']['prepared']):
        raise ValueError('Target teacher/branch/population differs')
    return value


def export(spec, teacher, native, cache, *, device):
    # The unchanged BANK/v1 describes the exact same native target science,
    # not an artifact claimed to have been produced in the old directory.
    return donor.export(directory(spec, teacher), native, cache, record(spec, teacher), device=device)


def manifests(spec, *, payloads=True):
    return {t: donor.manifest(directory(spec, t), record(spec, t),
        spec['anchor']['evidence']['prepared'], spec['counts']['train'], payload=payloads) for t in TEACHERS}


def descriptor(spec, records):
    return artifact('BUNDLE', parents=dict(spec=spec['content_hash'], **{t: records[t]['content_hash'] for t in TEACHERS}),
                    teacher_order=list(TEACHERS), banks=records)


class Bundle:
    def __init__(self, spec, cache):
        if cache.role != 'train' or cache.coordinate_name != 'D000':
            raise PermissionError('Only the HLT TRAIN role consumes target bundles')
        self.identities, self.labels = cache.identities, cache.labels
        self.targets, self.probabilities, records = {}, {}, {}
        for t in TEACHERS:
            records[t], arrays = donor.load(directory(spec, t), cache, record(spec, t))
            self.targets[t], self.probabilities[t] = arrays['targets'], arrays['probabilities']
        self.manifest = descriptor(spec, records)


def validate_bundle(value, spec, *, payloads=True):
    expected = descriptor(spec, manifests(spec, payloads=payloads))
    validate(value, 'BUNDLE', expected['parents'])
    if value != expected:
        raise ValueError('Frozen multi-teacher target bundle differs')
