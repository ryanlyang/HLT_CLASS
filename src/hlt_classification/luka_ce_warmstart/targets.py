"""Unchanged TRAIN-only U100 banks, authenticated under their original contract."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.luka_u100_embedding.targets import export, Bank
from hlt_classification.luka_u100_embedding.contracts import validate, recipe


def validate_manifest(directory, teacher, prepared, counts):
    directory = Path(directory); m = load_json(directory/'manifest.json')
    validate(teacher, 'TEACHER')
    validate(m, 'BANK', dict(recipe=recipe()['content_hash'], prepared=prepared, teacher=teacher['content_hash']))
    if (m['role'] != 'train' or m['coordinate'] != 'U100' or m['width'] != 128
            or m['rows'] != counts['train'] or m['teacher'] != teacher['files']['checkpoint']
            or m['payload']['path'] != 'targets.npz'):
        raise ValueError('Wrong shared TRAIN bank binding')
    validate_file_ref(m['payload'], root=directory)
    return m
