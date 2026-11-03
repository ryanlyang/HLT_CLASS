"""Reuse verified Brown HLT/U100 inputs and unchanged SPORC TRAIN keys."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle, load_view, embedding_bank
from hlt_classification.luka_u100_contrastive.targets import KeyBank
from .transfer import Delta


def delta(spec):
    return Delta(spec['delta_root'], spec['delta_hash'])


def key_bank(spec, train):
    d = delta(spec); owner = d.manifest['key_owner']; old = d.manifest['origin_spec']
    d.path('keys/manifest.json'); d.path('keys/keys.npz')
    return KeyBank(d.root/'keys', train, old['anchor']['evidence']['models']['OFFLINE']['checkpoint'],
        spec_hash=owner['content_hash'], imported_hash=old['imported']['content_hash'])


def bind(spec, d):
    """Metadata-only joins, including original U100 bytes; no historical paths."""
    old = d.manifest['origin_spec']; b = Bundle(spec['bundle_root'])
    transferred = b.json('origin/matched_spec.json')
    if old != transferred or old['imported'] != spec['imported'] or old['anchor'] != spec['anchor']:
        raise ValueError('Contrastive delta and existing Brown inputs have different origin')
    key = load_json(d.path('keys/manifest.json'))
    proof = d.manifest['original_replay']
    key_refs = [r for r in proof['setup_files'] if Path(r['path']).parts[-2:] == ('OFFLINE', 'manifest.json')]
    if not key_refs or not any(r['sha256'] == d.rows['keys/manifest.json']['sha256'] for r in key_refs):
        raise ValueError('OFFLINE manifest is not from authenticated setup')
    u100 = b.json('targets/u100/manifest.json')
    source_u100 = [r for r in proof['setup_files'] if r['sha256'] == b.rows['targets/u100/manifest.json']['sha256']]
    if not source_u100 or u100['parents']['prepared'] != old['anchor']['evidence']['prepared']:
        raise ValueError('U100 target bank differs')
    if (key['parents']['spec'] != d.manifest['key_owner']['content_hash']
            or key['parents']['imported'] != old['imported']['content_hash']
            or key['parents']['prepared'] != old['anchor']['evidence']['prepared']
            or any(key['payload'][k] != d.rows['keys/keys.npz'][k] for k in ('bytes', 'sha256'))):
        raise ValueError('Key bank ownership/payload differs')
