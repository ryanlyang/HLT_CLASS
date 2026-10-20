"""TRAIN-only compact factors, immutable export and one resident load per fit."""
from pathlib import Path
from types import SimpleNamespace
from io import BytesIO
import numpy as np
import torch
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import (load_json, write_immutable_json,
    atomic_publish_bytes, deterministic_npz_bytes, load_npz_arrays)
from hlt_classification.jetclass2_delphes.banks import load_bank
from hlt_classification.luka_gating import banks as shared
from hlt_classification.luka_gating_cold.training import new_student
from .contracts import artifact, validate, TEACHER


def original(gate, train):
    require_train(train)
    ref = gate['anchor']['logits']
    directory = Path(validate_file_ref(ref)).parent
    report = load_json(validate_file_ref(gate['anchor']['evidence']['models'][TEACHER]['report']))
    p = load_bank(directory, foundation_sha256=train.foundation_sha256,
        teacher_report_sha256=report['content_hash'], teacher_node=TEACHER,
        role='train', expected_identities=train.identities)
    p.flags.writeable = False
    return SimpleNamespace(probabilities=p, identities=train.identities, labels=train.labels,
        manifest=artifact('SUPERVISION', parents=dict(logits=load_json(directory/'manifest.json')['content_hash']),
            alignment=shared.alignment(train), teacher=TEACHER, coordinate='D033', technical=False))


def require_train(cache):
    if cache.role != 'train' or cache.coordinate_name != 'D000':
        raise PermissionError('Only exact D000 TRAIN consumes targets')


def export(directory, gate, train, context, fit_directory, *, technical, device):
    from .worker import validate_fit
    require_train(train)
    shared.paired(train, context)
    if context is None or context.role != 'train' or context.coordinate_name != 'D033':
        raise PermissionError('Privileged export requires aligned TRAIN D033')
    fit_directory, directory = Path(fit_directory), Path(directory)
    fit = load_json(fit_directory/'receipt.json')
    node = gate['nodes'][0]
    validate_fit(fit_directory, gate, node, fit['parents']['bank'], technical=technical)
    refs = {key: file_ref(fit_directory/name) for key, name in
            [('checkpoint', 'selected.pt'), ('report', 'training_report.json'), ('fit', 'receipt.json')]}
    parents = dict(gate=gate['content_hash'], fit=fit['content_hash'],
                   checkpoint=refs['checkpoint']['sha256'], report=refs['report']['sha256'])
    model = new_student(node)
    model.load_state_dict(torch.load(validate_file_ref(refs['checkpoint']),
                                     map_location='cpu', weights_only=True), strict=True)
    directory.mkdir(parents=True, exist_ok=False)
    payload = compact_export(directory/'payload', model, train, context,
                             parents=parents, device=device)
    result = artifact('BANK', parents=parents, technical=technical, teacher='PRIV_D000',
        alignment=shared.alignment(train), refs=refs,
        payload=file_ref(directory/'payload/manifest.json', root=directory),
        payload_hash=payload['content_hash'])
    write_immutable_json(directory/'manifest.json', result)
    return result


def compact_export(directory, model, train, context, *, parents, device):
    """Shared numerical layout, NEW FullStat identity (never trial holdout flags)."""
    require_train(train)
    shared.paired(train, context)
    if context is None or context.coordinate_name != 'D033' or model.mode != 'privileged':
        raise PermissionError('Only privileged TRAIN export')
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    lengths = shared.counts(train)
    probabilities = np.empty((len(train), 11), np.float32)
    pending, shards, begin = [], [], 0
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(train), 128):
            stop = min(start+128, len(train))
            ix = np.arange(start, stop)
            logits = shared.forward(model, train.batch(ix), context.batch(ix), device)
            probabilities[ix] = (logits.float()/2).softmax(-1).cpu().numpy()
            f = model.factors.detach().float().cpu().numpy()
            pending.append(np.concatenate([row[:n] for row, n in zip(f, lengths[ix])]))
            model.factors = None
            if stop-begin >= 4096 or stop == len(train):
                a = np.concatenate(pending)
                if a.shape != (int(lengths[begin:stop].sum()), 9) or not np.isfinite(a).all():
                    raise ValueError('Invalid factors')
                stream = BytesIO()
                np.save(stream, a, allow_pickle=False)
                path = directory/f'factors-{begin:07d}.npy'
                atomic_publish_bytes(path, stream.getvalue())
                shards.append(dict(start=begin, stop=stop, payload=file_ref(path, root=directory)))
                pending, begin = [], stop
            if (start//128) % 128 == 0:
                print(f'FULLSTAT GATE export rows={stop}/{len(train)}', flush=True)
    shared.check_probabilities(probabilities, len(train))
    atomic_publish_bytes(directory/'population.npz', deterministic_npz_bytes(dict(
        identities=train.identities, labels=train.labels, counts=lengths, probabilities=probabilities)))
    result = artifact('COMPACT_BANK', parents=parents, teacher='PRIV_D000',
        coordinate='D000', context_coordinate='D033', alignment=shared.alignment(train),
        temperature=2., factors=True, population=file_ref(directory/'population.npz', root=directory),
        shards=shards)
    write_immutable_json(directory/'manifest.json', result)
    return result


def check_manifest(directory, gate, fit_directory, *, technical=False):
    directory, fit_directory = Path(directory), Path(fit_directory)
    m = load_json(directory/'manifest.json')
    refs = {key: file_ref(fit_directory/name) for key, name in
            [('checkpoint', 'selected.pt'), ('report', 'training_report.json'), ('fit', 'receipt.json')]}
    fit = load_json(fit_directory/'receipt.json')
    parents = dict(gate=gate['content_hash'], fit=fit['content_hash'],
                   checkpoint=refs['checkpoint']['sha256'], report=refs['report']['sha256'])
    validate(m, 'BANK', parents)
    if (m['technical'] is not technical or m['teacher'] != 'PRIV_D000'
            or m['refs'] != refs or m['payload']['path'] != 'payload/manifest.json'):
        raise ValueError('Wrong selected teacher / technical target scope')
    payload = load_json(validate_file_ref(m['payload'], root=directory))
    validate(payload, 'COMPACT_BANK', parents)
    if (payload['content_hash'] != m['payload_hash'] or payload['alignment'] != m['alignment']
            or payload['teacher'] != 'PRIV_D000' or payload['coordinate'] != 'D000'
            or payload['context_coordinate'] != 'D033' or payload['factors'] is not True
            or payload['temperature'] != 2. or m['alignment']['role'] != 'train'
            or m['alignment']['rows'] != gate['counts']['train']
            or m['alignment']['prepared'] != gate['anchor']['evidence']['prepared']):
        raise ValueError('Compact factor bank identity differs')
    if payload['population']['path'] != 'population.npz':
        raise ValueError('Unexpected population locator')
    stop, seen = 0, set()
    for s in payload['shards']:
        if (type(s['start']) is not int or type(s['stop']) is not int
                or s['start'] != stop or not stop < s['stop'] <= min(stop+4096, gate['counts']['train'])
                or s['payload']['path'] != f'factors-{stop:07d}.npy'
                or s['payload']['path'] in seen):
            raise ValueError('Factor coverage differs')
        stop = s['stop']
        seen.add(s['payload']['path'])
    if stop != gate['counts']['train']:
        raise ValueError('Incomplete factor coverage')
    for ref in [payload['population']] + [s['payload'] for s in payload['shards']]:
        validate_file_ref(ref, root=directory/'payload')
    return m


class Resident:
    def __init__(self, directory, gate, train, fit_directory, *, technical=False):
        require_train(train)
        directory = Path(directory)
        self.manifest = check_manifest(directory, gate, fit_directory, technical=technical)
        payload = load_json(directory/'payload/manifest.json')
        pop = load_npz_arrays(validate_file_ref(payload['population'], root=directory/'payload'))
        if (set(pop) != {'identities', 'labels', 'counts', 'probabilities'}
                or pop['identities'].dtype != np.uint8 or pop['labels'].dtype != np.int64
                or pop['counts'].dtype != np.int64
                or not np.array_equal(pop['identities'], train.identities)
                or not np.array_equal(pop['labels'], train.labels)
                or not np.array_equal(pop['counts'], shared.counts(train))
                or self.manifest['alignment'] != shared.alignment(train)):
            raise ValueError('Resident target population differs')
        self.identities, self.labels = train.identities, train.labels
        self.probabilities, self.lengths = pop['probabilities'], pop['counts']
        shared.check_probabilities(self.probabilities, len(train))
        self.offsets = np.concatenate(([0], np.cumsum(self.lengths))).astype(np.int64)
        required = int(self.offsets[-1])*9*4
        if required > .25*gate['memory_mb']*1024**2:
            raise MemoryError('Resident compact factors exceed registered host envelope')
        self.factors = np.empty((int(self.offsets[-1]), 9), np.float32)
        for shard in payload['shards']:
            ref = shard['payload']
            value = np.load(validate_file_ref(ref, root=directory/'payload'),
                            mmap_mode='r', allow_pickle=False)
            a, b = self.offsets[shard['start']], self.offsets[shard['stop']]
            if value.dtype != np.float32 or value.shape != (b-a, 9) or not np.isfinite(value).all():
                raise ValueError('Invalid factor payload')
            self.factors[a:b] = value
        self.probabilities.flags.writeable = False
        self.factors.flags.writeable = False

    def factor_batch(self, indices, width):
        ix = np.asarray(indices)
        if (ix.ndim != 1 or ix.dtype.kind not in 'iu' or not len(ix)
                or np.any(ix < 0) or np.any(ix >= len(self.lengths))
                or not self.lengths[ix].max() <= width <= 512):
            raise ValueError('Invalid factor gather')
        out = np.zeros((len(ix), width, 9), np.float32)
        for j, i in enumerate(ix):
            out[j, :self.lengths[i]] = self.factors[self.offsets[i]:self.offsets[i+1]]
        return out


def check_join(bank, train, node, *, technical):
    require_train(train)
    kind = 'SUPERVISION' if node['teacher'] == TEACHER else 'BANK'
    validate(bank.manifest, kind)
    if (bank.manifest['teacher'] != node['teacher']
            or bank.manifest['alignment'] != shared.alignment(train)
            or not np.array_equal(bank.identities, train.identities)
            or not np.array_equal(bank.labels, train.labels)
            or (kind == 'BANK' and bank.manifest['technical'] is not technical)):
        raise ValueError('Target population/teacher/technical join differs')
    shared.check_probabilities(bank.probabilities, len(train))
