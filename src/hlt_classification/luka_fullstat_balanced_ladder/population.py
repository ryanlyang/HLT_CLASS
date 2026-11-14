"""Authenticate scalar membership, freeze eligible rows, and preserve validation."""
from pathlib import Path
import numpy as np
from hlt_classification.operations import fullstat_pt_metadata as meta
from hlt_classification.luka_fullstat.contracts import identity
from hlt_classification.data.cache_contracts import deterministic_npz_bytes, load_npz_arrays
from .contracts import artifact, validate, COUNTS, ELIGIBLE, EDGES, AUDIT_HASH
from .sampling import BalancedSampler


def occupancy():
    value = meta.load_json(Path(__file__).with_name('occupancy.json'))
    counts = np.asarray(value['counts'], dtype=np.int64)
    if (value['audit_content_hash'] != AUDIT_HASH or value['pt_edges_GeV'] != list(EDGES)
            or counts.shape != (11,18) or np.any(counts <= 0) or counts.sum() != ELIGIBLE):
        raise ValueError('Frozen audit occupancy differs')
    return counts


def arrays(foundation, memberships):
    output = {r:{k:[] for k in ('identities','labels','pt')} for r in ('train','validation')}
    for record, member in zip(foundation['files'], memberships, strict=True):
        for role, code in (('train',0),('validation',1)):
            entries = np.flatnonzero(member['roles'] == code)
            labels = member['labels'][entries].astype(np.int64)
            pt = member['jet_pt'][entries].astype(np.float64)
            if (not np.array_equal(labels,meta.map_labels(member['jet_label'][entries]))
                    or not np.isfinite(pt).all() or np.any(pt <= 0)):
                raise ValueError('Invalid ordinary-role scalar membership')
            ids = np.asarray([np.frombuffer(bytes.fromhex(identity(record,int(e))),np.uint8)
                              for e in entries],dtype=np.uint8).reshape(-1,32)
            for key, value in (('identities',ids),('labels',labels),('pt',pt)):
                output[role][key].append(value)
        print(f'BALANCED population source={record["path"]}',flush=True)
    return {f'{r}_{k}':np.concatenate(v) for r, fields in output.items() for k,v in fields.items()}


def prepare(folder, metadata, *, input_hash):
    manifest, foundation, memberships = metadata
    values = arrays(foundation,memberships)
    if any(len(values[r+'_labels']) != COUNTS[r] for r in ('train','validation')):
        raise ValueError('Original ordinary-role counts differ')
    sampler = BalancedSampler(values['train_labels'],values['train_pt'],expected_counts=occupancy())
    values['eligible_indices'] = sampler.indices
    folder = Path(folder); folder.mkdir(parents=True,exist_ok=False)
    meta.atomic_publish_bytes(folder/'rows.npz',deterministic_npz_bytes(values))
    report = artifact('POPULATION',parents=dict(inputs=input_hash,prepared=manifest['content_hash']),
        original_counts=dict(COUNTS),eligible_rows=ELIGIBLE,excluded_train_rows=COUNTS['train']-ELIGIBLE,
        class_bin_counts=sampler.counts.tolist(),pt_edges_GeV=list(EDGES),boundary='[low,high)',
        audit_hash=AUDIT_HASH,identity_order='foundation_file_then_source_entry',
        validation_selection='original_natural_500000',
        arrays={k:meta.array_sha256(k,v) for k,v in values.items()},
        payload=meta.file_ref(folder/'rows.npz',root=folder))
    meta.write_immutable_json(folder/'manifest.json',report)
    return report


def load(folder, *, input_hash, prepared_hash):
    folder = Path(folder)
    report = meta.load_json(folder/'manifest.json')
    validate(report,'POPULATION',dict(inputs=input_hash,prepared=prepared_hash))
    if (report['original_counts'] != COUNTS or report['eligible_rows'] != ELIGIBLE
            or report['excluded_train_rows'] != COUNTS['train']-ELIGIBLE
            or report['class_bin_counts'] != occupancy().tolist()
            or report['pt_edges_GeV'] != list(EDGES) or report['boundary'] != '[low,high)'
            or report['audit_hash'] != AUDIT_HASH or report['payload']['path'] != 'rows.npz'
            or report['identity_order'] != 'foundation_file_then_source_entry'
            or report['validation_selection'] != 'original_natural_500000'):
        raise ValueError('Population policy differs')
    values = load_npz_arrays(meta.validate_file_ref(report['payload'],root=folder))
    shapes = {f'{r}_{k}':((COUNTS[r],32),np.uint8) if k=='identities' else
              ((COUNTS[r],),np.int64 if k=='labels' else np.float64)
              for r in ('train','validation') for k in ('identities','labels','pt')}
    shapes['eligible_indices'] = ((ELIGIBLE,),np.int64)
    if set(values) != set(shapes) or set(report['arrays']) != set(shapes):
        raise ValueError('Population array inventory differs')
    for key,(shape,dtype) in shapes.items():
        a=values[key]
        if a.shape != shape or a.dtype != dtype or not np.isfinite(a).all() or meta.array_sha256(key,a)!=report['arrays'][key]:
            raise ValueError('Population array differs: '+key)
        a.flags.writeable=False
    sampler=BalancedSampler(values['train_labels'],values['train_pt'],expected_counts=occupancy())
    if not np.array_equal(values['eligible_indices'],sampler.indices):
        raise ValueError('Eligible membership differs')
    for role in ('train','validation'):
        a=values[role+'_identities']
        if len(np.unique(a.view('V32'))) != len(a):
            raise ValueError('Duplicate canonical identity')
    if np.intersect1d(values['train_identities'].view('V32'),values['validation_identities'].view('V32')).size:
        raise ValueError('TRAIN/validation identity overlap')
    return report, values, sampler
