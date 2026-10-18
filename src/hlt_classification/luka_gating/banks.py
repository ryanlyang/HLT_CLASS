"""Small authenticated TRAIN logits and ragged gate-factor banks, not attention."""
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (atomic_publish_bytes, deterministic_npz_bytes,
    load_json, load_npz_arrays, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref, safe
from hlt_classification.luka_attention.banks import alignment, counts
from .contracts import artifact, validate


def paired(cache, context):
    if context is not None and alignment(cache)!=alignment(context):
        raise ValueError('Unaligned view IDs/labels/counts/prepared parent')


def forward(model, raw, context, device):
    values = {k:torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')}
    if context is not None:
        if (not np.array_equal(raw['identities'],context['identities'])
                or not np.array_equal(raw['labels'],context['labels'])
                or not np.array_equal(raw['mask'],context['mask'])):
            raise ValueError('Batch pair order/mask differs')
        values.update(context_features=torch.from_numpy(context['features']).to(device),
                      context_mask=torch.from_numpy(context['mask']).to(device))
    with torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16,
                        enabled=str(device).startswith('cuda')):
        return model(**values)


def predict(model, cache, context=None, *, device, temperature=1.):
    paired(cache,context)
    if temperature not in (1.,2.):
        raise ValueError('Unregistered probability temperature')
    model.to(device).eval()
    p = np.empty((len(cache),11),np.float32)
    with torch.inference_mode():
        for start in range(0,len(cache),128):
            ix = np.arange(start,min(start+128,len(cache)))
            logits = forward(model,cache.batch(ix),None if context is None else context.batch(ix),device)
            p[ix] = (logits.float()/temperature).softmax(-1).cpu().numpy()
            model.factors = None
    check_probabilities(p,len(cache))
    return p


def check_probabilities(p, rows):
    if (p.shape!=(rows,11) or p.dtype!=np.float32 or not np.isfinite(p).all()
            or np.any(p<0) or not np.allclose(p.sum(-1),1.,atol=2e-6,rtol=0)):
        raise ValueError('Invalid probabilities')


def export(root, model, cache, context, *, parents, node, device):
    if cache.role!='train':
        raise PermissionError('Only TRAIN may produce targets')
    paired(cache,context)
    if (cache.coordinate_name!=node['coordinate']
            or (None if context is None else context.coordinate_name)!=node['context_coordinate']):
        raise ValueError('Export views differ')
    root = Path(root); root.mkdir(parents=True,exist_ok=False)
    factor_export = node['node_id']=='PRIV_D000'
    probabilities = np.empty((len(cache),11),np.float32)
    lengths = counts(cache); pending=[]; shards=[]; begin=0
    def flush(stop):
        nonlocal pending,begin
        a = np.concatenate(pending)
        if a.dtype!=np.float32 or a.shape!=(int(lengths[begin:stop].sum()),9) or not np.isfinite(a).all():
            raise ValueError('Invalid exported factor tensor')
        stream=BytesIO(); np.save(stream,a,allow_pickle=False)
        name=f'factors-{begin:07d}.npy'
        atomic_publish_bytes(root/name,stream.getvalue())
        shards.append(dict(start=begin,stop=stop,payload=file_ref(root/name,root=root)))
        pending=[]; begin=stop
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0,len(cache),128):
            stop=min(start+128,len(cache)); ix=np.arange(start,stop)
            logits=forward(model,cache.batch(ix),None if context is None else context.batch(ix),device)
            probabilities[ix]=(logits.float()/2).softmax(-1).cpu().numpy()
            if factor_export:
                factors=model.factors.detach().float().cpu().numpy()
                pending.append(np.concatenate([f[:n] for f,n in zip(factors,lengths[ix])]))
                if stop-begin>=4096: flush(stop)
            model.factors=None
            if start%(128*100)==0: print(f'Gate export {node["node_id"]}: {stop}/{len(cache)}',flush=True)
    if pending: flush(len(cache))
    check_probabilities(probabilities,len(cache))
    atomic_publish_bytes(root/'population.npz',deterministic_npz_bytes(dict(identities=cache.identities,
        labels=cache.labels,counts=lengths,probabilities=probabilities)))
    result=artifact('BANK',parents=parents,teacher=node['node_id'],coordinate=node['coordinate'],
        context_coordinate=node['context_coordinate'],alignment=alignment(cache),temperature=2.,
        factors=factor_export,population=file_ref(root/'population.npz',root=root),shards=shards)
    write_immutable_json(root/'manifest.json',result)
    return result


class Bank:
    def __init__(self, root, cache, *, parents, teacher, require_factors=False):
        if cache.role!='train': raise PermissionError('TRAIN target bank only')
        self.root=Path(root); m=self.manifest=load_json(self.root/'manifest.json')
        validate(m,'BANK',parents)
        if (m['teacher']!=teacher or m['alignment']!=alignment(cache) or m['temperature']!=2.
                or type(m['factors']) is not bool or m['factors']!=(teacher=='PRIV_D000')
                or (require_factors and not m['factors'])):
            raise ValueError('Teacher bank lineage/surface differs')
        expected = 'U100' if teacher=='ANCHOR' else teacher.rsplit('_',1)[-1]
        if expected not in ('U100','D066','D033','D000') or m['coordinate']!=expected:
            raise ValueError('Teacher coordinate differs')
        ctx = {'D066':'U100','D033':'D066','D000':'D033'}.get(m['coordinate']) if teacher.startswith('PRIV_') else None
        if m['context_coordinate']!=ctx: raise ValueError('Teacher context differs')
        pop=load_npz_arrays(validate_file_ref(m['population'],root=self.root))
        if (set(pop)!={'identities','labels','counts','probabilities'}
                or pop['identities'].dtype!=np.uint8 or pop['labels'].dtype!=np.int64
                or pop['counts'].dtype!=np.int64
                or not np.array_equal(pop['identities'],cache.identities)
                or not np.array_equal(pop['labels'],cache.labels)
                or not np.array_equal(pop['counts'],counts(cache))):
            raise ValueError('Bank ordered population differs')
        self.probabilities=pop['probabilities']; self.lengths=pop['counts']
        check_probabilities(self.probabilities,len(cache))
        stop=0; seen=set()
        for s in m['shards']:
            a,b=s['start'],s['stop']
            if a!=stop or not a<b<=min(a+4096,len(cache)) or s['payload']['path'] in seen:
                raise ValueError('Factor coverage/paths differ')
            seen.add(s['payload']['path'])
            path=validate_file_ref(s['payload'],root=self.root)
            x=np.load(path,mmap_mode='r',allow_pickle=False)
            if x.dtype!=np.float32 or x.shape!=(int(self.lengths[a:b].sum()),9) or not np.isfinite(x).all():
                raise ValueError('Factor payload invalid')
            del x
            stop=b
        if stop!=(len(cache) if m['factors'] else 0): raise ValueError('Incomplete factor bank')
        self.ends=np.asarray([s['stop'] for s in m['shards']]); self._lru=OrderedDict()

    def factor_batch(self, indices, width):
        ix=np.asarray(indices)
        if (not self.manifest['factors'] or ix.ndim!=1 or ix.dtype.kind not in 'iu' or not len(ix)
                or np.any(ix<0) or np.any(ix>=len(self.lengths)) or width<self.lengths[ix].max()):
            raise ValueError('Invalid factor selection')
        out=np.zeros((len(ix),width,9),np.float32)
        for dest,i in enumerate(ix):
            shard=int(np.searchsorted(self.ends,i,side='right')); s=self.manifest['shards'][shard]
            if shard not in self._lru:
                self._lru[shard]=np.load(safe(self.root,s['payload']['path']),mmap_mode='r',allow_pickle=False)
                while len(self._lru)>4: self._lru.popitem(last=False)
            self._lru.move_to_end(shard)
            offset=int(self.lengths[s['start']:i].sum()); n=self.lengths[i]
            out[dest,:n]=self._lru[shard][offset:offset+n]
        return out
