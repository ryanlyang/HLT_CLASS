"""Selected CONCAT/FUSION TRAIN-only target banks; original U100 stays immutable."""
from pathlib import Path
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    atomic_publish_bytes, deterministic_npz_bytes, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_embedding_aux.targets import standardize
from hlt_classification.luka_u100_multiview import teachers
from .contracts import artifact, validate, recipe


def parents(spec, name, refs):
    return dict(spec=spec['content_hash'],recipe=recipe()['content_hash'],
        prepared=spec['anchor']['evidence']['prepared'],checkpoint=refs['checkpoint']['sha256'],report=refs['report']['sha256'])


def export(folder, net, cache, name, refs, spec, device):
    if name not in ('CONCAT','FUSION') or cache.role!='train' or cache.coordinate_name!=name:
        raise PermissionError('Only new oracle TRAIN embeddings may be exported')
    for key in ('checkpoint','report'): validate_file_ref(refs[key])
    state=torch.load(refs['checkpoint']['path'],map_location='cpu',weights_only=True)
    if state.keys()!=net.state_dict().keys() or any(not torch.equal(v.cpu(),state[k]) for k,v in net.state_dict().items()):
        raise ValueError('Export teacher is not selected checkpoint')
    del state
    net.eval().requires_grad_(False)
    raw=np.empty((len(cache),128),np.float32)
    with torch.inference_mode(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        for start in range(0,len(cache),128):
            ix=np.arange(start,min(start+128,len(cache)))
            _,h=teachers.forward(net,cache.batch(ix),device,bf16=False,name=name,capture=True)
            raw[ix]=h.float().cpu().numpy()
            if start%(128*256)==0: print(f'FULLSTAT MULTIVIEW export={name} rows={start+len(ix)}/{len(cache)}',flush=True)
    targets,mean,scale=standardize(raw); del raw
    folder=Path(folder); folder.mkdir(exist_ok=False)
    atomic_publish_bytes(folder/'targets.npz',deterministic_npz_bytes(dict(
        identities=cache.identities,labels=cache.labels,targets=targets,mean=mean,scale=scale)))
    for key in ('checkpoint','report'): validate_file_ref(refs[key])
    record=artifact('BANK',parents=parents(spec,name,refs),teacher=name,role='train',rows=len(cache),width=128,
        checkpoint=refs['checkpoint'],report=refs['report'],payload=file_ref(folder/'targets.npz',root=folder),
        export_batch=128,precision='FP32_no_autocast',normalization=recipe()['normalization'])
    write_immutable_json(folder/'manifest.json',record)
    return record


class Bank:
    def __init__(self,folder,cache,name,refs,spec):
        if cache.role!='train' or cache.coordinate_name not in ('D000',name) or name not in ('CONCAT','FUSION'):
            raise PermissionError('New banks are TRAIN-only supervision')
        folder=Path(folder); m=load_json(folder/'manifest.json'); validate(m,'BANK',parents(spec,name,refs))
        if (m['teacher']!=name or m['role']!='train' or m['rows']!=len(cache) or m['width']!=128
                or m['checkpoint']!=refs['checkpoint'] or m['report']!=refs['report']
                or m['payload']['path']!='targets.npz' or m['export_batch']!=128
                or m['precision']!='FP32_no_autocast' or m['normalization']!=recipe()['normalization']):
            raise ValueError('Target bank definition differs')
        for key in ('checkpoint','report'): validate_file_ref(refs[key])
        a=load_npz_arrays(validate_file_ref(m['payload'],root=folder))
        shapes={'identities':(len(cache),32),'labels':(len(cache),),'targets':(len(cache),128),'mean':(128,),'scale':(128,)}
        if set(a)!=set(shapes): raise ValueError('Bank inventory differs')
        for k,shape in shapes.items():
            dtype={'identities':np.uint8,'labels':np.int64,'targets':np.float32,'mean':np.float64,'scale':np.float64}[k]
            if a[k].shape!=shape or a[k].dtype!=dtype or not np.isfinite(a[k]).all():
                raise ValueError('Target shape/type/finiteness differs')
        if (not np.array_equal(a['identities'],cache.identities) or not np.array_equal(a['labels'],cache.labels)
                or np.any(a['scale']<.001) or np.max(np.abs(a['targets'].mean(0,dtype=np.float64)))>1e-5
                or np.max(a['targets'].std(0,dtype=np.float64))>1.00001):
            raise ValueError('Target normalization/ordered TRAIN join differs')
        self.targets,self.identities,self.labels,self.manifest=a['targets'],a['identities'],a['labels'],m
        for value in a.values(): value.flags.writeable=False
