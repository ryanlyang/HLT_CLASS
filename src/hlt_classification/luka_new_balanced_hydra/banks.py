"""Fresh selected native teachers; immutable uncompressed TRAIN target arrays."""
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from hlt_classification.operations import fullstat_pt_metadata as meta
from hlt_classification.luka_embedding_aux.targets import standardize
from hlt_classification.luka_u100_multiview.teachers import forward
from .contracts import artifact, validate, recipe, COORDINATES, STAGES


def save_array(path,value):
    """Atomic publication without building a second whole-bank bytes object."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:
            np.save(handle,value,allow_pickle=False);handle.flush();os.fsync(handle.fileno())
        os.link(temp,path)
    finally:
        os.unlink(temp)


def parents(spec,stage,refs):
    return dict(spec=spec['content_hash'],population=spec['population']['content_hash'],
                recipe=recipe()['content_hash'],checkpoint=refs['checkpoint']['sha256'],report=refs['report']['sha256'])


def export(folder,net,cache,stage,refs,spec,device):
    if stage not in STAGES[:-1] or cache.role!='train' or cache.coordinate_name!=COORDINATES[stage]:
        raise PermissionError('Only outgoing selected eligible TRAIN teachers')
    for ref in refs.values():meta.validate_file_ref(ref)
    state=torch.load(refs['checkpoint']['path'],map_location='cpu',weights_only=True)
    if state.keys()!=net.state_dict().keys() or any(not torch.equal(v.cpu(),state[k]) for k,v in net.state_dict().items()):
        raise ValueError('Export model differs from selected native checkpoint')
    del state
    net.eval().requires_grad_(False)
    raw=np.empty((len(cache),128),np.float32);probabilities=np.empty((len(cache),11),np.float32)
    with torch.inference_mode():
        for start in range(0,len(cache),128):
            ix=np.arange(start,min(start+128,len(cache)));batch=cache.batch(ix)
            _,h=forward(net,batch,device,bf16=False,capture=True)
            raw[ix]=h.float().cpu().numpy()
            logits=forward(net,batch,device,bf16=torch.device(device).type=='cuda')
            probabilities[ix]=(logits.float()/2.).softmax(-1).cpu().numpy()
            if start%(128*256)==0:print(f'BALANCED export={stage} rows={start+len(ix)}/{len(cache)}',flush=True)
    targets,mean,scale=standardize(raw);del raw
    folder=Path(folder);folder.mkdir(exist_ok=False)
    values=dict(identities=cache.identities,labels=cache.labels,targets=targets,
                probabilities=probabilities,mean=mean,scale=scale)
    for key,value in values.items():save_array(folder/(key+'.npy'),value)
    for ref in refs.values():meta.validate_file_ref(ref)
    record=artifact('BANK',parents=parents(spec,stage,refs),stage=stage,coordinate=COORDINATES[stage],
        role='train',rows=len(cache),width=128,temperature=2.,refs=refs,
        normalization=recipe()['normalization'],embedding_export=recipe()['embedding_export'],
        probability_export=recipe()['probability_export'],
        payloads={k:meta.file_ref(folder/(k+'.npy'),root=folder) for k in values})
    meta.write_immutable_json(folder/'manifest.json',record)
    return record


class Bank:
    def __init__(self,folder,cache,stage,refs,spec):
        if cache.role!='train' or stage not in STAGES[:-1]:raise PermissionError('TRAIN-only target access')
        folder=Path(folder);m=meta.load_json(folder/'manifest.json');validate(m,'BANK',parents(spec,stage,refs))
        expected=dict(stage=stage,coordinate=COORDINATES[stage],role='train',rows=len(cache),width=128,
            temperature=2.,refs=refs,normalization=recipe()['normalization'],
            embedding_export=recipe()['embedding_export'],probability_export=recipe()['probability_export'])
        if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Bank scientific definition differs')
        for ref in refs.values():meta.validate_file_ref(ref)
        shapes=dict(identities=((len(cache),32),np.uint8),labels=((len(cache),),np.int64),
            targets=((len(cache),128),np.float32),probabilities=((len(cache),11),np.float32),
            mean=((128,),np.float64),scale=((128,),np.float64))
        if set(m['payloads'])!=set(shapes):raise ValueError('Bank payload inventory differs')
        values={}
        for key,(shape,dtype) in shapes.items():
            ref=m['payloads'][key]
            if ref['path']!=key+'.npy':raise ValueError('Bank filename differs')
            value=np.load(meta.validate_file_ref(ref,root=folder),allow_pickle=False)
            if value.shape!=shape or value.dtype!=dtype or not np.isfinite(value).all():
                raise ValueError('Bank shape/dtype/finiteness differs: '+key)
            value.flags.writeable=False;values[key]=value
        if (not np.array_equal(values['identities'],cache.identities)
                or not np.array_equal(values['labels'],cache.labels)
                or np.any(values['scale']<.001) or np.any(values['probabilities']<0)
                or not np.allclose(values['probabilities'].sum(1),1.,atol=2e-6,rtol=0)
                or np.max(np.abs(values['targets'].mean(0,dtype=np.float64)))>1e-5
                or np.max(values['targets'].std(0,dtype=np.float64))>1.00001):
            raise ValueError('Bank identity/normalization/probabilities differ')
        self.__dict__.update(values);self.manifest=m

    def keys(self):
        # Same FP32 NumPy norm/reduction as the previous OFFLINE key bank.
        norms=np.linalg.norm(self.targets,axis=1,keepdims=True)
        return np.ascontiguousarray(self.targets/np.maximum(norms,np.float32(1e-12)),dtype=np.float32)
