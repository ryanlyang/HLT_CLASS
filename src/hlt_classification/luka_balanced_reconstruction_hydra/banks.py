"""One fixed same-seed EMA05 teacher bank, TRAIN-only, atomic publication."""
from pathlib import Path
import numpy as np
import torch
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.luka_sixclass_balanced_oscar.banks import save_array
from hlt_classification.luka_embedding_aux.targets import standardize
from . import contracts as c,artifacts,kernel,model


def export(spec,task,train,ctx,*,folder):
    x=c.cell(task)
    if x['arm'] not in ('OFFLINE','U100') or train.role!='train' or train.coordinate_name!=x['coordinate']:
        raise PermissionError('Only selected outgoing TRAIN teachers')
    folder=Path(folder);report=load_json(folder/'training_report.json')
    checkpoint=folder/'EMA05/selected.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    with torch.random.fork_rng(devices=[]):net=model.NativeParticleTransformer()
    net=net.to(ctx.device).eval();net.load_state_dict(state,strict=True)
    width=128 if x['arm']=='U100' else 6
    indices=[];values=[]
    with torch.inference_mode():
        for start in range(0,len(train),c.BATCH):
            ix=np.arange(start,min(start+c.BATCH,len(train)));local=np.array_split(np.arange(len(ix)),ctx.world)[ctx.rank]
            if not len(local):continue
            raw=train.batch(ix);raw={k:v[local] for k,v in raw.items()}
            if width==128:
                tensors=[torch.as_tensor(raw[k],device=ctx.device) for k in ('features','vectors','mask')]
                _,value=model.head_input(net,*tensors)
            else:value=(kernel.native(net,raw,ctx.device)/2).softmax(1)
            indices.append(ix[local]);values.append(value.float().cpu().numpy())
    pairs=ctx.all((np.concatenate(indices),np.concatenate(values)))
    def publish():
        bank=folder/'bank';bank.mkdir(exist_ok=False)
        merged=np.empty((len(train),width),np.float32);seen=np.zeros(len(train),bool)
        for ix,v in pairs:
            if np.any(seen[ix]) or v.shape!=(len(ix),width):raise ValueError('Duplicate/missing teacher rows')
            merged[ix]=v;seen[ix]=True
        if not seen.all() or not np.isfinite(merged).all():raise ValueError('Incomplete teacher export')
        payload=dict(identities=train.identities,labels=train.labels)
        if width==128:
            target,mean,scale=standardize(merged);payload.update(targets=target,mean=mean,scale=scale)
        else:payload['probabilities']=merged
        for name,v in payload.items():save_array(bank/(name+'.npy'),v)
        record=c.artifact('BANK',parents=dict(spec=spec['content_hash'],population=spec['population']['content_hash'],
            checkpoint=file_ref(checkpoint)['sha256'],report=report['content_hash']),task=task,role='train',
            coordinate=x['coordinate'],readout='EMA05',rows=len(train),width=width,temperature=2. if width==6 else None,
            payloads={k:file_ref(bank/(k+'.npy'),root=bank) for k in payload})
        write_immutable_json(bank/'manifest.json',record);return record
    return ctx.main(publish)


class Bank:
    def __init__(self,spec,task,train):
        if train.role!='train':raise PermissionError('TRAIN-only teacher target access')
        x=c.cell(task);teacher=x['teacher']
        if teacher is None:raise PermissionError('Arm has no teacher')
        r=artifacts.completed(spec,teacher);folder=Path(r['receipt']['files']['bank']['path']).parent
        record=load_json(validate_file_ref(r['receipt']['files']['bank']))
        c.validate(record,'BANK',dict(spec=spec['content_hash'],population=spec['population']['content_hash'],
            checkpoint=r['receipt']['files']['EMA05_checkpoint']['sha256'],report=r['report']['content_hash']))
        width=6 if x['arm']=='U100' else 128
        if (record['task']!=teacher or record['role']!='train' or record['readout']!='EMA05'
                or record['rows']!=len(train) or record['width']!=width
                or record['coordinate']!=c.cell(teacher)['coordinate'] or record['temperature']!=(2. if width==6 else None)):
            raise ValueError('Teacher bank semantics differ')
        shapes=dict(identities=((len(train),32),np.uint8),labels=((len(train),),np.int64))
        if width==6:shapes['probabilities']=((len(train),6),np.float32)
        else:shapes.update(targets=((len(train),128),np.float32),mean=((128,),np.float64),scale=((128,),np.float64))
        if set(record['payloads'])!=set(shapes):raise ValueError('Bank inventory differs')
        for name,(shape,dtype) in shapes.items():
            ref=record['payloads'][name]
            if ref['path']!=name+'.npy':raise ValueError('Unexpected bank path')
            a=np.load(validate_file_ref(ref,root=folder),mmap_mode='r',allow_pickle=False)
            if a.shape!=shape or a.dtype!=dtype or not np.isfinite(a).all():raise ValueError('Invalid target matrix')
            setattr(self,name,a)
        if not np.array_equal(self.identities,train.identities) or not np.array_equal(self.labels,train.labels):
            raise ValueError('Canonical teacher TRAIN row order differs')
        if width==6:
            if (self.probabilities<0).any() or not np.allclose(self.probabilities.sum(1),1,atol=2e-6,rtol=0):raise ValueError('Invalid teacher probabilities')
        elif (self.scale<.001).any() or np.max(np.abs(self.targets.mean(0,dtype=np.float64)))>1e-5 or self.targets.std(0,dtype=np.float64).max()>1.00001:
            raise ValueError('Invalid TRAIN target normalization')
        self.manifest=record;self.teacher_receipt=r['receipt']['content_hash']
