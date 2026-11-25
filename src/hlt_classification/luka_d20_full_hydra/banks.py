"""Rank-partitioned TRAIN bank export without large Python object gathers."""
from pathlib import Path
import numpy as np
import torch
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.luka_sixclass_balanced_oscar.banks import save_array
from . import contracts as c,artifacts,kernel,model

CHUNK=65536


def moments(values):
    if values.ndim!=2 or not len(values):raise ValueError('Invalid standardization input')
    total=np.zeros(values.shape[1],np.float64)
    for lo in range(0,len(values),CHUNK):
        block=np.asarray(values[lo:lo+CHUNK],np.float64)
        if not np.isfinite(block).all():raise ValueError('Nonfinite teacher values')
        total+=block.sum(0)
    mean=total/len(values);squares=np.zeros_like(mean)
    for lo in range(0,len(values),CHUNK):
        delta=np.asarray(values[lo:lo+CHUNK],np.float64)-mean;squares+=(delta*delta).sum(0)
    return mean,np.maximum(np.sqrt(squares/len(values)),.001)


def export(spec,task,train,ctx,*,folder):
    x=c.cell(task)
    if x['arm'] not in ('OFFLINE','U100') or train.role!='train' or train.coordinate_name!=x['coordinate']:
        raise PermissionError('Only selected outgoing TRAIN teachers')
    folder=Path(folder);report=load_json(folder/'training_report.json');checkpoint=folder/'EMA05/selected.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    with torch.random.fork_rng(devices=[]):net=model.NativeParticleTransformer()
    net=net.to(ctx.device).eval();net.load_state_dict(state,strict=True)
    width=128 if x['arm']=='U100' else 6;bank=folder/'bank';raw_path=bank/'raw.npy';seen_path=bank/'seen.npy'
    def allocate():
        bank.mkdir(exist_ok=False)
        raw=np.lib.format.open_memmap(raw_path,mode='w+',dtype=np.float32,shape=(len(train),width));raw.flush();del raw
        seen=np.lib.format.open_memmap(seen_path,mode='w+',dtype=np.uint8,shape=(len(train),));seen[:]=0;seen.flush();del seen
    ctx.main(allocate)
    raw=np.load(raw_path,mmap_mode='r+');seen=np.load(seen_path,mmap_mode='r+')
    with torch.inference_mode():
        for start in range(0,len(train),c.BATCH):
            ix=np.arange(start,min(start+c.BATCH,len(train)));local=np.array_split(np.arange(len(ix)),ctx.world)[ctx.rank]
            if not len(local):continue
            batch=train.batch(ix);batch={k:v[local] for k,v in batch.items()}
            if width==128:
                _,value=model.head_input(net,*[torch.as_tensor(batch[k],device=ctx.device) for k in ('features','vectors','mask')])
            else:value=(kernel.native(net,batch,ctx.device)/2).softmax(1)
            raw[ix[local]]=value.float().cpu().numpy();seen[ix[local]]=1
    raw.flush();seen.flush();del raw,seen,net;ctx.barrier()
    def publish():
        raw=np.load(raw_path,mmap_mode='r');seen=np.load(seen_path,mmap_mode='r')
        if not np.all(seen==1):raise ValueError('Incomplete unique TRAIN teacher coverage')
        payload=dict(identities=train.identities,labels=train.labels)
        if width==128:
            mean,scale=moments(raw);payload.update(mean=mean,scale=scale)
            target_path=bank/'targets.npy'
            target=np.lib.format.open_memmap(target_path,mode='w+',dtype=np.float32,shape=raw.shape)
            for lo in range(0,len(raw),CHUNK):target[lo:lo+CHUNK]=((np.asarray(raw[lo:lo+CHUNK],np.float64)-mean)/scale).astype(np.float32)
            target.flush();del target
        else:
            for lo in range(0,len(raw),CHUNK):
                block=raw[lo:lo+CHUNK]
                if not np.isfinite(block).all() or (block<0).any() or not np.allclose(block.sum(1),1,atol=2e-6,rtol=0):raise ValueError('Invalid teacher probabilities')
            # Same-directory rename only after every rank flushed its partition.
            del block
            raw._mmap.close()
            del raw
            raw_path.rename(bank/'probabilities.npy')
        for name,value in payload.items():save_array(bank/(name+'.npy'),value)
        names=[*payload,'targets' if width==128 else 'probabilities']
        record=c.artifact('BANK',parents=dict(spec=spec['content_hash'],population=spec['population']['content_hash'],
            checkpoint=file_ref(checkpoint)['sha256'],report=report['content_hash']),task=task,role='train',
            coordinate=x['coordinate'],readout='EMA05',rows=len(train),width=width,temperature=2. if width==6 else None,
            payloads={k:file_ref(bank/(k+'.npy'),root=bank) for k in names})
        write_immutable_json(bank/'manifest.json',record)
        del seen
        if width==128:del raw;raw_path.unlink()
        seen_path.unlink();return record
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
        else:
            target_mean,target_scale=moments(self.targets)
            if (self.scale<.001).any() or np.max(np.abs(target_mean))>1e-5 or target_scale.max()>1.00001:
                raise ValueError('Invalid TRAIN target normalization')
        self.manifest=record;self.teacher_receipt=r['receipt']['content_hash']
