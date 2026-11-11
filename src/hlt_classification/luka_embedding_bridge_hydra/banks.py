"""Typed producer outputs, common TRAIN moments and evaluation-only diagnostic panels."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from .contracts import artifact, validate, recipe
from .registry import PRODUCERS, ASSISTANTS, CONDITIONS, SCALES, SLICES, cell
from .storage import Writer, Store, join,validate_population
from . import response, distributed as d, inputs


def prepare_u100(spec,ds,ctx):
    """Evaluation targets use the original checkpoint and TRAIN normalization only."""
    from .model import Model
    t=inputs.transferred(spec);path=t.path(t.manifest['teacher']);u=ds.u100('train')
    teacher=Model('TR').to(ctx.device).eval()
    teacher.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    ix=panel(ds.view('train','U100'));z,_=d.predict(teacher,ds,'train',ctx,coordinate='U100',indices=ix,embedding=True)
    normalized=response.normalize(z,u['mean'],u['scale']);expected=np.array(u['targets'][ix])
    error=float(np.max(np.abs(normalized-expected)))
    witness=artifact('U100_REPLAY',parents=dict(spec=spec['content_hash'],teacher=t.manifest['teacher']['sha256'],original=spec['embedding_hash']),
        rows=len(ix),maximum_error=error,atol=1e-4,rtol=1e-4,passed=bool(np.allclose(normalized,expected,atol=1e-4,rtol=1e-4)))
    ctx.main(lambda:write_immutable_json(ds.root.parent/'u100_replay.json',witness))
    if not witness['passed']:raise ValueError('Original U100 TRAIN target reproduction differs')
    z,_=d.predict(teacher,ds,'validation',ctx,coordinate='U100',embedding=True)
    def publish():
        view=ds.view('validation','D000');a=dict(identities=view.identities,labels=view.labels,
            targets=response.normalize(z,u['mean'],u['scale']),mean=u['mean'],scale=u['scale'])
        w=Writer(ds.root/'u100_validation',{k:(v.dtype,v.shape) for k,v in a.items()})
        for k,v in a.items():w.arrays[k][:]=v
        return w.finish('U100_BANK',parents=dict(**inputs.data_parents(spec),original=spec['embedding_hash']),role='validation',
            coordinate='U100',rows=len(view),meaning='original_standardized_U100',usage='evaluation_only',source_manifest=u.manifest['source_manifest'])
    result=ctx.main(publish);del teacher;d.release_memory(ctx)
    return result


def panel(view):
    digest=[hashlib.sha256(b'embedding_bridge_validation_v1'+bytes(i)).digest() for i in view.identities]
    return np.array(sorted(range(len(view)),key=lambda i:(digest[i],i))[:min(4096,len(view))],np.int64)


def meaning(task):
    if task in ASSISTANTS:return 'predicted_original_standardized_U100'
    if task=='TR':return 'shared_native_standardized_six_conditions'
    if task=='TP':return 'shared_native_regional_and_whole'
    raise ValueError('Task does not produce teacher targets')


def parents(spec,task,checkpoint):
    return dict(spec=spec['content_hash'],recipe=recipe()['content_hash'],checkpoint=checkpoint['sha256'])


def weighted_region_moments(values,valid):
    """Equal source, scale, nonempty-jet weight; equal regions within each jet."""
    means=[];seconds=[];counts=[]
    for source in range(2):
        for scale in SCALES:
            a,b=SLICES[scale];a+=source*14;b+=source*14
            sm=np.zeros(128);sq=sm.copy();n=0
            for start in range(0,len(values),4096):
                x=values[start:start+4096,a:b].astype(np.float64);v=valid[start:start+4096,a:b]
                den=v.sum(1);ok=den>0
                if ok.any():
                    sm+=((x[ok]*v[ok,:,None]).sum(1)/den[ok,None]).sum(0)
                    sq+=((x[ok]**2*v[ok,:,None]).sum(1)/den[ok,None]).sum(0);n+=int(ok.sum())
            counts.append(n)
            if n:means.append(sm/n);seconds.append(sq/n)
    mean=np.mean(means,axis=0) if means else np.zeros(128)
    second=np.mean(seconds,axis=0) if seconds else np.zeros(128)
    return mean,np.maximum(np.sqrt(np.maximum(second-mean**2,0)),.001),counts


def export(folder,spec,task,student,ds,ctx,checkpoint,*,technical=False,indices=None):
    if task not in PRODUCERS:raise ValueError('Only a registered producer exports targets')
    folder=Path(folder);ctx.main(lambda:folder.mkdir(parents=True,exist_ok=False));statistics=None
    # Each export chunk is bounded; no full multi-view tensor is gathered onto GPUs.
    results={}
    for role in ('train','validation'):
        view=ds.view(role,'D000')
        ix=np.arange(len(view),dtype=np.int64) if role=='train' else panel(view)
        if indices is not None:ix=np.asarray(indices[role],np.int64)
        width=1 if task in ASSISTANTS else 6 if task=='TR' else 29
        w=Writer(folder/role,dict(identities=(np.uint8,(len(ix),32)),labels=(np.int64,(len(ix),)),
            indices=(np.int64,(len(ix),)),targets=(np.float32,(len(ix),width,128)),valid=(bool,(len(ix),width)))) if ctx.rank==0 else None
        if w:
            w.arrays['identities'][:]=view.identities[ix];w.arrays['labels'][:]=view.labels[ix];w.arrays['indices'][:]=ix
        for start in range(0,len(ix),1024):
            batch=ix[start:start+1024];values=[];valids=[]
            if task in ASSISTANTS:
                v,m=d.predict(student,ds,role,ctx,coordinate=cell(task)['input'],indices=batch,predicted=True)
                values=[v[:,None]];valids=[m[:,None]]
            elif task=='TR':
                for condition in CONDITIONS:
                    v,m=d.predict(student,ds,role,ctx,condition=condition,indices=batch,embedding=True)
                    values.append(v[:,None]);valids.append(m[:,None])
            else:
                for source in ('HLT','OFFLINE'):
                    for scale in SCALES:
                        v,m=d.predict(student,ds,role,ctx,source=source,scale=scale,indices=batch,embedding=True)
                        values.append(v);valids.append(m)
                v,m=d.predict(student,ds,role,ctx,coordinate='OFFLINE',indices=batch,embedding=True)
                values.append(v[:,None]);valids.append(m[:,None])
            if w:
                w.arrays['targets'][start:start+len(batch)]=np.concatenate(values,axis=1)
                w.arrays['valid'][start:start+len(batch)]=np.concatenate(valids,axis=1)
            if ctx.rank==0 and start%20480==0:print(f'EMBEDDING_BRIDGE {task} export {role}: {start+len(batch)}/{len(ix)}',flush=True)
        def finish():
            nonlocal statistics
            a=w.arrays['targets'];mask=w.arrays['valid']
            if role=='train':
                if task=='TR':
                    mean,scale,count=response.moments(a[i:i+4096] for i in range(0,len(a),4096))
                    statistics=dict(mean=mean.tolist(),scale=scale.tolist(),count=count)
                elif task=='TP':
                    mean,scale,counts=weighted_region_moments(a,mask)
                    wm,ws,wn=response.moments(a[i:i+4096,28] for i in range(0,len(a),4096))
                    statistics=dict(mean=mean.tolist(),scale=scale.tolist(),counts=counts,whole_mean=wm.tolist(),whole_scale=ws.tolist(),whole_count=wn)
                else:statistics=dict(normalization='none_original_U100_axes')
            if task in ('TR','TP'):
                for start in range(0,len(a),4096):
                    stop=min(start+4096,len(a));v=a[start:stop].copy()
                    count=6 if task=='TR' else 28
                    v[:,:count]=response.normalize(v[:,:count],np.array(statistics['mean']),np.array(statistics['scale']))
                    if task=='TP':v[:,28]=response.normalize(v[:,28],np.array(statistics['whole_mean']),np.array(statistics['whole_scale']))
                    v[~mask[start:stop]]=0;a[start:stop]=v
                if role=='train' and task=='TR':statistics['response_scales']=response.scales(a[i:i+4096] for i in range(0,len(a),4096))
            return w.finish('TARGET_BANK',parents=parents(spec,task,checkpoint),task=task,role=role,
                technical=technical,meaning=meaning(task),rows=len(ix),checkpoint=checkpoint,statistics=statistics,
                usage='optimization' if role=='train' else 'evaluation_only',width=128,precision='fp32')
        result=ctx.main(finish);statistics=result['statistics'];results[role]=result
    return results


class Bank:
    def __init__(self,folder,spec,task,view,checkpoint,*,technical=False):
        role=view.role
        self.store=Store(Path(folder)/role,'TARGET_BANK',parents=parents(spec,task,checkpoint))
        m=self.store.manifest;self.manifest=m;self.targets=self.store['targets'];self.valid=self.store['valid']
        ix=self.store['indices'];self.indices=ix;self.identities=self.store['identities'];self.labels=self.store['labels']
        validate_population(self.identities,self.labels)
        if (m['task']!=task or m['role']!=role or m['technical']!=technical or m['meaning']!=meaning(task)
                or m['checkpoint']!=checkpoint or m['precision']!='fp32' or m['width']!=128
                or m['rows']!=len(ix) or self.targets.shape!=(len(ix),1 if task in ASSISTANTS else 6 if task=='TR' else 29,128)
                or self.targets.dtype!=np.float32 or self.valid.dtype!=bool or self.valid.shape!=self.targets.shape[:2]
                or ix.dtype!=np.int64 or len(np.unique(ix))!=len(ix) or np.any(ix<0) or np.any(ix>=len(view))
                or (role=='train' and not np.array_equal(ix,np.arange(len(view))))
                or (role=='validation' and not technical and not np.array_equal(ix,panel(view)))
                or not np.array_equal(self.identities,view.identities[ix]) or not np.array_equal(self.labels,view.labels[ix])):
            raise PermissionError('Target bank type, ownership or role/identity join differs')
        if set(self.store.arrays)!={'identities','labels','indices','targets','valid'} or m['usage']!=('optimization' if role=='train' else 'evaluation_only'):
            raise PermissionError('Target reader capability differs')
        if role=='validation':
            training=load_json(Path(folder)/'train/manifest.json');validate(training,'TARGET_BANK',parents(spec,task,checkpoint))
            if training['statistics']!=m['statistics'] or training['technical']!=technical or training['task']!=task:
                raise ValueError('Validation normalization differs from TRAIN')
        statistics=m['statistics']
        if task in ASSISTANTS:
            if statistics!=dict(normalization='none_original_U100_axes'):raise ValueError('Assistant was renormalized')
        else:
            keys=('mean','scale') if task=='TR' else ('mean','scale','whole_mean','whole_scale')
            for k in keys:
                a=np.asarray(statistics[k])
                if a.shape!=(128,) or not np.isfinite(a).all() or ('scale' in k and (a<.001).any()):raise ValueError('Invalid TRAIN moments')
            if task=='TR':
                a=statistics['response_scales']
                if set(a)!={'K','I','T','IT'} or any(not np.isfinite(v) or v<.1 for v in a.values()):raise ValueError('Invalid response scales')
        if task!='TP' and not self.valid.all():raise ValueError('Dense targets must all be valid')
        if task=='TP' and not self.valid[:,28].all():raise ValueError('Whole-jet target must be valid')
        for start in range(0,len(ix),4096):
            a=self.targets[start:start+4096];mask=self.valid[start:start+4096]
            if not np.isfinite(a).all() or (a[~mask]!=0).any():raise ValueError('Nonfinite or nonzero absent targets')


class Targets:
    """Optimization-only accessor, separate from validation diagnostic access."""
    def __init__(self,spec,task,ds,*,parent_bank=None):
        self.spec,self.task,self.ds=spec,task,ds;self.c=cell(task);self.u=ds.u100('train');self.parent=parent_bank
        if self.u.manifest['role']!='train':raise PermissionError('Optimization targets must be TRAIN')
        if bool(self.c['parent']) != bool(parent_bank):raise ValueError('Missing or extraneous parent bank')
        if parent_bank is not None:
            if parent_bank.manifest['role']!='train' or parent_bank.manifest['task']!=self.c['parent']:
                raise PermissionError('Wrong optimization parent')
            join(ds.view('train','D000'),parent_bank)
        self.parents=dict(u100=self.u.manifest['content_hash'])
        if parent_bank is not None:self.parents['predecessor']=parent_bank.manifest['content_hash']

    def batch(self,indices):
        ix=inputs.indices_checked(indices,len(self.ds.view('train','D000')))
        result=dict(u100=self.u['targets'][ix])
        if self.task=='TR':result.update(mean=self.u['mean'].astype(np.float32),scale=self.u['scale'].astype(np.float32))
        if self.parent is None:return result
        result.update(self.extra(ix))
        return result

    def extra(self,ix):
        a=self.parent.targets[ix];c=self.c
        if c['family']=='ladder':return dict(extra=a[:,0])
        if c['family']=='response':return dict(extra=response.targets(a,self.parent.manifest['statistics']['response_scales'],c['queries'],absolute=c['absolute']))
        if c['family']=='region':
            selected=[];groups=[]
            for scale in c['scales']:
                group=[28] if scale=='whole' else list(range(*SLICES[scale]))
                if scale!='whole' and c['source']=='OFFLINE':group=[i+14 for i in group]
                groups.append(list(range(len(selected),len(selected)+len(group))));selected.extend(group)
            return dict(extra=a[:,selected],valid=self.parent.valid[ix][:,selected],groups=groups)
        raise ValueError('Unexpected target consumer')
