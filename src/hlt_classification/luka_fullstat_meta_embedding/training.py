"""Paired TRAIN-only continuation and student-benefit target adaptation."""
from copy import deepcopy
import hashlib
import math
import time
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.runner import predict, selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities

from .contracts import artifact, validate, recipe, ARMS, SPLIT_SALT, learning_rate, meta_due
from .model import forward, objective, Adapter
from hlt_classification.luka_meta_embedding import meta


def optimizer_for(model):
    config = recipe()['base']; excluded = model.no_weight_decay()
    groups = [dict(params=[p for n,p in model.named_parameters()
        if p.requires_grad and (n in excluded) == exclude],
        weight_decay=0. if exclude else config['weight_decay']) for exclude in (False, True)]
    return torch.optim.AdamW(groups, lr=3e-5, betas=tuple(config['betas']), eps=config['eps'])


def split(identities):
    if (identities.dtype != np.uint8 or identities.ndim != 2 or identities.shape[1] != 32
            or len(np.unique(identities,axis=0)) != len(identities)):
        raise ValueError('Unique authenticated TRAIN identities required')
    keys = [hashlib.sha256(row.tobytes()+SPLIT_SALT).digest() for row in identities]
    ranked = sorted(range(len(keys)),key=keys.__getitem__)
    n = len(keys)//20
    if not n: raise ValueError('Insufficient TRAIN population for meta split')
    query = np.sort(np.array(ranked[:n],dtype=np.int64))
    inner = np.sort(np.array(ranked[n:],dtype=np.int64))
    inner.flags.writeable = query.flags.writeable = False
    return inner,query


def split_hash(identities, inner, query):
    return hashlib.sha256(identities[inner].tobytes()+identities[query].tobytes()+SPLIT_SALT).hexdigest()


def cpu_state(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_state(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_state(v) for v in value)
    return value


def calibrate(model, train, bank, node, device):
    if (train.role!='train' or train.coordinate_name!='D000'
            or not np.array_equal(bank.identities,train.identities)
            or not np.array_equal(bank.labels,train.labels)):
        raise PermissionError('Calibration requires authenticated TRAIN HLT/target join')
    inner,query = split(train.identities)
    before = model.native_state()
    model.to(device).eval()
    for n,p in model.named_parameters(): p.requires_grad_(n.startswith('predictor.'))
    optimizer = torch.optim.AdamW(model.predictor.parameters(),lr=3e-4,weight_decay=0.)
    history = []
    for epoch in range(1,4):
        order = np.random.default_rng(np.random.SeedSequence([node['sampler_seed'],4821,epoch])).permutation(inner)
        total = 0.
        for start in range(0,len(order),128):
            ix = order[start:start+128]; optimizer.zero_grad(set_to_none=True)
            _,prediction = forward(model,train.batch(ix),device)
            target = torch.from_numpy(bank.targets[ix]).to(device)
            loss = torch.nn.functional.smooth_l1_loss(prediction,target)
            if not torch.isfinite(loss): raise ValueError('Nonfinite calibration')
            loss.backward(); meta.finite_gradients(model.predictor.parameters()); optimizer.step()
            total += float(loss.detach())*len(ix)
        history.append(total/len(inner))
        print(f'FULLSTAT_META_EMBED calibration={epoch}/3 train_embedding_loss={history[-1]:.8f}',flush=True)
    if any(not torch.equal(v,model.native_state()[k]) for k,v in before.items()):
        raise ValueError('Calibration changed a native parameter or buffer')
    for p in model.parameters(): p.requires_grad_(True)
    return cpu_state(model.state_dict()), dict(history=history,inner_rows=len(inner),query_rows=len(query),
        split_sha256=split_hash(train.identities,inner,query),native_unchanged=True)


def batch(cache, bank, ix, device, *, target):
    raw = cache.batch(ix)
    out = dict(inputs={k:torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')},
               labels=torch.from_numpy(raw['labels']).to(device))
    if target: out['target'] = torch.from_numpy(bank.targets[ix]).to(device)
    return out


def meta_batches(train, bank, inner, query, device, seed):
    if train.role!='train' or train.coordinate_name!='D000' or np.intersect1d(inner,query).size:
        raise PermissionError('Meta supports/query must be disjoint subsets of TRAIN HLT')
    return _sample_batches(train,bank,inner,query,device,seed)


def _sample_batches(train,bank,inner,query,device,seed):
    # Private hot path; train() supplies the authenticated, disjoint split once.
    # Do not sort/intersect three million row indices at each meta event.
    if len(inner) < 80 or len(query) < 64:
        raise ValueError('Five distinct support batches16 and query64 required')
    rng = np.random.default_rng(seed)
    support_ix = rng.choice(inner,80,replace=False).reshape(5,16)
    query_ix = rng.choice(query,64,replace=False)
    return ([batch(train,bank,ix,device,target=True) for ix in support_ix],
            batch(train,bank,query_ix,device,target=False))


def train(model, train_cache, val, bank, node, arm, device, *, initial,
          classifier, mean, scale, parents, technical=False):
    if (arm not in ARMS or train_cache.role != 'train' or val.role != 'validation'
            or train_cache.coordinate_name != 'D000' or val.coordinate_name != 'D000'
            or train_cache.foundation_sha256 != val.foundation_sha256
            or node['coordinate'] != 'D000' or node['teacher'] is not None
            or node.get('context_coordinate') is not None or node['auxiliary_weight'] != 5.
            or not np.array_equal(bank.identities,train_cache.identities)
            or not np.array_equal(bank.labels,train_cache.labels)):
        raise PermissionError('Authenticated TRAIN HLT/target join and validation required')
    inner,query = split(train_cache.identities)
    model.load_state_dict(initial,strict=True); model.to(device)
    for p in model.parameters(): p.requires_grad_(True)
    torch.manual_seed(node['initialization_seed'])
    adapter = Adapter().to(device)
    classifier = classifier.to(device).eval()
    for p in classifier.parameters(): p.requires_grad_(False)
    mean,scale = mean.to(device),scale.to(device)
    optimizer = optimizer_for(model)
    adapter_optimizer = torch.optim.Adam(adapter.parameters(),lr=1e-4)
    steps = math.ceil(len(inner)/128); history=[]; selected=None; best_state=None
    epochs = [3] if technical else range(1,21)
    for epoch in epochs:
        model.train(); started=time.perf_counter(); ce_sum=aux_sum=0.; meta_rows=[]
        order=np.random.default_rng(np.random.SeedSequence([node['sampler_seed'],epoch])).permutation(inner)
        for start in range(0,len(order),128):
            index=start//128+1; position=(epoch-1)+index/steps
            lr=learning_rate(position)
            for group in optimizer.param_groups: group['lr']=lr
            if arm=='meta' and meta_due(epoch,index):
                seed=int(np.random.SeedSequence([node['sampler_seed'],epoch,index,917]).generate_state(1)[0])
                supports,query_batch=_sample_batches(train_cache,bank,inner,query,device,seed)
                rates=[learning_rate(min(20.,position+i/steps)) for i in range(5)]
                row=meta.update(model,optimizer,adapter,adapter_optimizer,classifier,mean,scale,
                                supports,query_batch,rates,seed=seed)
                meta_rows.append(dict(batch=index,**row))
                del supports,query_batch
            ix=order[start:start+128]; optimizer.zero_grad(set_to_none=True)
            raw=train_cache.batch(ix); logits,prediction=forward(model,raw,device)
            with torch.no_grad(): target=adapter(torch.from_numpy(bank.targets[ix]).to(device))
            loss,ce,aux=objective(logits,torch.from_numpy(raw['labels']).to(device),prediction,target)
            loss.backward(); meta.finite_gradients(model.parameters()); optimizer.step()
            ce_sum+=float(ce.detach())*len(ix); aux_sum+=float(aux.detach())*len(ix)
            if index%100==0 or index==steps:
                print(f'FULLSTAT_META_EMBED_PROGRESS arm={arm} epoch={epoch}/20 batch={index}/{steps} '
                      f'percent={100*index/steps:.1f}',flush=True)
        train_seconds=time.perf_counter()-started; started=time.perf_counter()
        metrics=evaluate_probabilities(val.labels,predict(model,val,device=device,batch_size=128))
        row=dict(pass_number=epoch,update=epoch*steps,learning_rate=lr,validation=metrics,
            ce_loss=ce_sum/len(inner),auxiliary_loss=aux_sum/len(inner),
            meta_updates=meta_rows,train_seconds=train_seconds,validation_seconds=time.perf_counter()-started)
        row['total_loss']=row['ce_loss']+5*row['auxiliary_loss']; history.append(row)
        if selected is None or selection_key(metrics,row['update'])>selection_key(selected['validation'],selected['update']):
            selected=deepcopy(row)
            best_state=cpu_state(dict(student=model.state_dict(),adapter=adapter.state_dict(),
                optimizer=optimizer.state_dict(),adapter_optimizer=adapter_optimizer.state_dict(),epoch=epoch,
                rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
        print(f'FULLSTAT_META_EMBED arm={arm} epoch={epoch}/20 auc={metrics["macro_ovr_auc"]:.8f} '
              f'best_auc={selected["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected["pass_number"]} '
              f'CE={row["ce_loss"]:.7f} embed={row["auxiliary_loss"]:.7f} meta_updates={len(meta_rows)}',flush=True)
    model.load_state_dict(best_state['student'],strict=True); model.eval()
    report=artifact('REPORT',parents=dict(parents,recipe=recipe()['content_hash']),arm=arm,
        technical=technical,counts=dict(train=len(train_cache),validation=len(val)),
        inner_rows=len(inner),query_rows=len(query),split_sha256=split_hash(train_cache.identities,inner,query),
        node=node,history=history,selected=selected,selected_pass=selected['pass_number'],
        optimizer_reset=True,all_native_trainable=True,selected_full_state=True)
    return report,model.native_state(),best_state


def validate_report(report,spec,parents,arm,*,technical=False):
    validate(report,'REPORT',dict(parents,recipe=recipe()['content_hash']))
    expected_epochs=[3] if technical else list(range(1,21)); n=spec['counts']['train']
    steps=math.ceil((n-n//20)/128)
    if (report['arm']!=arm or arm not in ARMS or report['technical'] is not technical
            or report['counts']!={r:spec['counts'][r] for r in ('train','validation')} or report['node']!=spec['node']
            or report['query_rows']!=n//20 or report['inner_rows']!=n-n//20
            or any(report[k] is not True for k in ('optimizer_reset','all_native_trainable','selected_full_state'))
            or [r['pass_number'] for r in report['history']]!=expected_epochs):
        raise ValueError('Continuation registration differs')
    for r in report['history']:
        expected_meta=[i for i in range(1,steps+1) if arm=='meta' and meta_due(r['pass_number'],i)]
        if (r['update']!=r['pass_number']*steps or r['learning_rate']!=learning_rate(r['pass_number'])
                or r['validation']['rows']!=spec['counts']['validation']
                or r['total_loss']!=r['ce_loss']+5*r['auxiliary_loss']
                or any(not math.isfinite(r[k]) or r[k]<0 for k in
                       ('ce_loss','auxiliary_loss','total_loss','train_seconds','validation_seconds'))
                or [m['batch'] for m in r['meta_updates']]!=expected_meta):
            raise ValueError('Epoch loss/schedule/meta population differs')
        for m in r['meta_updates']:
            if (any(not math.isfinite(m[k]) or m[k]<0 for k in ('query_ce','teacher_kl','anchor','total','grad_norm'))
                    or not math.isclose(m['total'],m['query_ce']+.1*m['teacher_kl']+m['anchor'],rel_tol=2e-6,abs_tol=1e-7)):
                raise ValueError('Invalid meta evidence')
    best=max(report['history'],key=lambda r:selection_key(r['validation'],r['update']))
    if best!=report['selected'] or best['pass_number']!=report['selected_pass']:
        raise ValueError('Selection differs from registered full validation rule')
