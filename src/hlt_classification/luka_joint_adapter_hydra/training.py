"""Shared scientific/miniature kernels, explicit oracle evaluation, immutable fits."""
from contextlib import nullcontext
from pathlib import Path
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import (load_json, write_immutable_json, load_npz_arrays,
    atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from .contracts import recipe, settings, artifact, validate
from .model import AdapterParT, loss as correction_loss


def forward(model, raw, device):
    fields = ('features','vectors','mask')
    data = {k:torch.from_numpy(raw[k]).to(device) for k in fields}
    privileged = 'u_features' in raw or 'u_vectors' in raw
    if privileged != (getattr(model,'mode',None) == 'teacher'):
        raise PermissionError('Only the explicit oracle forward may see U100')
    ctx = torch.autocast('cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext()
    with ctx:
        if privileged:
            return model.oracle(**data, **{k:torch.from_numpy(raw[k]).to(device) for k in ('u_features','u_vectors')})
        return model(**data)


def predict(model, cache, device):
    model.eval()
    if hasattr(model,'capture'): model.capture = False; model.supervision = False
    result = np.empty((len(cache),11),np.float32)
    with torch.inference_mode():
        for start in range(0,len(cache),128):
            ix = np.arange(start,min(start+128,len(cache)))
            p = forward(model,cache.batch(ix),device).float().softmax(-1)
            if not torch.isfinite(p).all(): raise ValueError('Nonfinite prediction')
            result[ix] = p.cpu().numpy()
    return result


def optimizer(model):
    config = recipe()['base']; excluded = model.no_weight_decay()
    groups = [dict(params=[p for name,p in model.named_parameters() if p.requires_grad and (name in excluded)==omit],
        weight_decay=0. if omit else config['weight_decay']) for omit in (False,True)]
    return torch.optim.AdamW(groups,lr=config['peak_lr'],betas=tuple(config['betas']),eps=config['eps'])


def step(model,opt,raw,device,*,lr,embedding=None,teacher=None,oracle_raw=None,scale=None,coefficient=0.):
    for group in opt.param_groups: group['lr'] = lr
    target = None
    if model.mode in ('aux','both','final_aux'):
        if teacher is None or oracle_raw is None or scale is None: raise ValueError('Missing frozen oracle targets')
        teacher.eval()
        with torch.no_grad():
            forward(teacher,oracle_raw,device)
            target = teacher.correction_value.detach()
    elif any(v is not None for v in (teacher,oracle_raw,scale)):
        raise PermissionError('CE-only control may not read oracle targets')
    model.train(); model.capture = bool(model.embedding); model.supervision = True
    opt.zero_grad(set_to_none=True)
    logits = forward(model,raw,device); labels = torch.from_numpy(raw['labels']).to(device)
    ce = F.cross_entropy(logits.float(),labels.long()); emb = ce.new_zeros(()); corr = ce.new_zeros(())
    if model.embedding:
        if embedding is None or model.prediction.shape != embedding.shape: raise ValueError('Missing U100 embedding supervision')
        emb = F.smooth_l1_loss(model.prediction,embedding.detach().float())
    elif embedding is not None: raise PermissionError('Standalone fit may not read embedding bank')
    if target is not None:
        corr = correction_loss(model.correction_value,target,torch.from_numpy(raw['mask'][:,0]).to(device),scale)
    total = ce+model.embedding*emb+coefficient*corr
    if not torch.isfinite(total): raise ValueError('Nonfinite loss')
    total.backward()
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError('Nonfinite gradient')
    opt.step()
    return np.array([float(x.detach()) for x in (ce,emb,corr,coefficient*corr,total)])


def rms(teacher, oracle_train, device):
    if oracle_train.role != 'train': raise PermissionError('Target statistics are TRAIN-only')
    teacher.eval(); sums = None; rows = min(recipe()['statistics_rows'],len(oracle_train))
    with torch.no_grad():
        for start in range(0,rows,128):
            ix = np.arange(start,min(start+128,rows)); raw = oracle_train.batch(ix)
            forward(teacher,raw,device); values = teacher.correction_value.float()
            mask = torch.from_numpy(raw['mask'][:,0]).to(device)
            valid = mask[:,:,None] & mask[:,None,:]
            perjet = (values.square()*valid[...,None]).sum((1,2))/valid.sum((1,2))[:,None].clamp_min(1)
            summed = perjet.double().sum(0); sums = summed if sums is None else sums+summed
    scale = (sums/rows).sqrt().clamp_min(recipe()['rms_floor']).float()
    if not torch.isfinite(scale).all(): raise ValueError('Nonfinite TRAIN RMS')
    return scale


def deployment(setting,state,device):
    model = DelphesParticleTransformer() if setting['mode'] in ('aux','final_aux') else AdapterParT(**setting)
    current = model.state_dict()
    expected = {k for k in current if not k.startswith('predictor.')}
    if set(state) != expected: raise ValueError('Deployment tensor coverage differs')
    current.update(state); model.load_state_dict(current,strict=True)
    if hasattr(model,'predictor'): del model.predictor
    return model.to(device).eval()


def fit(folder,spec,task,model,train,val,*,device,bank=None,teacher=None,oracle_train=None,scale=None,
        teacher_parent=None,technical=False):
    folder = Path(folder); folder.mkdir(parents=True,exist_ok=False)
    setting = settings(task); oracle = setting['mode']=='teacher'
    if train.role != 'train' or val.role != 'validation' or train.coordinate_name != 'D000' or val.coordinate_name != 'D000':
        raise PermissionError('Wrong fit roles or views')
    if train.foundation_sha256 != val.foundation_sha256: raise ValueError('Different preparation parents')
    if setting['embedding']:
        if bank is None or not np.array_equal(bank.identities,train.identities) or not np.array_equal(bank.labels,train.labels):
            raise ValueError('Embedding identity alignment differs')
    elif bank is not None: raise PermissionError('Standalone model supplied embedding bank')
    if setting['mode'] in ('aux','both','final_aux') and teacher_parent is None: raise ValueError('Missing selected teacher lineage')
    model.to(device); opt = optimizer(model)
    epochs = 1 if technical else recipe()['epochs']; steps = math.ceil(len(train)/128)
    history=[]; update=0; best=None; selected=None; state=None
    for epoch in range(1,epochs+1):
        order=np.random.default_rng(np.random.SeedSequence([spec['imported']['node']['sampler_seed'],epoch])).permutation(len(train))
        sums=np.zeros(5,np.float64); started=time.perf_counter()
        for start in range(0,len(order),128):
            ix=order[start:start+128]; update+=1
            coefficient=setting['correction_weight']*min(1.,update/(recipe()['correction_ramp_epochs']*steps)) if setting['mode'] in ('aux','both','final_aux') else 0.
            values=step(model,opt,train.batch(ix),device,
                lr=learning_rate(update/steps),
                embedding=None if bank is None else torch.from_numpy(bank.targets[ix]).to(device),teacher=teacher,
                oracle_raw=None if oracle_train is None else oracle_train.batch(ix),scale=scale,coefficient=coefficient)
            sums+=values*len(ix)
            if start//128%100==0 or start+128>=len(order):
                print(f'JOINT_ADAPTER {task} epoch={epoch}/{epochs} batch={start//128+1}/{steps}',flush=True)
        elapsed=time.perf_counter()-started; started=time.perf_counter()
        p=predict(model,val,device); metrics=evaluate_probabilities(val.labels,p)
        ce,emb,corr,weighted,total=(sums/len(train)).tolist()
        row=dict(pass_number=epoch,update=update,validation=metrics,ce_loss=ce,embedding_loss=emb,
            correction_loss=corr,weighted_correction=weighted,total_loss=total,
            learning_rate=learning_rate(epoch),
            train_seconds=elapsed,validation_seconds=time.perf_counter()-started)
        history.append(row)
        key=selection_key(metrics,update)
        if best is None or key>best: best=key; selected=epoch; state=model.deployment_state()
        write_immutable_json(folder/f'epoch_{epoch:02d}.json',artifact('EPOCH',parents=dict(spec=spec['content_hash']),task=task,**row))
        print(f'JOINT_ADAPTER {task} epoch={epoch}/{epochs} auc={metrics["macro_ovr_auc"]:.8f} best_epoch={selected} CE={ce:.6f} embed={emb:.6f} correction={corr:.6f}',flush=True)
    # Restore and publish an executable selected deployment, then reload it independently.
    model=deployment(setting,state,device); save_state(folder/'selected.pt',state)
    restored=deployment(setting,torch.load(folder/'selected.pt',map_location='cpu',weights_only=True),device)
    actual=predict(model,val,device); reloaded=predict(restored,val,device)
    np.testing.assert_allclose(reloaded,actual,rtol=1e-5,atol=1e-6)
    metrics=evaluate_probabilities(val.labels,reloaded)
    if metrics!=history[selected-1]['validation']: raise ValueError('Selected deployment metrics differ')
    parents=dict(spec=spec['content_hash'],recipe=recipe()['content_hash'],prepared=train.foundation_sha256)
    if teacher_parent is not None: parents['teacher']=teacher_parent
    if bank is not None: parents['embedding']=bank.manifest['content_hash']
    statistics=None
    if oracle:
        statistics=dict(scale=rms(model,train,device).cpu().tolist(),rows=min(len(train),recipe()['statistics_rows']),role='train')
    report=artifact('TRAINING_REPORT',parents=parents,task=task,settings=setting,technical=technical,
        counts=dict(train=len(train),validation=len(val)),epochs=epochs,selected_pass=selected,
        validation=metrics,validation_history=history,statistics=statistics,
        privileged_oracle=oracle,joint_backbone_training=True,initialization='scratch_matched_M0HLT',deployable_hlt_only=not oracle)
    atomic_publish_bytes(folder/'validation_predictions.npz',deterministic_npz_bytes(dict(
        identities=val.identities,labels=val.labels,probabilities=reloaded)))
    write_immutable_json(folder/'training_report.json',report)
    value=artifact('FIT',parents=dict(spec=spec['content_hash'],report=report['content_hash']),task=task,reload_verified=True,
        outputs=[file_ref(folder/name,root=folder) for name in ('selected.pt','training_report.json','validation_predictions.npz')])
    write_immutable_json(folder/'receipt.json',value)
    validate_fit(folder,spec,task,technical=technical)
    return report


def validate_fit(folder,spec,task,*,technical=False):
    folder=Path(folder); record=load_json(folder/'receipt.json'); report=load_json(folder/'training_report.json')
    validate(report,'TRAINING_REPORT'); validate(record,'FIT',dict(spec=spec['content_hash'],report=report['content_hash']))
    epochs=1 if technical else 30; setting=settings(task); oracle=setting['mode']=='teacher'
    if (report['parents'].get('spec')!=spec['content_hash'] or report['parents'].get('recipe')!=recipe()['content_hash']
            or report['task']!=task or record['task']!=task or report['settings']!=setting
            or report['technical'] is not technical or report['epochs']!=epochs or not record['reload_verified']
            or report['privileged_oracle']!=oracle or report['deployable_hlt_only']==oracle
            or report['joint_backbone_training'] is not True or report['initialization']!='scratch_matched_M0HLT' or (not technical and report['counts']!=spec['counts'])
            or [r['path'] for r in record['outputs']]!=['selected.pt','training_report.json','validation_predictions.npz']):
        raise ValueError('Fit registration differs')
    expected_parents={'spec','recipe','prepared'}
    if setting['embedding']: expected_parents.add('embedding')
    if setting['mode'] in ('aux','both','final_aux'): expected_parents.add('teacher')
    if set(report['parents'])!=expected_parents: raise ValueError('Unexpected or missing fit parents')
    if oracle:
        statistics=report['statistics']; scales=np.asarray(statistics['scale'])
        if (statistics['role']!='train' or statistics['rows']!=min(report['counts']['train'],recipe()['statistics_rows'])
                or scales.shape!=(8 if setting['site']=='bias' else 4,) or not np.isfinite(scales).all()
                or (scales<recipe()['rms_floor']-1e-8).any()): raise ValueError('TRAIN oracle normalization differs')
    elif report['statistics'] is not None: raise ValueError('Student must not fit target statistics')
    for ref in record['outputs']: validate_file_ref(ref,root=folder)
    history=report['validation_history']; steps=math.ceil(report['counts']['train']/128)
    if [r['pass_number'] for r in history]!=list(range(1,epochs+1)): raise ValueError('Epochs incomplete')
    for r in history:
        numbers=[r[k] for k in ('ce_loss','embedding_loss','correction_loss','weighted_correction','total_loss','train_seconds','validation_seconds')]
        if (not np.isfinite(numbers).all() or min(numbers)<0 or r['update']!=r['pass_number']*steps
                or r['validation']['rows']!=report['counts']['validation']
                or r['learning_rate']!=(learning_rate(r['pass_number']))
                or r['weighted_correction']>setting['correction_weight']*r['correction_loss']+1e-6
                or (not setting['embedding'] and r['embedding_loss']!=0)
                or not math.isclose(r['total_loss'],r['ce_loss']+setting['embedding']*r['embedding_loss']+r['weighted_correction'],rel_tol=1e-6,abs_tol=1e-6)
                or (setting['mode'] in ('ce','teacher') and (r['correction_loss']!=0 or r['weighted_correction']!=0))):
            raise ValueError('Loss/schedule/population evidence differs')
    best=max(history,key=lambda r:selection_key(r['validation'],r['update']))
    if best['pass_number']!=report['selected_pass'] or best['validation']!=report['validation']: raise ValueError('Selection differs')
    a=load_npz_arrays(folder/'validation_predictions.npz'); validate_predictions(a,report['counts']['validation'])
    if evaluate_probabilities(a['labels'],a['probabilities'])!=report['validation']: raise ValueError('Saved metrics differ')
    return report,a
