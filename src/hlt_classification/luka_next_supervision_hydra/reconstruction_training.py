"""One native batch, bounded decoder backward, and exact encoder vector-Jacobian injection."""
import time
import numpy as np
import torch
from torch.nn import functional as F
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.luka_cross_message_hydra.training import forward
from .contracts import coefficient
from .reconstruction_contracts import CHUNK
from . import reconstruction_loss as objective
from . import reconstruction_batched as batched
from .probabilistic_targets import conditioning_permutation


def decode_backward(decoder,h,tokens,mask,targets,weight,*,chunk=CHUNK,consistency=False,
        consistency_weight=.1,measure=False):
    """Keep only one decoder chunk's graph. Native h/T remain graph-connected via VJP."""
    sh=h.detach().requires_grad_(True)
    st=tokens.detach().requires_grad_(True) if decoder.cfg['memory']=='tokens' else None
    b=len(h); accumulated=0.; parts={}; forward_seconds=backward_seconds=decoder_seconds=0.; breakdown={}
    sync=lambda:torch.cuda.synchronize(h.device) if measure and h.is_cuda else None
    for start in range(0,b,chunk):
        end=min(start+chunk,b); sync(); began=time.perf_counter()
        scores=decoder(sh[start:end],None if st is None else st[start:end],None if st is None else mask[start:end])
        sync(); decoder_seconds+=time.perf_counter()-began
        losses,fields=batched.rows(scores,targets[start:end],consistency,
            consistency_weight=consistency_weight,field_weights=decoder.cfg.get('field_weights',(1.,1.,1.)),timings=breakdown if measure else None)
        loss=losses.sum()/b
        sync(); forward_seconds+=time.perf_counter()-began; began=time.perf_counter()
        (weight*loss).backward()
        sync(); backward_seconds+=time.perf_counter()-began
        accumulated+=float(loss.detach())
        for k,v in fields.items(): parts[k]=parts.get(k,0.)+float(v.detach().sum())/b
        del scores,losses,fields,loss
    gh=sh.grad; gt=None if st is None else st.grad
    if gh is None or (st is not None and gt is None): raise ValueError('Disconnected reconstruction condition')
    return accumulated,parts,gh,gt,dict(auxiliary_forward=forward_seconds,backward_optimizer=backward_seconds,
        reconstruction_timing=dict(decoder_forward_seconds=decoder_seconds,assignment_seconds=breakdown.get('assignment_seconds',0.),
            decoder_backward_seconds=backward_seconds))


def probes(model,targets,raw,weight,gh,gt):
    h=model.jet.detach(); tokens=model.particle_tokens.detach(); mask=model.particle_mask
    labels=np.asarray(raw['labels']); n=len(labels); perm=np.arange(n)
    for c in np.unique(labels):
        ix=np.where(labels==c)[0]
        if len(ix)>1: perm[ix]=np.roll(ix,1)
    kin=conditioning_permutation(raw)
    result=dict(weighted_auxiliary_jet_gradient_norm=float(gh.norm()),
        auxiliary_jet_gradient_norm=float(gh.norm())/weight if weight else 0.,
        weighted_auxiliary_token_gradient_norm=0. if gt is None else float(gt.norm()),
        auxiliary_token_gradient_norm=0. if gt is None or not weight else float(gt.norm())/weight,
        within_class_swapped_fraction=float(np.mean(perm!=np.arange(n))),
        within_class_kinematic_swapped_fraction=float(np.mean(kin!=np.arange(n))),
        comparison_scope='standardized_field_presence_count_only_consistency_excluded',
        gradient_scope='decoder_condition_VJPs_before_native_chain_rule_token_norm_is_direct_memory_path',
        presence_threshold=.5,validity_threshold=.5,train_aggregate_sample=[])
    with torch.no_grad():
        for name,indices,zero in (('correct',np.arange(n),False),('zero_query',np.arange(n),True),
                ('within_class_shuffle',perm,False),('within_class_kinematic_shuffle',kin,False)):
            loss=0.; count_error=presence=unphysical=0.; tp=predicted=actual=valid_correct=valid_total=0
            jet_error=mass_error=charged_error=0.; valid_fraction=0.
            for start in range(0,n,CHUNK):
                end=min(start+CHUNK,n); ix=torch.as_tensor(indices[start:end],device=h.device)
                q=h[ix]; t=tokens[ix]; m=mask[ix]
                if zero: q=torch.zeros_like(q); t=torch.zeros_like(t); m=torch.zeros_like(m)
                score=model.aux(q,t,m); losses,_=batched.rows(score,targets[start:end],False,field_weights=model.cfg.get('field_weights',(1.,1.,1.)))
                loss+=float(losses.sum())/n
                if name=='correct':
                    count=torch.tensor([len(t['values']) for t in targets[start:end]],device=h.device)
                    count_error+=float((score['count'].argmax(-1)-count).abs().sum())/n
                    presence+=float(score['presence'].sigmoid().sum(-1).sum())/n
                    for j,tgt in enumerate(targets[start:end]):
                        s={k:v[j] for k,v in score.items()}; t=objective.target(tgt,h.device)
                        p4=objective.physical(s,t)
                        unphysical+=float((p4[:,3]<p4[:,:3].norm(dim=-1)).double().mean())/n
                        slots,columns=objective.match(s,t,field_weights=model.cfg.get('field_weights',(1.,1.,1.))); present=s['presence'].sigmoid()>=.5
                        tp+=int(present[slots].sum()); predicted+=int(present.sum()); actual+=len(t['values'])
                        valid_correct+=int(((s['validity'][slots].sigmoid()>=.5)==t['valid'][columns]).sum())
                        valid_total+=4*len(slots); valid_fraction+=float(t['valid'].float().mean())/n if len(slots) else 0.
                        summed=(s['presence'].double().sigmoid()[:,None]*p4).sum(0); truth=t['aggregate']; q=t['frame'][2]
                        jet_error+=float((torch.asinh(summed/q)-torch.asinh(truth[:4]/q)).abs().mean())/n
                        mass=lambda v:(v[3].square()-v[:3].square().sum())/q.square()
                        mass_error+=float((torch.asinh(mass(summed))-torch.asinh(mass(truth[:4]))).abs())/n
                        charged=(s['presence'].double().sigmoid()*p4[:,3]*(1-s['charge'].double().softmax(-1)[:,1])).sum()
                        charged_error+=float((charged/summed[3].clamp(min=q*1e-12)-truth[4]/truth[3].clamp(min=q*1e-12)).abs())/n
                        if start+j<2:
                            result['train_aggregate_sample'].append(dict(identity=bytes(raw['identities'][start+j]).hex(),
                                target_p4=truth[:4].tolist(),predicted_soft_p4=summed.tolist(),target_count=len(slots)))
            result[name+'_loss']=loss
            if name=='correct': result.update(count_mae=count_error,soft_predicted_count=presence,unphysical_slot_fraction=unphysical,
                presence_true_positive=tp,presence_predicted_positive=predicted,presence_actual_positive=actual,
                presence_precision=tp/max(predicted,1),presence_recall=tp/max(actual,1),
                matched_validity_accuracy=valid_correct/max(valid_total,1),target_validity_fraction=valid_fraction,
                jet_p4_asinh_mae=jet_error,signed_mass_squared_asinh_mae=mass_error,charged_energy_fraction_mae=charged_error)
    return result


def step(model,opt,raw,embedding,device,*,candidates,indices,epoch,position,probe=True,measure=False):
    if set(raw)-{'features','vectors','mask','identities','labels'}: raise PermissionError('Unexpected student input')
    if candidates is None or model.aux is None: raise ValueError('Configured full reconstruction targets required')
    model.train(); model.capture=True; opt.zero_grad(set_to_none=True)
    for group in opt.param_groups: group['lr']=learning_rate(position)
    def sync():
        if measure and str(device).startswith('cuda'): torch.cuda.synchronize()
    sync(); began=time.perf_counter(); phases={}
    logits=forward(model,raw,device).float(); labels=torch.as_tensor(raw['labels'],device=device).long()
    target=torch.as_tensor(embedding,device=device).float().detach()
    if target.shape!=model.prediction.shape or not torch.isfinite(target).all(): raise ValueError('Invalid U100 target')
    ce=F.cross_entropy(logits,labels); emb=F.smooth_l1_loss(model.prediction,target); base=ce+5*emb
    sync(); phases['native_forward']=time.perf_counter()-began; began=time.perf_counter()
    targets,evidence=candidates.batch(indices,epoch,model.cfg)
    phases['candidate_sampling']=time.perf_counter()-began
    weight=coefficient(model.cfg,position)
    with torch.autocast(torch.device(device).type,enabled=False):
        aux,parts,gh,gt,times=decode_backward(model.aux,model.jet,model.particle_tokens,model.particle_mask,
            targets,weight,consistency=model.cfg['consistency'],
            consistency_weight=model.cfg.get('consistency_weight',.1),measure=measure)
        reconstruction_timing=times.pop('reconstruction_timing'); phases.update(times); sync(); began=time.perf_counter()
        diagnostics=probes(model,targets,raw,weight,gh,gt) if probe else {}
        if probe:
            # Compare signals at the common global embedding, not total backbone gradients.
            for name,term in (('ce',ce),('u100_w5',5*emb)):
                condition=model.ce_representation if name=='ce' else model.jet
                g=torch.autograd.grad(term,condition,retain_graph=True)[0].detach().float()
                diagnostics[name+'_jet_gradient_norm']=float(g.norm())
                diagnostics[name+'_aux_jet_gradient_cosine']=float(F.cosine_similarity(g.flatten(),gh.flatten(),dim=0,eps=1e-12))
        sync(); phases['diagnostics']=time.perf_counter()-began; began=time.perf_counter()
    if not torch.isfinite(base) or not np.isfinite(aux): raise ValueError('Nonfinite reconstruction objective')
    outputs=[base,model.jet]; gradients=[torch.ones_like(base),gh]
    if gt is not None: outputs.append(model.particle_tokens); gradients.append(gt)
    torch.autograd.backward(outputs,gradients)
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None): raise ValueError('Nonfinite reconstruction gradient')
    opt.step(); sync(); native_backward=time.perf_counter()-began; phases['backward_optimizer']+=native_backward
    reconstruction_timing['native_backward_optimizer_seconds']=native_backward
    losses=np.array([float(ce.detach()),float(emb.detach()),aux,weight*aux,float(base.detach())+weight*aux])
    details=dict(sampling=evidence,probes=diagnostics,reconstruction_losses=parts)
    if measure:
        details['phase_seconds']=dict(phases,input_and_dispatch=0.); details['reconstruction_timing']=reconstruction_timing
    return losses,details
