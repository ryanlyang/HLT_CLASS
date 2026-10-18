"""Fixed30 warm-start gated fits and registered final replacement losses."""
import math
import time
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.jetclass2_delphes.model import distillation_loss
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from .contracts import artifact,recipe
from .model import centered_gate_loss
from . import banks


def epoch_mode(model, freeze):
    model.train()
    for p in model.mod.parameters(): p.requires_grad_(not freeze)
    if freeze:
        model.mod.eval()  # Freeze buffers/dropout as well as gradients.


def train(model, tr, val, tr_context, val_context, *, bank, node, warm_parents, device, technical=False):
    if (tr.role!='train' or val.role!='validation' or tr.coordinate_name!=node['coordinate']
            or val.coordinate_name!=node['coordinate'] or tr.foundation_sha256!=val.foundation_sha256
            or bank.manifest['alignment']!=banks.alignment(tr) or bank.manifest['teacher']!=node['teacher']
            or model.mode!=node['gate_mode'] or node['gate_weight'] not in (0.,.1)
            or node['freeze_epochs'] not in (0,3)):
        raise ValueError('Training registration/teacher differs')
    for own,ctx in ((tr,tr_context),(val,val_context)):
        banks.paired(own,ctx)
        if (None if ctx is None else ctx.coordinate_name)!=node['context_coordinate']:
            raise ValueError('Training context coordinate differs')
    expected_teacher = (node['coordinate'] if node['branch']=='REPLACEMENT' else
                        {'D066':'U100','D033':'D066','D000':'D033'}[node['coordinate']])
    if bank.manifest['coordinate']!=expected_teacher:
        raise ValueError('Training teacher coordinate differs')
    model.to(device)
    torch.manual_seed(node['initialization_seed'])
    config=recipe()['base']; excluded=model.no_weight_decay()
    groups=[dict(params=[p for n,p in model.named_parameters() if (n in excluded)==ex],
                 weight_decay=0. if ex else config['weight_decay']) for ex in (False,True)]
    optimizer=torch.optim.AdamW(groups,lr=config['peak_lr'],betas=tuple(config['betas']),eps=config['eps'])
    batches=math.ceil(len(tr)/128)
    if not batches or not len(val): raise ValueError('Empty population')
    history=[]; best=None; state=None; updates=0; begin=time.monotonic()
    for epoch in range(1,(1 if technical else 30)+1):
        freeze=epoch<=node['freeze_epochs']; epoch_mode(model,freeze)
        order=np.random.default_rng(np.random.SeedSequence([node['sampler_seed'],epoch])).permutation(len(tr))
        sums=dict(classification=0.,gate=0.,gate_weighted=0.)
        start=time.monotonic()
        for lo in range(0,len(order),128):
            ix=order[lo:lo+128]; raw=tr.batch(ix)
            updates+=1; lr=learning_rate(updates/batches)
            for group in optimizer.param_groups: group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            logits=banks.forward(model,raw,None if tr_context is None else tr_context.batch(ix),device)
            labels=torch.from_numpy(raw['labels']).to(device)
            q=torch.from_numpy(bank.probabilities[ix]).to(device)
            main=distillation_loss(logits,labels,teacher_probabilities=q)
            aux=main.new_zeros(())
            if node['gate_weight']:
                target=torch.from_numpy(bank.factor_batch(ix,raw['mask'].shape[-1])).to(device)
                aux=centered_gate_loss(model.factors,target,torch.from_numpy(raw['mask']).to(device))
            loss=main+node['gate_weight']*aux
            if not torch.isfinite(loss): raise ValueError('Nonfinite gate training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite gate parameter gradients')
            optimizer.step()
            for key,value in [('classification',main),('gate',aux),('gate_weighted',node['gate_weight']*aux)]:
                sums[key]+=float(value.detach())*len(ix)
            model.factors=None
            del logits,loss,main,aux
        seconds=time.monotonic()-start; start=time.monotonic()
        prediction=banks.predict(model,val,val_context,device=device)
        metrics=evaluate_probabilities(val.labels,prediction)
        row=dict(pass_number=epoch,update=updates,learning_rate=lr,backbone_frozen=freeze,
            validation=metrics,losses={k:v/len(tr) for k,v in sums.items()},train_seconds=seconds,
            validation_seconds=time.monotonic()-start)
        history.append(row); key=selection_key(metrics,updates)
        if best is None or key>best:
            best=key; selected=epoch
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f'LUKA gate node={node["node_id"]} epoch={epoch}/{1 if technical else 30} '
              f'auc={metrics["macro_ovr_auc"]:.8f} gate_loss={row["losses"]["gate"]:.6g} '
              f'train_seconds={seconds:.2f} frozen={freeze}',flush=True)
    model.load_state_dict(state,strict=True); model.eval()
    report=artifact('TRAINING_REPORT',parents=dict(recipe=recipe()['content_hash'],
        prepared=tr.foundation_sha256,bank=bank.manifest['content_hash'],**warm_parents),
        node=node,scientific_fit=not technical,passes=len(history),selected_pass=selected,
        validation=history[selected-1]['validation'],validation_history=history,
        runtime_seconds=time.monotonic()-begin,selected_weights_restored=True,
        warm_start_policy=recipe()['warm_start'])
    return report,state
