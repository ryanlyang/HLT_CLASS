"""Fixed30 science; teacher KL can never act on the deployed class head."""
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from hlt_classification.jetclass2_delphes.runner import _forward, predict, selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from .contracts import artifact, validate, recipe, TEACHERS
from .model import new_model


def objective(logits, labels, auxiliary_logits, predictions, probabilities, targets, *, arm):
    r = recipe(arm)
    if (logits.shape != (len(labels), 11)
            or any(set(d) != set(TEACHERS) for d in (auxiliary_logits, predictions, probabilities, targets))):
        raise ValueError('Wrong head/target inventory')
    ce = F.cross_entropy(logits.float(), labels.long())
    kd, embed = {}, {}
    for t in TEACHERS:
        p, q = auxiliary_logits[t].float(), probabilities[t].detach().float()
        z, target = predictions[t].float(), targets[t].detach().float()
        if (p.shape != logits.shape or q.shape != p.shape or z.shape != (len(labels), 128) or target.shape != z.shape
                or any(not torch.isfinite(v).all() for v in (p, q, z, target))
                or (q < 0).any() or not torch.allclose(q.sum(-1), torch.ones_like(q[:, 0]), rtol=1e-5, atol=1e-6)):
            raise ValueError('Invalid teacher distributions/embeddings')
        kd[t] = 4*F.kl_div((p/2).log_softmax(-1), q, reduction='batchmean')
        embed[t] = F.smooth_l1_loss(z, target, beta=1., reduction='mean')
    loss = .5*ce + sum(1.25*embed[t] for t in TEACHERS)
    # Excluding these tensors entirely avoids zero-gradient AdamW decay at alpha0.
    if r['logit_total']:
        loss = loss + sum(r['logit_weights'][t]*kd[t] for t in TEACHERS)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite required loss')
    return loss, ce, kd, embed


def optimizer(model, config):
    excluded = model.no_weight_decay()
    groups = [dict(params=[p for n, p in model.named_parameters() if p.requires_grad and (n in excluded) == ex],
                   weight_decay=0. if ex else config['weight_decay']) for ex in (False, True)]
    return torch.optim.AdamW(groups, lr=config['peak_lr'], betas=tuple(config['betas']), eps=config['eps'])


def batch_loss(student, raw, bank, ix, arm, device):
    logits = _forward(student, raw, device, bf16=str(device).startswith('cuda'))
    return objective(logits, torch.from_numpy(raw['labels']).to(device), student.auxiliary_logits,
        student.predictions, {t: torch.from_numpy(bank.probabilities[t][ix]).to(device) for t in TEACHERS},
        {t: torch.from_numpy(bank.targets[t][ix]).to(device) for t in TEACHERS}, arm=arm)


def finite_gradients(student):
    values = [torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None]
    if not values or not torch.stack(values).all():
        raise ValueError('Nonfinite required gradients')


def train(tr, val, *, bank, node, device):
    arm = node['arm']; r = recipe(arm)
    if (tr.role != 'train' or val.role != 'validation' or tr.coordinate_name != 'D000' or val.coordinate_name != 'D000'
            or tr.foundation_sha256 != val.foundation_sha256 or not len(tr) or not len(val)
            or node['teacher'] is not None or node['context_coordinate'] is not None
            or not np.array_equal(tr.identities, bank.identities) or not np.array_equal(tr.labels, bank.labels)):
        raise ValueError('Student/target role or identity mismatch')
    student = new_model(node).to(device)
    opt = optimizer(student, r['base'])
    steps = math.ceil(len(tr)/128); update = 0; history = []; best_key = None; selected = None; state = None
    for epoch in range(1, 31):
        student.train(); student.capture = True
        order = np.random.default_rng(np.random.SeedSequence([node['sampler_seed'], epoch])).permutation(len(tr))
        ce_sum = 0.; kd_sum = {t: 0. for t in TEACHERS}; embed_sum = dict(kd_sum)
        started = time.monotonic()
        for start in range(0, len(order), 128):
            ix = order[start:start+128]; update += 1; lr = learning_rate(update/steps)
            for group in opt.param_groups:
                group['lr'] = lr
            opt.zero_grad(set_to_none=True)
            loss, ce, kd, embed = batch_loss(student, tr.batch(ix), bank, ix, arm, device)
            loss.backward(); finite_gradients(student); opt.step()
            ce_sum += float(ce.detach())*len(ix)
            for t in TEACHERS:
                kd_sum[t] += float(kd[t].detach())*len(ix)
                embed_sum[t] += float(embed[t].detach())*len(ix)
            student.predictions = student.auxiliary_logits = None
            del loss, ce, kd, embed
            if (start//128+1) % 128 == 0:
                print(f'AUXLOGIT progress arm={arm} epoch={epoch}/30 batch={start//128+1}/{steps}', flush=True)
        seconds = time.monotonic()-started
        student.capture = False; student.predictions = student.auxiliary_logits = None
        started = time.monotonic()
        metrics = evaluate_probabilities(val.labels, predict(student, val, device=device, batch_size=128))
        row = dict(pass_number=epoch, update=update, learning_rate=lr, validation=metrics,
            ce_loss=ce_sum/len(tr), logit_losses={t: kd_sum[t]/len(tr) for t in TEACHERS},
            embedding_losses={t: embed_sum[t]/len(tr) for t in TEACHERS},
            ce_coefficient=.5, logit_total=r['logit_total'], embedding_total=5.,
            train_seconds=seconds, validation_seconds=time.monotonic()-started)
        row['weighted_ce_loss'] = .5*row['ce_loss']
        row['weighted_logit_losses'] = {t: r['logit_weights'][t]*row['logit_losses'][t] for t in TEACHERS}
        row['weighted_embedding_losses'] = {t: 1.25*row['embedding_losses'][t] for t in TEACHERS}
        row['total_loss'] = row['weighted_ce_loss']+sum(row['weighted_logit_losses'].values())+sum(row['weighted_embedding_losses'].values())
        history.append(row); key = selection_key(metrics, update)
        if best_key is None or key > best_key:
            best_key, selected, state = key, row, student.native_state()
        print(f'AUXLOGIT arm={arm} epoch={epoch}/30 auc={metrics["macro_ovr_auc"]:.8f} '
              f'best_auc={selected["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected["pass_number"]} '
              f'CE={row["ce_loss"]:.6f} weighted_KD={sum(row["weighted_logit_losses"].values()):.6f} '
              f'weighted_embed={sum(row["weighted_embedding_losses"].values()):.6f} train_s={seconds:.2f}', flush=True)
    full = student.state_dict(); full.update(state); student.load_state_dict(full, strict=True)
    student.eval(); student.capture = False; student.predictions = student.auxiliary_logits = None
    report = artifact('TRAINING_REPORT', parents=dict(recipe=r['content_hash'], prepared=tr.foundation_sha256,
        bank=bank.manifest['content_hash']), node=node, scientific_fit=True, technical=False,
        passes=30, validation_history=history, selected_pass=selected['pass_number'], validation=selected['validation'],
        selected_state='native_backbone_only', selected_weights_restored=True, initialization=r['initialization'])
    return student, report, state


def validate_report(report, spec, arm, bank_hash):
    r = recipe(arm); history = report['validation_history']; steps = math.ceil(spec['counts']['train']/128)
    validate(report, 'TRAINING_REPORT', dict(recipe=r['content_hash'], prepared=spec['anchor']['evidence']['prepared'], bank=bank_hash))
    if (report['node'] != spec['nodes'][arm] or report['scientific_fit'] is not True or report['technical'] is not False
            or report['passes'] != 30 or len(history) != 30 or report['selected_state'] != 'native_backbone_only'
            or report['selected_weights_restored'] is not True or report['initialization'] != r['initialization']
            or [x['pass_number'] for x in history] != list(range(1, 31))):
        raise ValueError('Scientific training registration differs')
    for x in history:
        names = ('logit_losses', 'embedding_losses', 'weighted_logit_losses', 'weighted_embedding_losses')
        if any(set(x[k]) != set(TEACHERS) for k in names):
            raise ValueError('Per-teacher loss inventory differs')
        nums = [x[k] for k in ('ce_loss', 'weighted_ce_loss', 'total_loss', 'train_seconds', 'validation_seconds')]
        nums += [v for k in names for v in x[k].values()]
        if (not np.isfinite(nums).all() or min(nums) < -1e-6
                or x['update'] != x['pass_number']*steps or x['learning_rate'] != learning_rate(x['pass_number'])
                or x['validation']['rows'] != spec['counts']['validation']
                or (x['ce_coefficient'], x['logit_total'], x['embedding_total']) != (.5, r['logit_total'], 5.)
                or x['weighted_ce_loss'] != .5*x['ce_loss']
                or any(x['weighted_logit_losses'][t] != r['logit_weights'][t]*x['logit_losses'][t] for t in TEACHERS)
                or any(x['weighted_embedding_losses'][t] != 1.25*x['embedding_losses'][t] for t in TEACHERS)
                or x['total_loss'] != x['weighted_ce_loss']+sum(x['weighted_logit_losses'].values())+sum(x['weighted_embedding_losses'].values())):
            raise ValueError('Epoch loss/schedule/population evidence differs')
    best = max(history, key=lambda x: selection_key(x['validation'], x['update']))
    if report['selected_pass'] != best['pass_number'] or report['validation'] != best['validation']:
        raise ValueError('Main-head full-validation checkpoint selection differs')
