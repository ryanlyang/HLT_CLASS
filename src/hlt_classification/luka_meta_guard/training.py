"""Short matched real blocks and isolated adapter phases; TRAIN-only guard."""
from copy import deepcopy
import hashlib
import math
import time
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.runner import predict, selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_meta_embedding import training as old, model as models, meta
from .contracts import artifact, validate, recipe, ARMS, BLOCKS, SALT, events, FIXED_QUERY_CE_ATOL
from .policy import Guard


def split(identities):
    inner, original_query = old.split(identities)
    ranked = sorted(original_query, key=lambda i: hashlib.sha256(identities[i].tobytes()+SALT).digest())
    half = len(ranked)//2
    query, guard = (np.sort(np.asarray(ix, dtype=np.int64)) for ix in (ranked[:half], ranked[half:]))
    if len(inner) < 80 or len(query) < 64 or not len(guard):
        raise ValueError('Insufficient TRAIN-only support/query/guard population')
    return inner, query, guard


def split_hash(identities, parts):
    return hashlib.sha256(b''.join(identities[ix].tobytes() for ix in parts)+SALT).hexdigest()


def order_for(inner, seed, block):
    if not 1 <= block <= BLOCKS:
        raise ValueError('Unregistered real block')
    order = np.random.default_rng(np.random.SeedSequence([seed, 7501, (block+1)//2])).permutation(inner)
    midpoint = (len(order)+1)//2
    return order[:midpoint] if block % 2 else order[midpoint:]


class Subset:
    def __init__(self, cache, indices):
        self.cache, self.indices = cache, indices
        self.labels = cache.labels[indices]
    def __len__(self):
        return len(self.indices)
    def batch(self, indices):
        return self.cache.batch(self.indices[indices])


def score(model, cache, device):
    return evaluate_probabilities(cache.labels, predict(model, cache, device=device, batch_size=128))


def restore_student(model, optimizer, anchor, lr):
    # Intentionally no adapter, adapter optimizer, RNG or sampling rewind here.
    model.load_state_dict(anchor['student'], strict=True)
    optimizer.load_state_dict(deepcopy(anchor['optimizer']))
    for group in optimizer.param_groups:
        group['lr'] = lr


def diagnostics(model, optimizer, adapter, supports, query, lr, seed):
    identity = models.Adapter().to(next(model.parameters()).device)
    losses = []
    for candidate in (identity, adapter):
        value = meta.virtual_query(model, optimizer, candidate, supports, query, [lr]*5, seed=seed)
        losses.append(float(value.detach()))
        del value
    with torch.no_grad():
        z = torch.cat([s['target'] for s in supports])
        residual = adapter(z)-z
        rms, maximum = float(residual.square().mean().sqrt()), float(residual.abs().max())
    result = dict(identity_query_ce=losses[0], adapted_query_ce=losses[1],
        query_ce_delta=losses[1]-losses[0], residual_rms=rms, residual_max=maximum)
    if any(not math.isfinite(v) for v in result.values()) or maximum > .10001:
        raise ValueError('Invalid bounded-adapter diagnostic')
    return result


def validate_diagnostic(d, arm):
    """Retain measured signed evidence; tolerate only FIXED query roundoff."""
    fields = {'identity_query_ce','adapted_query_ce','query_ce_delta','residual_rms','residual_max'}
    if arm not in ARMS or set(d) != fields:
        raise ValueError('Invalid adapter diagnostic fields or arm')
    if (any(not math.isfinite(v) for v in d.values())
            or not 0 <= d['residual_rms'] <= d['residual_max'] <= .10001
            or min(d['identity_query_ce'],d['adapted_query_ce']) < 0
            or d['query_ce_delta'] != d['adapted_query_ce']-d['identity_query_ce']):
        raise ValueError('Invalid signed adapter diagnostic')
    if arm == 'fixed':
        if d['residual_rms'] != 0 or d['residual_max'] != 0:
            raise ValueError('FIXED adapter must leave targets exactly unchanged')
        # Two independently evaluated five-step GPU paths need not be bitwise
        # identical, even with identical targets and restored RNG state. This is
        # an absolute-only consistency allowance, not a performance threshold.
        if abs(d['query_ce_delta']) > FIXED_QUERY_CE_ATOL:
            raise ValueError(f'FIXED query diagnostic differs by {d["query_ce_delta"]:.9g}; '
                             f'absolute tolerance is {FIXED_QUERY_CE_ATOL:.9g}')


def train(model, train_cache, val, bank, node, arm, device, *, initial,
          classifier, mean, scale, parents, publish, technical=False, initializer_source='completed_fixed'):
    registered_recipe=recipe(initializer_source)
    if (arm not in ARMS or train_cache.role != 'train' or val.role != 'validation'
            or train_cache.coordinate_name != 'D000' or val.coordinate_name != 'D000'
            or train_cache.foundation_sha256 != val.foundation_sha256
            or node['coordinate'] != 'D000' or node['teacher'] is not None
            or not np.array_equal(bank.identities, train_cache.identities)
            or not np.array_equal(bank.labels, train_cache.labels)):
        raise PermissionError('Authenticated HLT-only TRAIN join and validation required')
    parts = split(train_cache.identities); inner, query, guard_indices = parts
    train_ids = np.ascontiguousarray(train_cache.identities).view('V32').ravel()
    val_ids = np.ascontiguousarray(val.identities).view('V32').ravel()
    if np.intersect1d(train_ids, val_ids).size:
        raise PermissionError('TRAIN and reporting validation identities overlap')
    guard_cache = Subset(train_cache, guard_indices)
    model.load_state_dict(initial, strict=True); model.to(device)
    for p in model.parameters(): p.requires_grad_(True)
    adapter = models.Adapter().to(device)
    classifier = classifier.to(device).eval()
    for p in classifier.parameters(): p.requires_grad_(False)
    mean, scale = mean.to(device), scale.to(device)
    optimizer = old.optimizer_for(model)
    adapter_optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-4)
    started = time.perf_counter()
    initial_guard = score(model, guard_cache, device)
    policy = Guard(initial_guard['macro_ovr_auc'])
    for group in optimizer.param_groups: group['lr'] = policy.lr
    initial_val = score(model, val, device)
    initialization_seconds = time.perf_counter()-started
    anchor = old.cpu_state(dict(student=model.state_dict(), optimizer=optimizer.state_dict()))
    selected = dict(block=0, validation=deepcopy(initial_val))
    history, snapshots = [], []

    def checkpoint(block):
        value = old.cpu_state(dict(student=model.state_dict(), adapter=adapter.state_dict(),
            optimizer=optimizer.state_dict(), adapter_optimizer=adapter_optimizer.state_dict(), block=block,
            guard=vars(policy), rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
        snapshots.append(publish('selected', block, value))
        return value

    best_state = checkpoint(0)
    for block in range(1, (2 if technical else BLOCKS)+1):
        lr = policy.lr
        for group in optimizer.param_groups: group['lr'] = lr
        started = time.perf_counter(); updates = []; diagnostic = None
        for event in range(events(arm, block)):
            seed = int(np.random.SeedSequence([node['sampler_seed'], 7251, block, event]).generate_state(1)[0])
            support, q = old.meta_batches(train_cache, bank, inner, query, device, seed)
            result = meta.update(model, optimizer, adapter, adapter_optimizer, classifier, mean, scale,
                                 support, q, [lr]*5, seed=seed)
            updates.append(dict(event=event+1, **result))
            del support, q
            if (event+1) % 25 == 0:
                print(f'META_GUARD_ADAPTER arm={arm} block={block} event={event+1}/{events(arm,block)} '
                      f'query_ce={result["query_ce"]:.8f} grad_norm={result["grad_norm"]:.8g}', flush=True)
        # Paired diagnostic after a phase, including frozen phases. This uses query,
        # never guard/validation, and does not update any parameter or real moment.
        seed = int(np.random.SeedSequence([node['sampler_seed'], 9917, block]).generate_state(1)[0])
        support, q = old.meta_batches(train_cache, bank, inner, query, device, seed)
        diagnostic = diagnostics(model, optimizer, adapter, support, q, lr, seed)
        del support, q
        adapter_seconds = time.perf_counter()-started
        # Adapter construction/updates cannot change the real block's RNG stream.
        torch.manual_seed(int(np.random.SeedSequence([node['initialization_seed'], 8153, block]).generate_state(1)[0]))
        model.train(); started = time.perf_counter(); ce_sum = aux_sum = 0.
        order = order_for(inner, node['sampler_seed'], block); batches = math.ceil(len(order)/128)
        for start in range(0, len(order), 128):
            ix = order[start:start+128]; raw = train_cache.batch(ix)
            optimizer.zero_grad(set_to_none=True)
            logits, prediction = models.forward(model, raw, device)
            with torch.no_grad(): target = adapter(torch.from_numpy(bank.targets[ix]).to(device))
            loss, ce, aux = models.objective(logits, torch.from_numpy(raw['labels']).to(device), prediction, target)
            loss.backward(); meta.finite_gradients(model.parameters()); optimizer.step()
            ce_sum += float(ce.detach())*len(ix); aux_sum += float(aux.detach())*len(ix)
            index = start//128+1
            if index % 100 == 0 or index == batches:
                print(f'META_GUARD_PROGRESS arm={arm} block={block}/{BLOCKS} batch={index}/{batches} '
                      f'percent={100*index/batches:.1f}', flush=True)
        train_seconds = time.perf_counter()-started; started = time.perf_counter()
        guard_metrics = score(model, guard_cache, device)
        decision = policy.observe(block, guard_metrics['macro_ovr_auc'])
        if decision['action'] == 'accept':
            anchor = old.cpu_state(dict(student=model.state_dict(), optimizer=optimizer.state_dict()))
        elif decision['action'] == 'rollback':
            restore_student(model, optimizer, anchor, policy.lr)
        guard_seconds = time.perf_counter()-started; started = time.perf_counter()
        metrics = score(model, val, device)
        row = dict(block=block, rows=len(order), batches=batches, learning_rate=lr,
            ce_loss=ce_sum/len(order), embedding_loss=aux_sum/len(order), guard=guard_metrics,
            decision=decision, validation=metrics, meta_updates=updates, diagnostic=diagnostic,
            adapter_seconds=adapter_seconds, train_seconds=train_seconds, guard_seconds=guard_seconds,
            validation_seconds=time.perf_counter()-started)
        row['total_loss'] = row['ce_loss']+5*row['embedding_loss']; history.append(row)
        candidate = dict(block=block, validation=metrics)
        if selection_key(metrics, block) > selection_key(selected['validation'], selected['block']):
            selected = deepcopy(candidate); best_state = checkpoint(block)
        publish('history', block, artifact('BLOCK', parents=parents, arm=arm, technical=technical, row=row))
        print(f'META_GUARD arm={arm} block={block}/{BLOCKS} auc={metrics["macro_ovr_auc"]:.8f} '
              f'best_auc={selected["validation"]["macro_ovr_auc"]:.8f} best_block={selected["block"]} '
              f'guard_auc={guard_metrics["macro_ovr_auc"]:.8f} decision={decision["action"]} '
              f'lr={lr:.8g} query_delta={diagnostic["query_ce_delta"]:+.8g} '
              f'residual_rms={diagnostic["residual_rms"]:.8g}', flush=True)
    model.load_state_dict(best_state['student'], strict=True); model.eval()
    report = artifact('REPORT', parents=dict(parents, recipe=registered_recipe['content_hash']), arm=arm,
        technical=technical, counts=dict(train=len(train_cache), validation=len(val)), node=node,
        populations=dict(inner=len(inner), query=len(query), guard=len(guard_indices)),
        split_sha256=split_hash(train_cache.identities, parts), initial_guard=initial_guard,
        initial_validation=initial_val, initialization_seconds=initialization_seconds,
        history=history, selected=selected, snapshots=snapshots, optimizer_reset=True,
        all_native_trainable=True, selected_full_state=True)
    return report, model.native_state(), best_state


def validate_report(report, spec, parents, arm, *, technical=False):
    validate(report, 'REPORT', dict(parents, recipe=recipe(spec['initializer_source'])['content_hash']))
    n = spec['counts']['train']; q = n//20
    populations = dict(inner=n-q, query=q//2, guard=q-q//2)
    blocks = list(range(1, (2 if technical else BLOCKS)+1))
    if (report['arm'] != arm or arm not in ARMS or report['technical'] is not technical
            or report['counts'] != spec['counts'] or report['node'] != spec['node']
            or report['populations'] != populations or [r['block'] for r in report['history']] != blocks
            or any(report[k] is not True for k in ('optimizer_reset','all_native_trainable','selected_full_state'))):
        raise ValueError('Guarded continuation registration differs')
    policy = Guard(report['initial_guard']['macro_ovr_auc'])
    if report['initial_guard']['rows'] != populations['guard'] or report['initial_validation']['rows'] != spec['counts']['validation']:
        raise ValueError('Initial metric population differs')
    candidates = [dict(block=0, validation=report['initial_validation'])]
    for r in report['history']:
        block = r['block']; expected_rows = (populations['inner']+1)//2 if block % 2 else populations['inner']//2
        if (r['learning_rate'] != policy.lr or r['rows'] != expected_rows
                or r['batches'] != math.ceil(expected_rows/128)
                or r['guard']['rows'] != populations['guard'] or r['validation']['rows'] != spec['counts']['validation']
                or r['decision'] != policy.observe(block, r['guard']['macro_ovr_auc'])
                or r['total_loss'] != r['ce_loss']+5*r['embedding_loss']
                or [m['event'] for m in r['meta_updates']] != list(range(1, events(arm, block)+1))):
            raise ValueError('Guard trajectory/loss/population/events differ')
        for name in ('adapter_seconds','train_seconds','guard_seconds','validation_seconds','ce_loss','embedding_loss','total_loss'):
            if not math.isfinite(r[name]) or r[name] < 0: raise ValueError('Nonfinite/negative report evidence')
        for m in r['meta_updates']:
            if (any(not math.isfinite(m[k]) or m[k] < 0 for k in ('query_ce','teacher_kl','anchor','total','grad_norm'))
                    or not math.isclose(m['total'],m['query_ce']+.1*m['teacher_kl']+m['anchor'],rel_tol=2e-6,abs_tol=1e-7)):
                raise ValueError('Invalid meta-update evidence')
        validate_diagnostic(r['diagnostic'], arm)
        candidates.append(dict(block=block, validation=r['validation']))
    selected = candidates[0]; improvements = [0]
    for candidate in candidates[1:]:
        if selection_key(candidate['validation'],candidate['block']) > selection_key(selected['validation'],selected['block']):
            selected = candidate; improvements.append(candidate['block'])
    if selected != report['selected']:
        raise ValueError('Selection must include initial and post-guard validation only')
    if [r['path'] for r in report['snapshots']] != [f'snapshots/selected-{b:03d}.pt' for b in improvements]:
        raise ValueError('Durable selected-checkpoint inventory differs')
    if not math.isfinite(report['initialization_seconds']) or report['initialization_seconds'] < 0:
        raise ValueError('Invalid initialization timing')
