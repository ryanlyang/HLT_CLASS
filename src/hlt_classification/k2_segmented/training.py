"""Epoch-boundary resume of the frozen K2 AdamW/CE+KD training kernel.

Scientific primitives come from the original pinned checkout, not the current
checkout. No patching of that checkout or of its modules is performed.
"""
from __future__ import annotations

import hashlib
import math
from copy import deepcopy
import random
import time
from pathlib import Path

import numpy as np
import torch

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, load_json, sha256_file, with_content_hash,
    validate_content_hash, write_immutable_json,
)
from hlt_classification.jetclass2_delphes import salience_learned_training as donor
from hlt_classification.jetclass2_delphes.salience_learned_graph import TRAINING, learning_rate


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def rng_state():
    name, keys, position, gaussian, cached = np.random.get_state()
    return dict(python=random.getstate(), numpy=(name, keys.tolist(), position, gaussian, cached),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(value):
    random.setstate(value['python'])
    name, keys, position, gaussian, cached = value['numpy']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(value['torch'])
    if value['cuda']:
        if len(value['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume CUDA RNG device count differs')
        # Drain lazy manual_seed callbacks before restoring Philox state in a
        # fresh process; otherwise a deferred seed can overwrite the restore.
        torch.cuda.init()
        torch.cuda.set_rng_state_all(value['cuda'])


def backend_state():
    """Record execution precision, not just seeds; never change it on resume."""
    return dict(deterministic=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cpu_threads=torch.get_num_threads())


def population_hash(cache):
    return hashlib.sha256(np.ascontiguousarray(cache.identities).tobytes()).hexdigest()


def write_checkpoint(directory, state, *, previous=None):
    """Publish payload then immutable commit marker; never trust an orphan .pt.

Keep all committed checkpoints. The gate budgets their maximum total size,
so no recovery state or user file is deleted by this implementation.
"""
    from io import BytesIO
    directory = Path(directory)
    payload = directory / f"epoch_{state['pass_number']:03d}.pt"
    buffer = BytesIO()
    torch.save(state, buffer)
    atomic_publish_bytes(payload, buffer.getvalue())
    receipt = with_content_hash(dict(contract='K2_SEGMENT_CHECKPOINT/v1', schema_version=1,
        binding=state['binding'], pass_number=state['pass_number'], complete=state['complete'],
        previous_sha256=previous, payload=payload.name, payload_sha256=sha256_file(payload),
        payload_bytes=payload.stat().st_size, final_test_accessed=False))
    write_immutable_json(payload.with_suffix('.json'), receipt)
    return receipt


def load_checkpoint(directory, binding):
    directory = Path(directory)
    commits = sorted(directory.glob('epoch_*.json'))
    if not commits:
        if list(directory.glob('*.pt')):
            raise ValueError('Uncommitted checkpoint payload; reconcile before retry')
        return None, None
    previous = None
    for epoch, path in enumerate(commits, 1):
        receipt = load_json(path)
        validate_content_hash(receipt, expected_contract='K2_SEGMENT_CHECKPOINT/v1', expected_schema_version=1)
        if (receipt['binding'] != binding or receipt['pass_number'] != epoch
                or path.name != f'epoch_{epoch:03d}.json' or receipt['previous_sha256'] != previous
                or receipt['payload'] != f'epoch_{epoch:03d}.pt' or receipt['final_test_accessed']):
            raise ValueError('Checkpoint chain or identity differs')
        previous = receipt['content_hash']
    payload = directory / receipt['payload']
    if sha256_file(payload) != receipt['payload_sha256'] or payload.stat().st_size != receipt['payload_bytes']:
        raise ValueError('Latest checkpoint bytes differ; do not silently roll back')
    state = torch.load(payload, map_location='cpu', weights_only=True)
    if (state['binding'] != binding or state['pass_number'] != receipt['pass_number']
            or state['complete'] != receipt['complete'] or state['format'] != 'K2_FULL_TRAINING_STATE/v1'):
        raise ValueError('Checkpoint payload identity differs')
    if len(list(directory.glob('epoch_*.pt'))) != len(commits):
        raise ValueError('Uncommitted checkpoint payload; reconcile before retry')
    return state, receipt


def train_segment(model, train, validation, *, node, device, teacher, binding,
                  resume=None, maximum=100, acceptance=False, on_epoch=None,
                  can_start_epoch=None):
    """Continue the same fit; callbacks cannot change science or reset patience.

    ``on_epoch`` durably commits the latest state and may request a clean pause.
    ``can_start_epoch`` reserves time for a whole epoch/validation/checkpoint.
    """
    if (node['role'] not in {'reference_ce', 'direct_kd'} or node['context_coordinate'] is not None
            or train.role != 'train' or validation.role != 'validation'
            or train.foundation_sha256 != validation.foundation_sha256 or min(len(train), len(validation)) == 0
            or (not acceptance and maximum != TRAINING['maximum_passes'])
            or (acceptance and maximum not in (1, 2, 3))):
        raise ValueError('Segmented K2 recipe/cache differs')
    if (node['teacher_distribution'] is None) != (teacher is None):
        raise ValueError('CE/KD teacher presence differs')
    if teacher is not None and (teacher.dtype != np.float32 or teacher.shape != (len(train), 11)
            or not np.isfinite(teacher).all() or np.any(teacher < 0)
            or not np.allclose(teacher.sum(-1), 1., atol=2e-6, rtol=0)):
        raise ValueError('Teacher probabilities differ')
    identities = dict(train=population_hash(train), checkpoint=population_hash(validation))
    full_binding = deepcopy(dict(**binding, identities=identities, node=node, maximum=maximum, acceptance=acceptance))
    model.to(device)
    optimizer = donor._optimizer(model)
    state = dict(format='K2_FULL_TRAINING_STATE/v1', binding=full_binding, pass_number=0,
        update=0, history=[], best_state=None, best_metrics=None, best_key=None,
        significant_auc=-math.inf, significant_pass=0, selected_pass=None,
        complete=False, runtime_seconds=0., backend=backend_state())
    if resume is not None:
        if resume['binding'] != full_binding or resume['format'] != state['format']:
            raise ValueError('Resume population/teacher/recipe/node differs')
        state = resume
        if state['backend'] != backend_state():
            raise ValueError('Resume precision/determinism/thread policy differs')
        if not 1 <= state['pass_number'] <= maximum or len(state['history']) != state['pass_number']:
            raise ValueError('Resume epoch/history differs')
        expected_updates = state['pass_number'] * math.ceil(len(train) / 128)
        if state['update'] != expected_updates:
            raise ValueError('Resume optimizer update position differs')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        # Model construction / cache rebuilding may consume RNG; restore LAST.
        restore_rng(state['rng'])
        if state['complete']:
            return state
    started = time.monotonic()
    prior_runtime = state['runtime_seconds']
    for pass_number in range(state['pass_number'] + 1, maximum + 1):
        if can_start_epoch is not None and not can_start_epoch(state):
            if state['pass_number'] == 0:
                raise TimeoutError('No full epoch fits after preparation')
            return state
        order = np.random.default_rng(np.random.SeedSequence([node['sampler_seed'], pass_number])).permutation(len(train))
        model.train()
        train_started = time.monotonic()
        totals = []
        for start in range(0, len(train), 128):
            indices = order[start:start + 128]
            state['update'] += 1
            position = pass_number - 1 + min(1., (start + len(indices)) / len(train))
            lr = learning_rate(position)
            for group in optimizer.param_groups:
                group['lr'] = lr
            raw = train.batch(indices)
            q = None if teacher is None else torch.from_numpy(teacher[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = donor._train_batch(model, raw, node=node, device=device, teacher=q, alpha=1.)
            loss.backward()
            gradients = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if gradients and not torch.stack(gradients).all():
                raise ValueError('Nonfinite segmented K2 gradient')
            optimizer.step()
            totals.append(terms)
        train_seconds = time.monotonic() - train_started
        validation_started = time.monotonic()
        probabilities = donor.predict(model, validation, node=node, device=device, batch_size=128)
        metrics = donor.evaluate_probabilities(validation.labels, probabilities)
        key = donor._selection_key(metrics, state['update'])
        if state['best_key'] is None or key > state['best_key']:
            state.update(best_key=key, best_metrics=metrics, best_state=cpu_tree(model.state_dict()), selected_pass=pass_number)
        if metrics['macro_ovr_auc'] > state['significant_auc'] + TRAINING['minimum_auc_delta']:
            state.update(significant_auc=metrics['macro_ovr_auc'], significant_pass=pass_number)
        row = dict(pass_number=pass_number, update=state['update'], learning_rate=lr,
            alpha_end=1., selection_eligible=True,
            loss_terms={name: float(np.mean([entry[name] for entry in totals])) for name in totals[0]},
            validation=metrics, train_seconds=train_seconds,
            validation_seconds=time.monotonic() - validation_started)
        state['history'].append(row)
        complete = pass_number == maximum or (not acceptance and pass_number >= TRAINING['minimum_passes']
            and pass_number - max(state['significant_pass'], TRAINING['patience_clock_start_pass']) >= TRAINING['patience'])
        state.update(pass_number=pass_number, complete=complete, model=cpu_tree(model.state_dict()),
            optimizer=cpu_tree(optimizer.state_dict()), rng=rng_state(),
            runtime_seconds=prior_runtime + time.monotonic() - started)
        print(f"JC2-K2S node={node['node_id']} pass={pass_number}/{maximum} "
              f"auc={metrics['macro_ovr_auc']:.8f} train_seconds={train_seconds:.2f} "
              f"validation_seconds={row['validation_seconds']:.2f} complete={complete}", flush=True)
        pause = on_epoch(state) if on_epoch is not None else False
        if complete or pause:
            return state
    raise RuntimeError('Segment failed to reach a valid boundary')


def kernel_report(state):
    if not state['complete']:
        raise PermissionError('An intermediate checkpoint is not a completed teacher')
    return dict(node=state['binding']['node'], passes=state['pass_number'],
        selected_pass=state['selected_pass'], validation=state['best_metrics'],
        validation_history=state['history'], runtime_seconds=state['runtime_seconds'],
        batching=dict(training_batch_size=128, inference_batch_size=128, gradient_accumulation_steps=1),
        scientific_fit=not state['binding']['acceptance'], acceptance_only=state['binding']['acceptance'],
        selected_weights_restored=True, rolling_resume_written=True, final_test_accessed=False)
