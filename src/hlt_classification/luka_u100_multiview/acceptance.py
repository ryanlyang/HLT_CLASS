"""Genuine debug miniature and full-support stress before science admission."""
from pathlib import Path
import gc
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from .contracts import artifact, validate, ARMS, TASKS, node as registered_node
from . import views, teachers, training, targets
from .model import new_student, FrozenReadout


class Subset:
    def __init__(self, cache, indices):
        self.cache, self.indices = cache, np.asarray(indices, np.int64)
        for key in ('role', 'coordinate_name', 'foundation_sha256'):
            setattr(self, key, getattr(cache, key))
        self.identities, self.labels = cache.identities[self.indices], cache.labels[self.indices]
    def __len__(self): return len(self.indices)
    def batch(self, indices): return self.cache.batch(self.indices[indices])


def balanced(cache):
    # A metric-complete miniature, not a scientific sample or an admission score.
    ix = np.concatenate([np.flatnonzero(cache.labels == c)[:12] for c in range(11)])
    if any(np.sum(cache.labels[ix] == c) < 2 for c in range(11)):
        raise ValueError('Cannot form real metric-complete miniature')
    return Subset(cache, ix)


def estimate(train_step, eval_step, prepare_seconds, train_rows, val_rows):
    seconds = 1.5 * (30*(math.ceil(train_rows/128)*train_step + math.ceil(val_rows/128)*eval_step)
                     + prepare_seconds + 2*math.ceil(train_rows/128)*eval_step) + 1800
    minutes = max(180, math.ceil(seconds/60))
    if minutes > 1440:
        raise RuntimeError(f'Measured conservative full-support request {minutes} minutes exceeds 24h; review resources')
    return minutes


def stress(model, task, cache, banks, device, validation=None):
    """Longest real full batch + fixed representative batches; no cropped support."""
    length = views.lengths(cache)
    indices = np.argsort(length, kind='stable')[-128:]
    if isinstance(cache, views.PairedCache):
        extremes = [int(views.lengths(cache.primary).argmax()), int(views.lengths(cache.context).argmax())]
        indices = np.asarray(list(dict.fromkeys(extremes + list(indices)))[:128])
    batches = [indices] + [np.random.default_rng(20261029+i).choice(len(cache), 128, replace=False) for i in range(3)]
    opt = training.optimizer(model); step_times = []; eval_times = []
    oracle = task.startswith('teacher_'); name = task.removeprefix('teacher_') if oracle else None
    # Targets for stress only come from registered banks; no generated random substitute.
    readout = None
    if not oracle and ARMS[task]['readout']:
        a = banks['U100'].arrays
        readout = FrozenReadout(a['mean'], a['scale'], a['weight'], a['bias']).to(device)
    torch.cuda.reset_peak_memory_stats()
    for ix in batches:
        raw = cache.batch(ix)
        model.train()
        if not oracle: model.capture = True
        opt.zero_grad(set_to_none=True); torch.cuda.synchronize(); start = time.monotonic()
        loss, _ = training.loss_for(model, task, raw, ix, banks, readout, 3., device)
        loss.backward()
        gs = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
        if not torch.isfinite(loss) or not gs or not torch.stack(gs).all():
            raise ValueError('Real stress has nonfinite loss/gradients')
        opt.step(); torch.cuda.synchronize(); step_times.append(time.monotonic()-start)
        model.eval()
        if not oracle: model.capture = False
        with torch.no_grad():
            torch.cuda.synchronize(); start = time.monotonic()
            out = teachers.forward(model, raw, device, bf16=True, name=name)
            torch.cuda.synchronize(); eval_times.append(time.monotonic()-start)
            if not torch.isfinite(out).all(): raise ValueError('Nonfinite stress inference')
    val_max = None
    if validation is not None:
        val_length = views.lengths(validation)
        val_max = int(val_length.max())
        vi = np.argsort(val_length, kind='stable')[-128:]
        if isinstance(validation, views.PairedCache):
            extremes = [int(views.lengths(validation.primary).argmax()), int(views.lengths(validation.context).argmax())]
            vi = np.asarray(list(dict.fromkeys(extremes + list(vi)))[:128])
        with torch.no_grad():
            torch.cuda.synchronize(); start = time.monotonic()
            out = teachers.forward(model, validation.batch(vi), device, bf16=True, name=name)
            torch.cuda.synchronize(); eval_times.append(time.monotonic()-start)
            if not torch.isfinite(out).all(): raise ValueError('Nonfinite longest-validation inference')
    peak = torch.cuda.max_memory_allocated()
    if peak > .9*torch.cuda.get_device_properties(0).total_memory:
        raise MemoryError('Less than ten percent GPU headroom')
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    return dict(maximum_support=int(length.max()), maximum_validation_support=val_max,
                peak_host_rss_bytes=rss, batch_size=128, truncation=False,
                train_step_seconds=max(step_times), eval_step_seconds=max(eval_times), peak_gpu_bytes=peak)


def run(spec, job):
    if installed_environment() != spec['anchor']['evidence']['environment']:
        raise ValueError('Matched historical installed environment differs; review before accepting')
    root = Path(spec['root'])/'accept'; root.mkdir(exist_ok=False)
    evidence = {}; mini_banks = {}; prepared_seconds = {}
    # Each oracle is built/read/verified independently. No persisted particle caches.
    for name in ('U100', 'CONCAT', 'FUSION'):
        start = time.monotonic()
        tr, va = (views.load(spec, r, name) for r in ('train', 'validation'))
        prepared_seconds[name] = time.monotonic()-start
        node = spec['imported']['teacher']['node'] if name == 'U100' else registered_node(spec['imported'], 'teacher_'+name)
        model = teachers.new_teacher(name, node).to('cuda')
        if name == 'U100':
            ref = spec['imported']['teacher']['files']['checkpoint']
            validate_file_ref(ref)
            model.load_state_dict(torch.load(ref['path'], weights_only=True, map_location='cpu'), strict=True)
            from hlt_classification.luka_u100_embedding.worker import teacher_readback
            # Replay original full ordered batches: BF16 kernels/padding differ on a subset.
            proof = teacher_readback(model, va, spec['imported']['teacher'], 'cuda')
            write_immutable_json(root/'u100_readback.json', proof)
        else:
            evidence['teacher_'+name] = stress(model, 'teacher_'+name, tr, {}, 'cuda', va)
            # Fresh model discards every technical/stress update before the miniature.
            model = teachers.new_teacher(name, node).to('cuda')
            model, _ = training.fit(root/('mini_'+name), spec, 'teacher_'+name, model,
                balanced(tr), balanced(va), node, {}, device='cuda', technical=True)
            training.validate_fit(root/('mini_'+name), spec, 'teacher_'+name, technical=True)
            ref = file_ref(root/('mini_'+name)/'selected_full.pt')
        # Only compact TRAIN bank survives this phase. Used for genuine student stress.
        targets.export(root/('bank_'+name), model, tr, name, ref, device='cuda', source_hash=spec['source_snapshot']['content_hash'])
        mini_banks[name] = ref
        del tr, va, model
        gc.collect(); torch.cuda.empty_cache()
    start = time.monotonic()
    tr, va = (views.load(spec, r, 'D000') for r in ('train', 'validation'))
    student_prepare = time.monotonic()-start
    banks = {n: targets.Bank(root/('bank_'+n), tr, n, ref, source_hash=spec['source_snapshot']['content_hash'])
             for n, ref in mini_banks.items()}
    from types import SimpleNamespace
    for arm, config in ARMS.items():
        node = registered_node(spec['imported'], arm)
        selected_banks = {n: banks[n] for n in config['targets']}
        model = new_student(arm, node).to('cuda')
        evidence[arm] = stress(model, arm, tr, selected_banks, 'cuda', va)
        model = new_student(arm, node).to('cuda')
        tiny = balanced(tr)
        tiny_banks = {n: SimpleNamespace(manifest=b.manifest, arrays={k: v[tiny.indices]
            if k in ('targets', 'logits', 'identities', 'labels') else v for k, v in b.arrays.items()})
            for n, b in selected_banks.items()}
        training.fit(root/('mini_'+arm), spec, arm, model, tiny, balanced(va), node,
                     tiny_banks, device='cuda', technical=True)
        training.validate_fit(root/('mini_'+arm), spec, arm, technical=True)
        del model
        gc.collect(); torch.cuda.empty_cache()
    minutes = dict(export_U100=max(60, math.ceil(1.5*student_prepare/60)+30), report=120)
    for task, row in evidence.items():
        if row['peak_host_rss_bytes'] > spec['memory_mb']*1024**2*.85:
            raise MemoryError('Less than fifteen percent host-RAM headroom')
        prep = prepared_seconds[task.removeprefix('teacher_')] if task.startswith('teacher_') else student_prepare
        minutes[task] = estimate(row['train_step_seconds'], row['eval_step_seconds'], prep,
                                spec['counts']['train'], spec['counts']['validation'])
    # The original selected U100 bank is reusable. Technical C/F target banks
    # never masquerade as targets from fully trained scientific teachers.
    files = [p for p in root.rglob('*') if p.is_file()]
    result = artifact('ACCEPTANCE', parents=dict(spec=spec['content_hash']), job_id=job,
        environment=installed_environment(), gpu=gpu_identity(), counts=spec['counts'],
        evidence=evidence, minutes=minutes, real_data=True, technical_only=True,
        outputs=[file_ref(p, root=root) for p in sorted(files)])
    write_immutable_json(root/'receipt.json', result)
    return result


def accepted(spec):
    root = Path(spec['root'])/'accept'; value = load_json(root/'receipt.json')
    validate(value, 'ACCEPTANCE', dict(spec=spec['content_hash']))
    if (not value['real_data'] or not value['technical_only'] or value['counts'] != spec['counts']
            or set(value['minutes']) != set(TASKS)-{'accept'}
            or set(value['evidence']) != set(ARMS)|{'teacher_CONCAT', 'teacher_FUSION'}
            or any(type(m) is not int or not 1 <= m <= 1440 for m in value['minutes'].values())):
        raise ValueError('Measured acceptance coverage differs')
    expected = {'u100_readback.json'} | {f'bank_{n}/{f}' for n in ('U100', 'CONCAT', 'FUSION') for f in ('manifest.json', 'targets.npz')}
    expected |= {f'mini_{n}/{f}' for n in ('CONCAT', 'FUSION', *ARMS)
                  for f in ('selected.pt', 'selected_full.pt', 'validation_predictions.npz',
                            'training_report.json', 'fit_receipt.json')}
    if {r['path'] for r in value['outputs']} != expected or len(value['outputs']) != len(expected):
        raise ValueError('Acceptance output evidence coverage differs')
    from .retention import validate_output
    for ref in value['outputs']: validate_output(spec, ref, root=root)
    for task in value['evidence']:
        folder = root/('mini_'+task.removeprefix('teacher_') if task.startswith('teacher_') else 'mini_'+task)
        training.validate_fit(folder, spec, task, technical=True)
    from . import submission
    if submission.accepted_jobs(spec, 'accept').get('accept') != value['job_id']:
        raise ValueError('Acceptance receipt is not bound to its exact debug job')
    return value
