"""Versioned storage/allocation policy; failed admission keeps its measurements."""
from pathlib import Path
import json
import math
from copy import deepcopy
from hlt_classification.data.cache_contracts import write_immutable_json
from hlt_classification.luka_fullsim.alljets_fusion_sync import PAIR_STORAGE
from .contracts import artifact


def policy():
    return artifact('RUNTIME_POLICY',
        minutes=dict(teacher_CONCAT=2880, teacher_FUSION=2880, train_multi3=1440),
        fusion_pair_storage=deepcopy(PAIR_STORAGE),
        fusion_cross_mask='compact_rectangle_merge_padding_once',
        gpu_peak_fraction=.9, host_peak_fraction=.85, host_reserve_bytes=8*2**30,
        timing_factor=1.5, fixed_margin_minutes=120,
        measurements_before_admission=True, automatic_fallback=False)


def limits(spec):
    p = policy()
    return dict(gpu_peak_limit_bytes=p['gpu_peak_fraction']*spec['reuse']['u100']['gpu']['total_memory_bytes'],
        host_peak_limit_bytes=p['host_peak_fraction']*spec['memory_mb']*1024**2-p['host_reserve_bytes'])


def check_memory(spec, value):
    bounds = limits(spec)
    for key, limit in (('gpu_peak_bytes', bounds['gpu_peak_limit_bytes']),
                       ('peak_rss_bytes', bounds['host_peak_limit_bytes'])):
        peak = value[key]
        if not math.isfinite(peak) or peak <= 0 or peak >= limit:
            raise MemoryError(f'Multiview {key}: measured {peak/2**30:.2f} GiB; '
                              f'limit {limit/2**30:.2f} GiB. See startup_diagnostics; no fallback.')


class Recorder:
    """Diagnostic records are never acceptance receipts or reusable fit outputs."""
    def __init__(self, spec, task, sample):
        self.spec, self.task, self.sample = spec, task, sample
        self.folder = Path(spec['root'])/task/'startup_diagnostics'
        self.index = 0
        self.stage = 'begin'

    def record(self, stage, **fields):
        self.stage = stage
        value = artifact('STARTUP_DIAGNOSTIC', parents=dict(spec=self.spec['content_hash']),
            task_id=self.task, stage=stage, accepted=False,
            runtime_policy=policy(), memory=self.sample(), limits=limits(self.spec), **fields)
        write_immutable_json(self.folder/f'{self.index:02d}-{stage}.json', value)
        self.index += 1
        print('FULLSTAT MULTIVIEW diagnostic '+json.dumps(value, sort_keys=True, allow_nan=False), flush=True)
        return value

    def failure(self, exc):
        self.record('failure', failed_stage=self.stage, error_type=type(exc).__name__, message=str(exc))
