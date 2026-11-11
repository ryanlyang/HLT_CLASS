"""Exact two-Pascal-GPU production allocation; CPU release tolerates Slurm rounding."""
from hlt_classification.luka_subclass_hydra.benchmark_site import allocation as checked_allocation
from hlt_classification.luka_next_supervision_hydra.reconstruction_release import cpu_allocation

PROFILES=('1080ti_2_fp32',)


def configuration(name='1080ti_2_fp32'):
    if name not in PROFILES: raise ValueError('This execution requires two GTX 1080 Ti GPUs and FP32')
    return dict(gpu_profile=name,world_size=2,precision='fp32',global_batch=128,
        execution_site=dict(cluster='hydra',partition='gpus',account='default',qos='normal',
            gres='gpu:geforce_gtx_1080_ti:2',gpu_name='GeForce GTX 1080 Ti',architecture='x86_64'))


def check_devices(devices):
    if len(devices)!=2: raise ValueError('Two GPUs required')
    for d in devices:
        if (d['name'] not in ('GeForce GTX 1080 Ti','NVIDIA GeForce GTX 1080 Ti')
                or d['compute_capability']!=[6,1] or type(d['total_memory_bytes']) is not int
                or d['total_memory_bytes']<10*2**30): raise ValueError('GPU family/capacity differs')


def allocation(spec):
    if any(spec.get(k)!=v for k,v in configuration().items()): raise ValueError('Execution profile changed')
    devices=checked_allocation(spec); check_devices(devices)
    return devices
