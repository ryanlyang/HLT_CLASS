"""Separate torchrun options from rank arguments; parse before expensive I/O."""
from pathlib import Path
import sys
from hlt_classification.data.cache_contracts import with_content_hash
from . import contracts as c,transport


def policy():
    from . import startup
    return with_content_hash(dict(contract='LUKA_BALANCED_RECONSTRUCTION_HYDRA_WORKER_LAUNCH_POLICY/v3',
        schema_version=3,parents={},**c.FLAGS,module='torch.distributed.run',
        argument_boundary='--_before_training_script',installed_parser_check=True,
        checks=['registration_before_population','GPU_worker_before_population_and_inputs'],
        early_gpu_policy=startup.policy(),gpu_transport=transport.policy(),control_backend='gloo',tensor_backend='nccl',
        qualification_minutes=c.QUALIFICATION_MINUTES,science_minutes=c.RESOURCES['minutes']))


def command(project,root,task,node,*,startup_only=False):
    if task!='qualify' and task not in c.FITS:raise ValueError('Registered GPU task required')
    return [sys.executable,'-X','faulthandler','-u','-B','-s','-m','torch.distributed.run',
        '--standalone','--nnodes=1',f'--nproc-per-node={c.RESOURCES["gpus"]}','--max-restarts=0','--',
        str(Path(project)/'scripts/luka_balanced_reconstruction_hydra.py'),
        'rank','--spec',str(Path(root)/'campaign_spec.json'),'--task',task,'--node',str(node)]+(['--startup-only'] if startup_only else [])


def startup_command(project,root,task,node):
    from . import startup
    return ['srun','--exact','--nodes=1','--ntasks=1','--cpus-per-task=12','--mem=64000M',
        '--gres=gpu:geforce_gtx_1080_ti:4','--kill-on-bad-exit=1','--export=ALL',
        f'--time=00:{startup.MINUTES:02d}:00',*command(project,root,task,node,startup_only=True)]


def validate(argv):
    """Exercise the installed torchrun parser without starting processes/devices."""
    from torch.distributed.run import get_args_parser
    args=argv[argv.index('torch.distributed.run')+1:]
    if '--' not in args:raise ValueError('Missing torchrun/worker argument boundary')
    worker=args[args.index('--')+1:]
    try:parsed=get_args_parser().parse_args(args)
    except SystemExit as error:raise ValueError('Installed torchrun rejected worker command') from error
    if (not worker or parsed.training_script!=worker[0] or parsed.training_script_args!=worker[1:]
            or parsed.nproc_per_node!=str(c.RESOURCES['gpus']) or parsed.nnodes!='1'
            or not parsed.standalone or parsed.max_restarts!=0):
        raise ValueError('Installed torchrun changed worker arguments/allocation')
    return dict(training_script=parsed.training_script,training_script_args=parsed.training_script_args,
                processes=parsed.nproc_per_node,nodes=parsed.nnodes)
