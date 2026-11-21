"""Separate torchrun options from rank arguments; parse before expensive I/O."""
from pathlib import Path
import sys
from . import contracts as c


def policy():
    return c.artifact('WORKER_LAUNCH_POLICY',module='torch.distributed.run',
        argument_boundary='--_before_training_script',installed_parser_check=True,
        checks=['registration_before_population','GPU_worker_before_population_and_inputs'],
        qualification_minutes=c.QUALIFICATION_MINUTES,science_minutes=c.RESOURCES['minutes'])


def command(project,root,task,node):
    if task!='qualify' and task not in c.FITS:raise ValueError('Registered GPU task required')
    return [sys.executable,'-X','faulthandler','-u','-B','-s','-m','torch.distributed.run',
        '--standalone','--nnodes=1',f'--nproc-per-node={c.RESOURCES["gpus"]}','--max-restarts=0','--',
        str(Path(project)/'scripts/luka_balanced_reconstruction_hydra.py'),
        'rank','--spec',str(Path(root)/'campaign_spec.json'),'--task',task,'--node',str(node)]


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
