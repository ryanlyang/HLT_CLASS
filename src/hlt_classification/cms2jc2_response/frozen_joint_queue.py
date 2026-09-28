"""Read-only Slurm resource probes; estimates are not queue-time promises."""
from datetime import datetime
import re

from .dev_submission import scheduler

SHAPES = {'gate': (8, 32, 2), 'evaluation': (16, 64, 4), 'report': (1, 32, 4)}


def probe():
    rows = []
    for partition in ('debug', 'tier3'):
        for task, (cpus, memory, hours) in SHAPES.items():
            argv = ['sbatch', '--test-only', '--nodes=1', '--ntasks=1', '--export=NONE',
                f'--cpus-per-task={cpus}', f'--mem={memory}G', f'--time={hours:02}:00:00',
                f'--partition={partition}', '--account=reu-aisocial', '--qos=qos_tier3', '--wrap=true']
            result = scheduler(argv)
            output = (result.stdout+result.stderr).strip()
            match = re.search(r'\bto start at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\b', output)
            start = match.group(1) if result.returncode == 0 and match else None
            if start is not None:
                datetime.fromisoformat(start)
            rows.append(dict(partition=partition, task=task, cpus=cpus, memory_gib=memory,
                hours=hours, returncode=result.returncode, start=start, output=output, argv=argv))
    eligible = []
    for partition in ('debug', 'tier3'):
        subset = [r for r in rows if r['partition'] == partition]
        evaluation = next(r for r in subset if r['task'] == 'evaluation')
        if all(r['returncode'] == 0 for r in subset) and evaluation['start']:
            eligible.append((evaluation['start'], partition != 'tier3', partition))
    return dict(read_only=True, jobs_submitted=False, probes=rows,
        recommended_partition=min(eligible)[2] if eligible else None,
        caveat='Single-job estimates only; not multi-job completion guarantees. Unknown estimates require an explicit choice.')


def render(row):
    lines = ['Read-only resource tests; no jobs submitted.']
    for r in row['probes']:
        lines.append(f"{r['partition']:<6} {r['task']:<10} {r['cpus']:2} CPUs {r['memory_gib']:2} GiB {r['hours']}h: {r['output']}")
    lines += ['Suggested partition: '+str(row['recommended_partition']), row['caveat']]
    return '\n'.join(lines)
