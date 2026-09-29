"""Read-only probes of actual benchmark requests, not a queue-time oracle."""
from .dev_submission import scheduler


def probe():
    rows = []
    for partition in ('debug', 'tier3'):
        for cpus, ram, hours in ((4, 32, 4), (4, 64, 8), (8, 64, 8), (16, 64, 8), (1, 32, 4)):
            argv = ['sbatch', '--test-only', '--nodes=1', '--ntasks=1', '--export=NONE',
                f'--partition={partition}', '--account=reu-aisocial', '--qos=qos_tier3',
                f'--cpus-per-task={cpus}', f'--mem={ram}G', f'--time={hours:02}:00:00', '--wrap=true']
            result = scheduler(argv)
            rows.append(dict(partition=partition, cpus=cpus, memory_gib=ram, hours=hours,
                returncode=result.returncode, output=(result.stdout+result.stderr).strip(), argv=argv))
    return dict(read_only=True, jobs_submitted=False, probes=rows,
                caveat='Single-job estimates only. No partition change or automatic choice; ten-job completion may differ.')


def render(row):
    return '\n'.join([f"{r['partition']:<6} {r['cpus']:2} CPUs {r['memory_gib']} GiB {r['hours']}h: {r['output']}"
                      for r in row['probes']]+[row['caveat']])
