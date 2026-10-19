"""Read-only FullStat parent replay in its original pinned source tree."""
from pathlib import Path
import json
import os
import subprocess
import sys

from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_fullsim.preflight_reuse import _files
from hlt_classification.luka_fullstat.recipe import recipe as base_recipe
from hlt_classification.luka_fullstat.contracts import validate as parent_validate
from .contracts import artifact, COUNTS, REFERENCES, TEACHER

PARENT_COMMIT = '9bc3cae082915259ad569a46eefa348ddce74ea3'
DONOR_COMMIT = 'a1d3a2c6caa20871ce2d0b3c1525fd347fd27df6'
MODULES = ('__init__', 'contracts', 'source', 'banks', 'training', 'campaign',
           'worker', 'preflight', 'submission')
ADDITIONS = {f'src/hlt_classification/luka_fullstat_attention/{n}.py' for n in MODULES}
ENTRYPOINTS = {'scripts/luka_fullstat_attention.py', 'scripts/queue_luka_fullstat_attention.sh',
              'sbatch/run_luka_fullstat_attention.sh'}
REQUIRED = ADDITIONS | ENTRYPOINTS | {'tests/test_luka_fullstat_attention.py',
    'docs/plans/LUKA_FULLSTAT_LAST_ATTENTION_PLAN.md',
    'docs/contracts/LUKA_FULLSTAT_LAST_ATTENTION.md', 'docs/LUKA_FULLSTAT_LAST_ATTENTION_RUNBOOK.md'}

PARENT_CHECK = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json,load_npz_arrays
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_fullstat import campaign as c,worker as w
s=load_json(sys.argv[1]); gate,meta,accepted=c.validate_spec(s)
seen={}
if w.completed(s,'complete',seen) is None: raise ValueError('Parent campaign is incomplete')
models={}; identities=labels=None
for name in ('M0HLT','OFFLINE','DIRECT_D000','COARSE_D033','COARSE_D000'):
    directory=Path(s['root'])/('train_'+name)
    report=load_json(directory/'training_report.json')
    best=max(report['validation_history'],key=lambda r:selection_key(r['validation'],r['update']))
    if report['selected_pass']!=best['pass_number'] or report['validation']!=best['validation']:
        raise ValueError('Parent selection differs')
    p=load_npz_arrays(directory/'validation_predictions.npz')
    import numpy as np
    if identities is None: identities,labels=p['identities'],p['labels']
    if not np.array_equal(identities,p['identities']) or not np.array_equal(labels,p['labels']):
        raise ValueError('Parent reference populations differ')
    if evaluate_probabilities(p['labels'],p['probabilities'])!=report['validation']:
        raise ValueError('Parent selected prediction metrics differ')
    models[name]={key:file_ref(directory/filename) for key,filename in
        (('receipt','receipt.json'),('checkpoint','selected.pt'),('report','training_report.json'),
         ('predictions','validation_predictions.npz'))}
    models[name]['selected_pass']=report['selected_pass']
if models['COARSE_D033']['selected_pass']!=17: raise ValueError('Expected frozen D033 epoch17')
runtime=load_json(Path(s['preflight']['path']).parent/'runtime.json')
print('FULLSTAT_ATTENTION_ANCHOR='+json.dumps(dict(cache_root=s['cache_root'],
    foundation_root=s['foundation_root'],prepared=meta[0]['content_hash'],
    foundation_gate=meta[1]['parents']['gate'],raw_container=gate['container'],
    environment=accepted['environment'],cudnn_v8_disabled=runtime['cudnn_v8_disabled'],
    models=models,complete=file_ref(Path(s['root'])/'complete/receipt.json')),sort_keys=True,allow_nan=False))
'''


def current(project, commit):
    project = Path(project).resolve()
    value = _source(project, commit)
    tracked = set(subprocess.run(['git', '-C', str(project), 'ls-files'], check=True,
                                capture_output=True, text=True).stdout.splitlines())
    if not REQUIRED <= tracked or Path(__file__).resolve() != project/'src/hlt_classification/luka_fullstat_attention/source.py':
        raise ValueError('Commit the complete FullStat attention implementation first')
    # Freeze all pre-existing runtime code, including the attention surface and
    # parent view/recipe kernels, against this reviewed in-repository donor.
    changed = set(subprocess.run(['git', '-C', str(project), 'diff', '--name-only',
        DONOR_COMMIT, commit, '--', 'src', 'scripts', 'sbatch'], check=True,
        capture_output=True, text=True).stdout.splitlines())
    if not changed <= ADDITIONS | ENTRYPOINTS:
        raise ValueError('Unregistered change to existing runtime source')
    return value


def compatible(parent, project):
    old, new = Path(parent['project_dir']), Path(project)
    validate_source_snapshot(parent['source_snapshot'], repository=old, require_clean=True)
    before, after = _files(old), _files(new)
    if not before or not before <= after:
        raise ValueError('Parent scientific files are absent')
    records = {}
    for name in sorted(before):
        if sha256_file(old/name) != sha256_file(new/name):
            raise ValueError('Frozen FullStat scientific source differs: '+name)
        records[name] = sha256_file(new/name)
    return records


def authenticate(path, project):
    path = Path(path).resolve(strict=True)
    parent = load_json(path)
    parent_validate(parent, 'RECOVERY_SPEC')
    if (parent['source_commit'] != PARENT_COMMIT or parent['counts'] != COUNTS
            or parent['training_recipe'] != base_recipe(128)
            or path != Path(parent['root'])/'campaign_spec.json'):
        raise ValueError('Expected completed FullStat20 batch128 recovery parent')
    records, ref = compatible(parent, project), file_ref(path)
    env = dict(os.environ, PYTHONPATH=str(Path(parent['project_dir'])/'src'),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating completed FullStat parent and selected D033 in original source...', flush=True)
    result = subprocess.run([sys.executable, '-B', '-s', '-c', PARENT_CHECK, str(path)],
        cwd=parent['project_dir'], env=env, capture_output=True, text=True)
    if result.returncode:
        raise ValueError('Original-source authentication failed:\n'+result.stderr)
    rows = [s.split('=', 1)[1] for s in result.stdout.splitlines() if s.startswith('FULLSTAT_ATTENTION_ANCHOR=')]
    if len(rows) != 1:
        raise ValueError('Missing original-source FullStat evidence')
    evidence = json.loads(rows[0])
    if evidence['prepared'] != parent['prepared_sha256'] or set(evidence['models']) != set(REFERENCES):
        raise ValueError('Parent prepared/reference identity differs')
    validate_file_ref(ref)
    return artifact('ANCHOR', parents=dict(parent=parent['content_hash']), spec=ref,
                    evidence=evidence, compatible_source=records), parent
