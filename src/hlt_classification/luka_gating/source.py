"""Read-only original-source anchor replay; old schemas are never relabelled."""
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
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as base_recipe
from .contracts import artifact, COUNTS

PARENT_COMMIT = '74aa58c582ff6f497a2c6d91c503f5489fac3e9a'
CONCURRENT_COMMIT = '9bc3cae082915259ad569a46eefa348ddce74ea3'
ATTENTION_COMMIT = '94f560259dd04253c0aa47b31dec3aedc105ac14'
CONCURRENT = {'src/hlt_classification/luka_fullstat/'+n+'.py'
              for n in ('campaign', 'submission', 'worker', 'recovery')}
MODULES = ('__init__', 'contracts', 'model', 'banks', 'training', 'source',
           'campaign', 'worker', 'preflight', 'submission')
ADDITIONS = {'src/hlt_classification/luka_gating/'+n+'.py' for n in (*MODULES, 'followup')}
ATTENTION = {'src/hlt_classification/luka_attention/'+n+'.py' for n in MODULES}
REQUIRED = ADDITIONS | {'scripts/luka_gating.py', 'scripts/queue_luka_gating.sh',
    'sbatch/run_luka_gating.sh', 'tests/test_luka_gating.py', 'tests/test_luka_gating_followup.py',
    'docs/plans/LUKA_PRIVILEGED_GATE_PLAN.md', 'docs/contracts/LUKA_PRIVILEGED_GATE.md',
    'docs/LUKA_PRIVILEGED_GATE_RUNBOOK.md'}

PARENT_CHECK = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullsim import alljets_fusion_campaign as c, alljets_fusion_worker as w
from hlt_classification.luka_fullsim.regular_cache import metadata
s=load_json(sys.argv[1]); _,_,accepted=c.validate_spec(s)
if s['schema_version']!=6: raise ValueError('Expected fixed30 v6')
seen={}; selected={}
for name in ('M0HLT','OFFLINE','DIRECT_D000','COARSE_U100'):
    r=w.completed(s,'train_'+name,seen)
    if r is None: raise ValueError('Parent fit is not complete: '+name)
    selected[name]=dict(receipt=file_ref(Path(s['root'])/('train_'+name)/'receipt.json'),
        checkpoint=file_ref(Path(s['root'])/r['result']['checkpoint']),
        report=file_ref(Path(s['root'])/r['result']['training_report']),
        diagnostics=file_ref(Path(s['root'])/('train_'+name)/'validation_diagnostics.json'))
parent=load_json(s['parent_import']['spec']['path'])
root=parent['admission']['prepared_root']
m,p,_,_=metadata(root)
print('LUKA_GATE_ANCHOR='+json.dumps(dict(prepared_root=root,prepared=p['content_hash'],
    raw_container=load_json(parent['gate_path'])['input_container'],
    environment=accepted['environment'],
    cache=m['content_hash'],models=selected),sort_keys=True,allow_nan=False))
'''


def current(project, commit):
    project = Path(project).resolve()
    value = _source(project, commit)
    tracked = set(subprocess.run(['git','-C',str(project),'ls-files'], check=True,
                                 capture_output=True,text=True).stdout.splitlines())
    if not REQUIRED <= tracked or Path(__file__).resolve()!=project/'src/hlt_classification/luka_gating/source.py':
        raise ValueError('Commit the complete privileged-gate implementation/tests/runbook first')
    return value


def compatible(parent, project):
    old, new = Path(parent['project_dir']), Path(project)
    validate_source_snapshot(parent['source_snapshot'], repository=old, require_clean=True)
    before, after = _files(old), _files(new)
    if not before or not before <= after or not after-before <= ADDITIONS | CONCURRENT | ATTENTION or not ADDITIONS <= after:
        raise ValueError('Unregistered source addition/deletion')
    records = {}
    for name in sorted(before | CONCURRENT | ATTENTION):
        if name in CONCURRENT | ATTENTION:
            pin = ATTENTION_COMMIT if name in ATTENTION else CONCURRENT_COMMIT
            expected = subprocess.run(['git','-C',str(new),'show',pin+':'+name],
                                       check=True,capture_output=True).stdout.replace(b'\r\n',b'\n')
            actual = (new/name).read_bytes().replace(b'\r\n',b'\n')
            if actual != expected:
                raise ValueError('Pinned existing module bytes differ: '+name)
        elif sha256_file(old/name) != sha256_file(new/name):
            raise ValueError('Frozen scientific source differs: '+name)
        records[name] = sha256_file(new/name)
    return records


def authenticate(path, project):
    path = Path(path).resolve(strict=True)
    parent = load_json(path)
    from hlt_classification.luka_fullsim.alljets_fusion_campaign import validate
    validate(parent, 'SPEC')
    if (parent['schema_version']!=6 or parent['source_commit']!=PARENT_COMMIT
            or parent['counts']!=COUNTS or parent['training_recipe']!=base_recipe()
            or path!=Path(parent['root'])/'campaign_spec.json'):
        raise ValueError('Expected original fixed30 trial ALLJETS scientific parent')
    records = compatible(parent, project)
    ref = file_ref(path)
    env = dict(os.environ, PYTHONPATH=str(Path(parent['project_dir'])/'src'),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating regular U100 and controls in their original pinned worktree...', flush=True)
    result = subprocess.run([sys.executable,'-B','-s','-c',PARENT_CHECK,str(path)],
                             cwd=parent['project_dir'],env=env,capture_output=True,text=True)
    if result.returncode:
        raise ValueError('Original-source authentication failed:\n'+result.stderr)
    rows = [s.split('=',1)[1] for s in result.stdout.splitlines() if s.startswith('LUKA_GATE_ANCHOR=')]
    if len(rows)!=1:
        raise ValueError('Missing original-source evidence')
    evidence = json.loads(rows[0])
    if evidence['prepared']!=parent['prepared_sha256']:
        raise ValueError('Anchor population differs')
    validate_file_ref(ref)
    return artifact('ANCHOR', parents=dict(parent=parent['content_hash']), spec=ref,
                     evidence=evidence, compatible_source=records), parent
