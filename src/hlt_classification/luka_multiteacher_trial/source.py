"""Original-source teacher authentication; retired attention banks stay retired."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref, file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_embedding_aux.source import authenticate
from hlt_classification.luka_embedding_aux.contracts import validate as validate_anchor
from hlt_classification.luka_attention_weight.source import replay
from hlt_classification.luka_u100_embedding.source import teacher as u100_teacher
from hlt_classification.luka_d066_embedding.source import reference
from hlt_classification.luka_d066_embedding.contracts import validate as validate_reference
from hlt_classification.luka_u100_embedding.direct import resources, validate_resources
from hlt_classification.luka_u100_embedding.contracts import node as donor_node
from .contracts import artifact, validate, TEACHERS

BASE = 'd98359f76abe3a46c70f948aa4b99d51c82787e9'
MODULES = ('__init__', 'contracts', 'source', 'model', 'banks', 'training', 'worker', 'campaign', 'submission')
ADDITIONS = {'src/hlt_classification/luka_multiteacher_trial/'+n+'.py' for n in MODULES}
REQUIRED = ADDITIONS | {'scripts/luka_multiteacher_trial.py', 'scripts/queue_luka_multiteacher_trial.sh',
    'sbatch/run_luka_multiteacher_trial.sh', 'tests/test_luka_multiteacher_trial.py',
    'docs/plans/LUKA_MULTITEACHER_TRIAL_PLAN.md', 'docs/contracts/LUKA_MULTITEACHER_TRIAL.md',
    'docs/LUKA_MULTITEACHER_TRIAL_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git','-C',str(project),*args], check=True, capture_output=True, text=True).stdout.splitlines()
    missing = REQUIRED-set(git('ls-files'))
    if missing:
        raise ValueError('Commit complete trial implementation: '+', '.join(sorted(missing)))
    bad = [r for r in git('diff','--name-status','--no-renames',BASE,commit,'--','src')
           if not r.startswith('A\t') or r[2:] not in ADDITIONS]
    if bad:
        raise ValueError('Existing runtime changed or unreviewed source additions:\n'+'\n'.join(bad))
    if Path(__file__).resolve() != project/'src/hlt_classification/luka_multiteacher_trial/source.py':
        raise ValueError('Imported code escapes pinned source')
    return snapshot


ARCHIVE_CHECK = '''
import json,sys,runpy
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_attention.contracts import validate
a=runpy.run_path(sys.argv[2])['read_archive'](sys.argv[1])
s=load_json(sys.argv[1]); g=load_json(s['gate_path']); root=Path(s['root'])
teachers={}
for name,coordinate in (('LOGIT_D066','D066'),('LOGIT_D033','D033')):
    n=next(n for n in g['nodes'] if n['node_id']==name)
    if (n['coordinate'],n['branch'],n['attention_coefficient'])!=(coordinate,'LOGIT',0.):
        raise ValueError('Expected original LOGIT teacher')
    task='train_'+name; r=load_json(root/task/'receipt.json')
    validate(r,'TASK',dict(spec=s['content_hash']))
    if r['content_hash']!=a['archive']['training_receipts'][task]:
        raise ValueError('Archived selected receipt differs')
    matches=[x for x in r['outputs'] if x['path']==task+'/selected.pt']
    if len(matches)!=1: raise ValueError('Missing selected checkpoint binding')
    validate_file_ref(matches[0],root=root)
    teachers[name]=dict(node=n,files={k:file_ref(root/task/f) for k,f in dict(
        checkpoint='selected.pt',report='training_report.json',predictions='validation_predictions.npz',receipt='receipt.json').items()})
print('MULTI_TEACHERS='+json.dumps(dict(teachers=teachers,anchor=a['anchor'],prepared=a['prepared'],
    archive=a['archive']),sort_keys=True,allow_nan=False))
'''


def teachers(anchor, attention_path):
    p = Path(attention_path).resolve(strict=True); spec = load_json(p)
    if spec['source_commit'] != '94f560259dd04253c0aa47b31dec3aedc105ac14' or p != Path(spec['root'])/'campaign_spec.json':
        raise ValueError('Expected original94f56025 LOGIT ladder')
    gate = load_json(spec['gate_path'])
    validate_source_snapshot(gate['source_snapshot'], repository=spec['project_dir'], require_clean=True)
    helper = Path(__file__).parents[1]/'luka_attention_weight/reference_archive.py'
    ref, evidence = replay(p, spec['project_dir'], ARCHIVE_CHECK, 'MULTI_TEACHERS=', extra=(helper,))
    if (evidence['prepared'] != anchor['evidence']['prepared'] or evidence['anchor']['spec'] != anchor['spec']
            or any(evidence['anchor']['evidence']['models'][n][k] != anchor['evidence']['models'][n][k]
                   for n in ('M0HLT','OFFLINE') for k in ('checkpoint','report','receipt','diagnostics'))):
        raise ValueError('Teacher and CE population/lineage differ')
    u = u100_teacher(anchor)
    parent = load_json(anchor['spec']['path'])
    output = {}
    for name, co in TEACHERS.items():
        files = (anchor['evidence']['models']['OFFLINE'] if name == 'OFFLINE' else
                 u['files'] if name == 'COARSE_U100' else evidence['teachers'][name]['files'])
        files = {k: files[k] for k in ('checkpoint','report','predictions','receipt')}
        original = anchor['spec'] if name in ('OFFLINE','COARSE_U100') else ref
        node = (next(n for n in parent['nodes'] if n['node_id']=='OFFLINE') if name=='OFFLINE' else
                u['node'] if name=='COARSE_U100' else evidence['teachers'][name]['node'])
        output[name] = artifact('TEACHER', parents=dict(anchor=anchor['content_hash']),
            model_id=name, coordinate=co, node=node, prepared=anchor['evidence']['prepared'], role='train',
            spec=original, files=files, selected_native_checkpoint_revalidated=True)
        for item in files.values(): validate_file_ref(item)
    return output


U100_REFERENCE = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_u100_embedding import direct
s=load_json(sys.argv[1]); direct.results(s)
r=Path(s['root'])/'fit'
print('MULTI_U100_REF='+json.dumps(dict(anchor=s['anchor'],node=s['node'],counts=s['counts'],
    files={k:file_ref(r/f) for k,f in dict(report='training_report.json',predictions='validation_predictions.npz').items()}),sort_keys=True))
'''


def optional_u100(path, anchor, parent):
    if path is None:
        return None
    saved = load_json(path)
    validate_source_snapshot(saved['source_snapshot'], repository=saved['project_dir'], require_clean=True)
    ref, data = replay(path, saved['project_dir'], U100_REFERENCE, 'MULTI_U100_REF=')
    if data['anchor'] != anchor or data['node'] != donor_node(parent) or data['counts'] != parent['counts']:
        raise ValueError('U100-w5 reference is not the matched completed direct fit')
    return artifact('REFERENCE', parents=dict(anchor=anchor['content_hash']), spec=ref, files=data['files'])


def imports(spec):
    anchor = spec['anchor']
    validate_anchor(anchor, 'ANCHOR')
    validate_reference(spec['reference'], 'REFERENCE')
    parent = load_json(validate_file_ref(anchor['spec']))
    historical, registered = dict(spec['reference']['node']), donor_node(parent)
    for n in (historical, registered):
        n.pop('node_id'); n.pop('branch')
    if historical != registered or spec['reference']['parents']['anchor'] != anchor['content_hash']:
        raise ValueError('Historical w5 recipe/anchor differs')
    validate_file_ref(anchor['spec'])
    if set(spec['teachers']) != set(TEACHERS):
        raise ValueError('Teacher set differs')
    for name, t in spec['teachers'].items():
        validate(t, 'TEACHER', dict(anchor=anchor['content_hash']))
        if (t['model_id'] != name or t['coordinate'] != TEACHERS[name] or t['role'] != 'train'
                or t['node']['node_id'] != name or t['node']['coordinate'] != TEACHERS[name]
                or t['node']['context_coordinate'] is not None
                or t['prepared'] != anchor['evidence']['prepared'] or not t['selected_native_checkpoint_revalidated']
                or set(t['files']) != {'checkpoint','report','predictions','receipt'}):
            raise ValueError('Teacher identity differs')
        validate_file_ref(t['spec'])
        for ref in t['files'].values(): validate_file_ref(ref)
    for t in anchor['evidence']['models'].values():
        for ref in t.values(): validate_file_ref(ref)
    for ref in spec['reference']['files'].values(): validate_file_ref(ref)
    validate_file_ref(spec['reference']['spec'])
    if spec['u100_reference'] is not None:
        r = spec['u100_reference']; validate(r, 'REFERENCE', dict(anchor=anchor['content_hash']))
        validate_file_ref(r['spec'])
        for ref in r['files'].values(): validate_file_ref(ref)
    validate_resources(spec['resources'], spec['reference'], anchor)
