"""Authenticate surviving controls and archived recipes, not deleted weights."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_fullsim import regular_cache
from hlt_classification.luka_attention import contracts as attention
from hlt_classification.luka_embedding_aux.contracts import validate as validate_anchor
from hlt_classification.luka_multiteacher_trial import source as donor
from hlt_classification.luka_multiteacher_trial.contracts import artifact as teacher_artifact, validate as teacher_validate
from .contracts import artifact, validate, REPLACEMENTS

BASE = '01d45d6057e799b4e262f8c590b51706803ebfd1'
MODULES = ('__init__', 'contracts', 'source', 'campaign', 'submission', 'worker')
ADDITIONS = {'src/hlt_classification/luka_multiteacher_recovery/'+n+'.py' for n in MODULES}
REQUIRED = ADDITIONS | {'scripts/luka_multiteacher_recovery.py', 'scripts/queue_luka_multiteacher_recovery.sh',
    'sbatch/run_luka_multiteacher_recovery.sh', 'tests/test_luka_multiteacher_recovery.py',
    'docs/plans/LUKA_MULTITEACHER_RECOVERY_PLAN.md', 'docs/contracts/LUKA_MULTITEACHER_RECOVERY.md',
    'docs/LUKA_MULTITEACHER_RECOVERY_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True,
                              capture_output=True, text=True).stdout.splitlines()
    missing = REQUIRED - set(git('ls-files'))
    if missing:
        raise ValueError('Commit complete recovery implementation: '+', '.join(sorted(missing)))
    bad = [r for r in git('diff', '--name-status', '--no-renames', BASE, commit, '--', 'src')
           if not r.startswith('A\t') or r[2:] not in ADDITIONS]
    if bad:
        raise ValueError('Existing scientific runtime changed:\n'+'\n'.join(bad))
    if Path(__file__).resolve() != project/'src/hlt_classification/luka_multiteacher_recovery/source.py':
        raise ValueError('Imported implementation escapes pinned worktree')
    return snapshot


ARCHIVE_CHECK = '''
import json,sys,runpy
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
a=runpy.run_path(sys.argv[2])['read_archive'](sys.argv[1])
s=load_json(sys.argv[1]); g=load_json(s['gate_path'])
p=load_json(Path(g['root'])/'preflight/receipt.json')
nodes=[n for n in g['nodes'] if n['node_id'] in ('LOGIT_D066','LOGIT_D033')]
recorded={}
for n in nodes:
    task='train_'+n['node_id']; r=load_json(Path(s['root'])/task/'receipt.json')
    refs=[x for x in r['outputs'] if x['path']==task+'/selected.pt']
    if len(refs)!=1: raise ValueError('Ambiguous archived checkpoint descriptor')
    recorded[n['node_id']]=refs[0]
print('RECOVERY_ARCHIVE='+json.dumps(dict(anchor=a['anchor'],prepared=a['prepared'],
    archive=a['archive'],nodes=nodes,recorded_checkpoints=recorded,
    prior_request=dict(train_minutes=p['train_minutes'],export_minutes=p['export_minutes'],
                       environment=p['environment'],gpu=p['gpu'])),sort_keys=True,allow_nan=False))
'''


def archived(path, anchor, parent):
    path = Path(path).resolve(strict=True)
    spec = load_json(path)
    if spec['source_commit'] != '94f560259dd04253c0aa47b31dec3aedc105ac14':
        raise ValueError('Expected original attention-comparison archive')
    gate = load_json(spec['gate_path'])
    validate_source_snapshot(gate['source_snapshot'], repository=spec['project_dir'], require_clean=True)
    helper = Path(__file__).parents[1]/'luka_attention_weight/reference_archive.py'
    ref, data = donor.replay(path, spec['project_dir'], ARCHIVE_CHECK, 'RECOVERY_ARCHIVE=', extra=(helper,))
    expected = [n for n in attention.nodes(parent['nodes']) if n['node_id'] in REPLACEMENTS]
    if (data['nodes'] != expected or data['prepared'] != anchor['evidence']['prepared']
            or data['anchor']['spec'] != anchor['spec']
            or any(data['anchor']['evidence']['models'][n][k] != anchor['evidence']['models'][n][k]
                   for n in ('M0HLT','OFFLINE') for k in ('checkpoint','report','receipt','diagnostics'))):
        raise ValueError('Archived teacher recipe, population or controls differ')
    return artifact('ARCHIVE', parents=dict(anchor=anchor['content_hash']), spec=ref, **data)


def survivors(anchor):
    u100 = donor.u100_teacher(anchor)
    parent = load_json(anchor['spec']['path'])
    output = {}
    for name, co in (('OFFLINE', 'OFFLINE'), ('COARSE_U100', 'U100')):
        files = anchor['evidence']['models']['OFFLINE'] if name == 'OFFLINE' else u100['files']
        node = next(n for n in parent['nodes'] if n['node_id'] == name)
        output[name] = teacher_artifact('TEACHER', parents=dict(anchor=anchor['content_hash']),
            model_id=name, coordinate=co, node=node, prepared=anchor['evidence']['prepared'], role='train',
            spec=anchor['spec'], files={k:files[k] for k in ('checkpoint','report','predictions','receipt')},
            selected_native_checkpoint_revalidated=True, origin='original_fixed30')
    return output


def imports(spec):
    anchor = spec['anchor']
    validate_anchor(anchor, 'ANCHOR')
    parent = load_json(validate_file_ref(anchor['spec']))
    donor.validate_reference(spec['reference'], 'REFERENCE')
    old, expected = dict(spec['reference']['node']), donor.donor_node(parent)
    for n in (old, expected):
        n.pop('node_id'); n.pop('branch')
    if old != expected or spec['reference']['parents']['anchor'] != anchor['content_hash']:
        raise ValueError('Comparison reference recipe differs')
    validate_file_ref(spec['reference']['spec'])
    for ref in spec['reference']['files'].values():
        validate_file_ref(ref)
    for row in anchor['evidence']['models'].values():
        for ref in row.values(): validate_file_ref(ref)
    if set(spec['survivors']) != {'OFFLINE', 'COARSE_U100'}:
        raise ValueError('Wrong surviving teacher set')
    for name, t in spec['survivors'].items():
        teacher_validate(t, 'TEACHER', dict(anchor=anchor['content_hash']))
        if (t['node'] != next(n for n in parent['nodes'] if n['node_id'] == name)
                or t['model_id'] != name or t['coordinate'] != t['node']['coordinate']
                or t['spec'] != anchor['spec'] or t['prepared'] != anchor['evidence']['prepared']
                or t['role'] != 'train' or t['origin'] != 'original_fixed30'
                or t['selected_native_checkpoint_revalidated'] is not True
                or set(t['files']) != {'checkpoint','report','predictions','receipt'}):
            raise ValueError('Surviving teacher binding differs')
        for ref in t['files'].values(): validate_file_ref(ref)
    a = spec['archive']; validate(a, 'ARCHIVE', dict(anchor=anchor['content_hash']))
    validate_file_ref(a['spec'])
    for ref in a['archive']['verified_files']: validate_file_ref(ref)
    if (a['prepared'] != anchor['evidence']['prepared'] or a['anchor']['spec'] != anchor['spec']
            or a['nodes'] != [n for n in attention.nodes(parent['nodes']) if n['node_id'] in REPLACEMENTS]
            or a['prior_request']['environment'] != spec['resources']['environment']
            or a['prior_request']['gpu'] != spec['resources']['gpu']):
        raise ValueError('Recovery recipe or historical environment changed')
    donor.validate_resources(spec['resources'], spec['reference'], anchor)


def inputs(spec):
    """Authenticate every reused block needed by new teachers AND students."""
    root = Path(spec['anchor']['evidence']['prepared_root'])
    cache, prepared, inventory, splits = regular_cache.metadata(root)
    if (cache['content_hash'] != spec['anchor']['evidence']['cache']
            or prepared['content_hash'] != spec['anchor']['evidence']['prepared']):
        raise ValueError('Prepared input metadata changed')
    for role, coordinates in (('train', ('D000','U100','D066','D033')),
                              ('validation', ('D000','D066','D033'))):
        for co in coordinates:
            for ref in cache['blocks'][role][co]:
                try:
                    validate_file_ref(ref, root=root)
                except (OSError, ValueError) as exc:
                    raise ValueError(f'Required {role}/{co} block unavailable or corrupt: {ref["path"]}') from exc
    # OFFLINE TRAIN is reconstructed in RAM by the original exporter. Check
    # its raw sources now, before spending GPU time on either replacement.
    raw = Path(spec['anchor']['evidence']['raw_container'])/'raw'
    for member in splits['memberships']['train']:
        ref = inventory['files'][member['file_index']]
        try:
            validate_file_ref(ref, root=raw)
        except (OSError, ValueError) as exc:
            raise ValueError('Required OFFLINE TRAIN raw source unavailable or corrupt: '+ref['path']) from exc
