"""Explicit one-job tier3 execution using prior evidence, NOT a new preflight."""
from pathlib import Path
import gc
import getpass
import math
import re
import subprocess
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_fullsim import regular_cache
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from hlt_classification.luka_fullsim.regular_submission import check_site, clean_environment
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger, assemble_submission_ledger
from .contracts import artifact, validate, recipe, node, COUNTS
from . import source, targets, worker
from .preflight import teacher_readback

AUTHORIZATION = 'AUTHORIZE_LUKA_D066_EMBEDDING_DIRECT_TIER3_V1'
POLICY = 'reuse_accepted_sweep_student_envelope_new_D066_targets_no_new_preflight_v1'
OLD_JOB = '21944417'

# Runs under the accepted sweep's ORIGINAL clean source/environment. This is
# saved-evidence authentication only: no model loading, no inference/training.
RESOURCE_CHECK = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_embedding_sweep import debug
from hlt_classification.luka_fullsim.science_campaign import preflight_accounting
s=load_json(sys.argv[1]); g,r=debug.check_spec(s,retired=True)
result=debug.results(s)
if 'w5' in result['pending']: raise ValueError('Completed matched w5 required')
w=load_json(Path(s['root'])/'w5/receipt.json')
accounting=preflight_accounting(w['job_id'])
print('D066_DIRECT_RESOURCES='+json.dumps(dict(anchor=g['anchor'],counts=g['counts'],
    node=g['nodes']['w5'],site=g['execution_site'],cpus=g['cpus'],memory_mb=g['memory_mb'],
    report=r,accounting=accounting,refs=[file_ref(s['gate_path']),
    file_ref(Path(g['root'])/'preflight/receipt.json'),file_ref(Path(s['root'])/'w5/receipt.json')]),
    sort_keys=True,allow_nan=False))
'''


def resources(reference, anchor, student):
    path = reference['spec']['path']; saved = load_json(path)
    if saved['source_commit'] != source.REFERENCE_COMMIT:
        raise ValueError('Expected original matched OFFLINE-w5 source')
    ref, data = source.replay(path, saved['project_dir'], RESOURCE_CHECK, 'D066_DIRECT_RESOURCES=')
    old, new = dict(data['node']), dict(student)
    for n in (old, new):
        n.pop('node_id'); n.pop('branch')
    report = data['report']; minutes = report['train_minutes']['w5']
    if (ref != reference['spec'] or data['anchor'] != anchor or data['counts'] != COUNTS
            or old != new or data['site'] != execution_site('sporc_a100')
            or (data['cpus'], data['memory_mb']) != (6, 90000)
            or report['environment'] != anchor['evidence']['environment']
            or report['technical_checks_passed'] is not True
            or report['checkpoint_probability_roundtrip'] is not True
            or data['accounting']['state'] != 'COMPLETED' or data['accounting']['exit_code'] != '0:0'
            or type(minutes) is not int or not 1 <= minutes <= 1320):
        raise ValueError('Inherited resource evidence, population or recipe differs')
    for item in data['refs']: validate_file_ref(item)
    result = artifact('DIRECT_RESOURCES', parents=dict(reference=reference['content_hash']),
        policy=POLICY, donor_preflight_hash=report['content_hash'], environment=report['environment'],
        gpu=report['gpu'], measured_w5_minutes=minutes, setup_allowance_minutes=120,
        train_minutes=max(360, minutes+120), donor_refs=data['refs'], w5_accounting=data['accounting'],
        new_D066_preflight_performed=False)
    validate_resources(result, reference, anchor)
    return result


def validate_resources(value, reference, anchor):
    validate(value, 'DIRECT_RESOURCES', dict(reference=reference['content_hash']))
    minutes = value['measured_w5_minutes']
    if (type(minutes) is not int or not 1 <= minutes <= 1320
            or value['policy'] != POLICY or value['setup_allowance_minutes'] != 120
            or value['train_minutes'] != max(360, minutes+120)
            or value['new_D066_preflight_performed'] is not False
            or value['environment'] != anchor['evidence']['environment']
            or len(value['donor_refs']) != 3
            or value['w5_accounting']['state'] != 'COMPLETED'
            or value['w5_accounting']['exit_code'] != '0:0'):
        raise ValueError('Direct inherited resource policy differs')
    for ref in value['donor_refs']: validate_file_ref(ref)
    gate, report, fit = [load_json(ref['path']) for ref in value['donor_refs']]
    if (value['donor_preflight_hash'] != report['content_hash']
            or value['measured_w5_minutes'] != report['train_minutes']['w5']
            or value['environment'] != report['environment'] or value['gpu'] != report['gpu']
            or gate['execution_site'] != execution_site('sporc_a100')
            or gate['anchor'] != anchor or gate['counts'] != COUNTS
            or report['technical_checks_passed'] is not True
            or report['checkpoint_probability_roundtrip'] is not True
            or fit['job_id'] != value['w5_accounting']['job_id']
            or value['donor_refs'][2] != reference['files']['receipt']):
        raise ValueError('Inherited report binding differs')


def cancelled_original():
    """No scheduler mutation; fail if the old gate could launch a duplicate."""
    proc = subprocess.run(['sacct', '-X', '-n', '-P', '-j', OLD_JOB, '-o',
        'JobIDRaw,JobName%100,User%100,Account%100,Partition%100,State%40,ElapsedRaw'],
        check=True, capture_output=True, text=True)
    rows = [r.strip().rstrip('|').split('|') for r in proc.stdout.splitlines() if r.strip()]
    if (getpass.getuser() != 'ryreu' or len(rows) != 1 or len(rows[0]) != 7
            or rows[0][:5] != [OLD_JOB, 'lukad6e_preflight', 'ryreu', 'reu-aisocial', 'debug']
            or not re.fullmatch(r'CANCELLED(?: by [0-9]+)?', rows[0][5]) or rows[0][6] != '0'):
        raise ValueError('First retire ONLY pending debug21944417; require exact CANCELLED/zero elapsed accounting')
    return dict(job_id=OLD_JOB, state=rows[0][5], elapsed_seconds=0)


def registration(parent, inherited):
    return dict(counts=COUNTS, original50k_rows=50000, node=node(parent), recipe=recipe(),
        execution_site=execution_site('sporc_a100'), cpus=6, memory_mb=90000,
        train_minutes=inherited['train_minutes'], storage_budget_bytes=2*2**30,
        execution_policy=POLICY, preflight_jobs=0, scientific_fits=1,
        technical_training_epochs=0, automatic_retry=False, replaces_job=OLD_JOB)


def create(root, parent_path, teacher_path, reference_path, project, commit, *, available_quota_gib):
    cancelled_original()
    root, project = Path(root).resolve(), Path(project).resolve()
    snapshot = source.current(project, commit)
    anchor, parent = source.authenticate(parent_path)
    teacher = source.teacher(teacher_path, anchor)
    reference = source.reference(reference_path, anchor, node(parent))
    inherited = resources(reference, anchor, node(parent))
    protected = [project, Path(parent['project_dir']), Path(parent['root']).parent,
        Path(anchor['evidence']['prepared_root']), Path(anchor['evidence']['raw_container'])]
    protected += [Path(p).resolve().parent.parent for p in (teacher_path, reference_path)]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('Fresh direct root overlaps shared source/data/campaign')
    if type(available_quota_gib) is not int or available_quota_gib < 2:
        raise ValueError('Reserve2GiB personally available persistent storage')
    spec = artifact('DIRECT_SPEC', parents=dict(source=snapshot['content_hash'], anchor=anchor['content_hash'],
        teacher=teacher['content_hash'], reference=reference['content_hash'], resources=inherited['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=snapshot,
        anchor=anchor, teacher=teacher, reference=reference, resources=inherited,
        available_quota_gib=available_quota_gib, **registration(parent, inherited))
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def validate_spec(spec):
    validate(spec, 'DIRECT_SPEC', {k:spec[k if k!='source' else 'source_snapshot']['content_hash']
                                 for k in ('source','anchor','teacher','reference','resources')})
    validate_resources(spec['resources'], spec['reference'], spec['anchor'])
    parent = dict(nodes=[dict(spec['node'], node_id='M0HLT', branch='BASELINE')])
    parent['nodes'][0].pop('auxiliary_weight')
    if (any(spec.get(k) != v for k,v in registration(parent, spec['resources']).items())
            or type(spec['available_quota_gib']) is not int or spec['available_quota_gib'] < 2
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Direct one-job scientific/execution registration differs')


def check_spec(spec):
    validate_spec(spec); cancelled_original()
    if source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Direct source changed')
    anchor, parent = source.authenticate(spec['anchor']['spec']['path'])
    if (anchor != spec['anchor'] or source.teacher(spec['teacher']['spec']['path'], anchor) != spec['teacher']
            or source.reference(spec['reference']['spec']['path'], anchor, node(parent)) != spec['reference']
            or resources(spec['reference'], anchor, node(parent)) != spec['resources']):
        raise ValueError('Direct imported evidence changed')


def render(spec):
    validate_spec(spec)
    root = Path(spec['root']); site = spec['execution_site']
    command = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
        '--partition='+site['partition'], '--account='+site['account'], '--qos='+site['qos'],
        '--cpus-per-task=6', '--mem=90000M', '--time='+_walltime(spec['train_minutes']), '--gres='+site['gres'],
        '--job-name=lukad6e_direct', '--comment=lukad6e_direct:'+spec['content_hash'],
        '--chdir='+spec['project_dir'], '--output='+str(root/'direct-%j.out'),
        spec['project_dir']+'/sbatch/run_luka_d066_embedding.sh', spec['project_dir'],
        str(root/'campaign_spec.json'), 'direct']
    return artifact('DIRECT_PLAN', parents=dict(spec=spec['content_hash']), jobs=1, preflight_jobs=0,
        mode='direct_science', authorization=AUTHORIZATION,
        commands=[dict(task_id='direct', dependencies=[], command=command)])


def submit(spec, *, execute=False, plan_hash=None, authorization=None):
    check_spec(spec); plan = render(spec)
    if execute and (plan_hash != plan['content_hash'] or authorization != AUTHORIZATION):
        raise PermissionError('Exact direct plan hash and authorization required')
    root = Path(spec['root'])/'submission'; root.mkdir(exist_ok=True)
    saved, dry, ledger = root/'command_plan.json', root/'dry_run_submission_ledger.json', root/'submission_ledger.json'
    if execute and (not saved.is_file() or load_json(saved) != plan):
        raise ValueError('Saved unchanged direct dry review required')
    write_immutable_json(saved, plan); check_site(plan)
    if execute:
        expected = build_submission_ledger(campaign_spec_sha256=spec['content_hash'],
            jobs={'direct':'1'}, commands={'direct':plan['commands'][0]['command']}, dry_run=True)
        if not dry.is_file() or load_json(dry) != expected:
            raise ValueError('Canonical direct dry ledger required')
        if not ledger.is_file():
            (root/'live_claim').mkdir(exist_ok=False)
            write_immutable_json(root/'live_claim/intent.json', artifact('DIRECT_CLAIM',
                parents=dict(spec=spec['content_hash'], plan=plan['content_hash'])))
    value = submit_exact_dag(identity=spec['content_hash'], plan=plan, output=ledger if execute else dry,
        canonical_dry_run=dry, execute=execute, environment=clean_environment())
    return value if execute else plan


def accepted_job(spec):
    root = Path(spec['root'])/'submission'
    events, jobs = load_exact_dag_journal(root/'submission_ledger_journal',
        identity=spec['content_hash'], plan=render(spec))
    if (set(jobs) != {'direct'} or load_json(root/'submission_ledger.json') !=
            assemble_submission_ledger(campaign_spec_sha256=spec['content_hash'], events=events)):
        raise ValueError('Direct exact submission journal differs')
    return jobs['direct']


def run(spec):
    check_spec(spec)
    job = worker.execution(spec, spec['resources']); worker.storage(spec)
    if job != accepted_job(spec):
        raise ValueError('Direct allocation differs from exact submitted job')
    root = Path(spec['root'])
    # Permanent exclusive claim: preserve partial targets/fits; never overwrite/retry.
    (root/'run_claim').mkdir(exist_ok=False)
    write_immutable_json(root/'run_claim/intent.json', artifact('DIRECT_RUN_CLAIM',
        parents=dict(spec=spec['content_hash']), job_id=job))
    log = PhaseLog(root, lambda: memory_snapshot(torch))
    torch.cuda.reset_peak_memory_stats()
    meta = regular_cache.metadata(spec['anchor']['evidence']['prepared_root'])
    print('DIRECT TIER3: no new preflight, stress tests or technical epoch; setup then fixed30 science.', flush=True)
    with log.phase('authenticate_D066_selected_checkpoint_predictions'):
        ref = spec['teacher']['files']['checkpoint']; validate_file_ref(ref)
        teacher = DelphesParticleTransformer().to('cuda').eval()
        teacher.load_state_dict(torch.load(ref['path'], map_location='cpu', weights_only=True), strict=True)
        val_teacher = worker.data(spec, 'validation', 'D066', meta)
        proof = teacher_readback(teacher, val_teacher, spec['teacher'], 'cuda')
        write_immutable_json(root/'teacher_readback.json', proof)
        del val_teacher
    with log.phase('export_frozen_D066_TRAIN_targets_once'):
        train_teacher = worker.data(spec, 'train', 'D066', meta)
        bank_record = targets.export(root/'targets', teacher, train_teacher,
            teacher_record=spec['teacher'], device='cuda')
        del train_teacher, teacher
        gc.collect(); torch.cuda.empty_cache()
    with log.phase('load_authenticated_HLT_and_join_TRAIN_targets'):
        train, val = (worker.data(spec, role, 'D000', meta) for role in ('train','validation'))
        bank = targets.Bank(root/'targets', train, teacher_record=spec['teacher'])
        worker.snapshot_references(spec, val, meta, folder=root/'references')
    with log.phase('fresh_fixed30_HLT_CE_plus_D066_embedding'):
        fit = worker.fit(root/'fit', spec, train, val, bank, device='cuda', technical=False)
    with log.phase('selected_comparison'):
        _, student = worker.validate_fit(root/'fit', spec, bank_record['content_hash'], technical=False)
        result = artifact('DIRECT_RESULTS', parents=dict(spec=spec['content_hash'], fit=fit['content_hash']),
            counts=spec['counts'], selection_population='entire_validation', original50k_selection=False,
            single_training_seed=True, new_preflight_performed=False,
            rows=worker.comparison(spec, student, folder=root/'references'))
        write_immutable_json(root/'results.json', result)
    worker.storage(spec)
    if source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Source changed during direct science')
    files = ['teacher_readback.json','targets/manifest.json','targets/targets.npz','results.json',
        'fit/receipt.json','fit/selected.pt','fit/training_report.json','fit/validation_predictions.npz']
    files += ['references/'+name+'.npz' for name in ('M0HLT','OFFLINE','OFFLINE_EMBED_W5','LOGIT_D066','original50k')]
    receipt = artifact('DIRECT_SCIENCE', parents=dict(spec=spec['content_hash'], resources=spec['resources']['content_hash']),
        job_id=job, new_preflight_performed=False, outputs=[file_ref(root/f, root=root) for f in files])
    write_immutable_json(root/'receipt.json', receipt)
    return result


def results(spec):
    validate_spec(spec)
    root = Path(spec['root']); receipt = load_json(root/'receipt.json')
    validate(receipt, 'DIRECT_SCIENCE', dict(spec=spec['content_hash'], resources=spec['resources']['content_hash']))
    required = {'teacher_readback.json','targets/manifest.json','targets/targets.npz','results.json',
        'fit/receipt.json','fit/selected.pt','fit/training_report.json','fit/validation_predictions.npz'}
    required |= {'references/'+n+'.npz' for n in ('M0HLT','OFFLINE','OFFLINE_EMBED_W5','LOGIT_D066','original50k')}
    if (receipt['new_preflight_performed'] is not False or not re.fullmatch('[1-9][0-9]*',receipt['job_id'])
            or len(receipt['outputs']) != len(required) or {r['path'] for r in receipt['outputs']} != required):
        raise ValueError('Direct output evidence coverage differs')
    for ref in receipt['outputs']: validate_file_ref(ref, root=root)
    bank = load_json(root/'targets/manifest.json')
    validate(bank, 'BANK', dict(recipe=recipe()['content_hash'], prepared=spec['anchor']['evidence']['prepared'],
        teacher=spec['teacher']['content_hash']))
    proof = load_json(root/'teacher_readback.json')
    validate(proof, 'TEACHER_READBACK', dict(teacher=spec['teacher']['content_hash']))
    if (bank['role'] != 'train' or bank['coordinate'] != 'D066' or bank['width'] != 128
            or bank['rows'] != spec['counts']['train'] or bank['teacher'] != spec['teacher']['files']['checkpoint']
            or bank['payload'] != file_ref(root/'targets/targets.npz', root=root/'targets')
            or proof['native_probability_parity'] is not True or proof['normalization_used'] is not False
            or proof['coordinate'] != 'D066' or proof['role'] != 'validation'
            or proof['rows'] != spec['counts']['validation']
            or (proof['rtol'],proof['atol']) != (1e-5,1e-6)
            or not math.isfinite(proof['maximum_absolute_error'])
            or not 0 <= proof['maximum_absolute_error'] <= 1.1e-5):
        raise ValueError('Direct teacher/target provenance differs')
    _, student = worker.validate_fit(root/'fit', spec, bank['content_hash'], technical=False)
    fit = load_json(root/'fit/receipt.json'); result = load_json(root/'results.json')
    validate(result, 'DIRECT_RESULTS', dict(spec=spec['content_hash'], fit=fit['content_hash']))
    if (result['counts'] != spec['counts'] or result['new_preflight_performed'] is not False
            or result['selection_population'] != 'entire_validation' or result['original50k_selection'] is not False
            or result['single_training_seed'] is not True
            or result['rows'] != worker.comparison(spec, student, folder=root/'references')):
        raise ValueError('Saved direct comparison differs')
    return result
