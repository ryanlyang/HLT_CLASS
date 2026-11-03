"""Fresh-process export of a finished teacher, then the unchanged HLT student."""
from pathlib import Path
import faulthandler
import gc
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.luka_fullstat_readout.worker import reference_arrays
from hlt_classification.luka_fullstat_retained.worker import sync
from hlt_classification.luka_fullstat_u100_embedding import banks as original_banks
from hlt_classification.luka_fullstat_fine_tail.worker import rejection
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from hlt_classification.luka_fullstat_multiview import (
    worker as original, training, model, targets, views, startup, resources, contracts as scientific)
from hlt_classification.luka_u100_multiview import teachers
from . import campaign, submission
from .execution import execution
from .contracts import artifact, validate, TASKS


def output_names(task):
    if task == 'export_FUSION':
        return [task+'/'+n for n in ('export_check.json','bank/manifest.json','bank/targets.npz')]
    if task == 'train_multi3': return original.output_names(task)
    raise PermissionError('Unregistered recovery task')


def fusion_bank(spec):
    return Path(spec['root'])/'export_FUSION/bank'


def fit_parents(spec):
    old = campaign.parent(spec)
    p = dict(spec=spec['content_hash'],prepared=spec['anchor']['evidence']['prepared'],
             U100=spec['reuse']['u100']['bank_hash'])
    for name,folder,owner in (
        ('CONCAT',original.teacher_folder(old,'CONCAT')/'bank',original.teacher_spec(old,'CONCAT')),
        ('FUSION',fusion_bank(spec),old)):
        m = load_json(folder/'manifest.json')
        scientific.validate(m,'BANK',targets.parents(owner,name,original.teacher_refs(old,name)))
        p[name] = m['content_hash']
    receipt = load_json(Path(spec['root'])/'export_FUSION/task_receipt.json')
    validate(receipt,'TASK',dict(spec=spec['content_hash']))
    p['export_receipt'] = receipt['content_hash']
    return p


def load_banks(spec, tr):
    old = campaign.parent(spec)
    completed(spec,'export_FUSION')
    original.completed(old,'teacher_CONCAT')
    path = Path(validate_file_ref(spec['bank_manifest'])).parent
    banks = {'U100':original_banks.Bank(path,tr,refs=spec['teacher_refs'])}
    if banks['U100'].manifest['content_hash'] != spec['reuse']['u100']['bank_hash']:
        raise ValueError('Original U100 bank differs')
    banks['CONCAT'] = targets.Bank(original.teacher_folder(old,'CONCAT')/'bank',tr,'CONCAT',
        original.teacher_refs(old,'CONCAT'),original.teacher_spec(old,'CONCAT'))
    banks['FUSION'] = targets.Bank(fusion_bank(spec),tr,'FUSION',original.teacher_refs(old,'FUSION'),old)
    return banks


def validate_fit(spec):
    task = 'train_multi3'; folder = Path(spec['root'])/task
    startup.check(spec,task,load_json(folder/'startup.json'))
    report = load_json(folder/'training_report.json'); parents = fit_parents(spec)
    training.validate_report(report,spec,task,parents)
    if report != load_json(folder/'completed_training.json'): raise ValueError('Completed report differs')
    for epoch in range(1,21):
        directory = folder/'epochs'/('%02d'%epoch); rec = load_json(directory/'receipt.json')
        scientific.validate(rec,'EPOCH',dict(parents,recipe=scientific.recipe()['content_hash']))
        if rec['row'] != report['validation_history'][epoch-1] or rec['state'] != file_ref(directory/'state.pt'):
            raise ValueError('Epoch receipt/state differs')
    chosen = folder/'epochs'/('%02d'%report['selected_pass'])/'state.pt'
    if file_ref(chosen)['sha256'] != file_ref(folder/'selected_training_state.pt')['sha256']:
        raise ValueError('Selected training state differs')
    full = torch.load(chosen,map_location='cpu',weights_only=True)
    saved = torch.load(folder/'selected_full.pt',map_location='cpu',weights_only=True)
    if (full['epoch'] != report['selected_pass'] or saved.keys() != full['student'].keys()
            or any(not torch.equal(v,full['student'][k]) for k,v in saved.items())):
        raise ValueError('Selected full model differs')
    net = model.new_model(spec['nodes'][task],task); net.load_state_dict(saved,strict=True)
    deployed = model.deploy(net,task)
    state = torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
    if state.keys() != deployed.state_dict().keys() or any(not torch.equal(v,deployed.state_dict()[k]) for k,v in state.items()):
        raise ValueError('Deployment differs from selected model')
    deployed.load_state_dict(state,strict=True)
    a = load_npz_arrays(folder/'validation_predictions.npz'); _,ref = reference_arrays(spec,'M0HLT')
    if (not np.array_equal(a['identities'],ref['identities']) or not np.array_equal(a['labels'],ref['labels'])
            or evaluate_probabilities(a['labels'],a['probabilities']) != report['validation']):
        raise ValueError('Published validation population/metrics differ')
    return report,a


def check_export(spec, proof):
    old = campaign.parent(spec)
    validate(proof,'EXPORT_CHECK',dict(spec=spec['content_hash'],fit=spec['reuse']['report_hash']))
    if (proof['selected_pass'] != spec['reuse']['selected_pass']
            or proof['refs'] != original.teacher_refs(old,'FUSION')
            or proof['train'] != load_json(Path(old['root'])/'teacher_FUSION/startup.json')['train']
            or proof['precision'] != 'FP32_no_autocast' or proof['batch_size'] != 128
            or proof['stress_finite'] is not True or proof['benchmark_steps'] != 16
            or not math.isfinite(proof['seconds_per_batch']) or proof['seconds_per_batch'] <= 0
            or not math.isfinite(proof['preparation_seconds']) or proof['preparation_seconds'] < 0):
        raise ValueError('Export witness differs')
    minutes = math.ceil(proof['preparation_seconds']/60+120+
        1.5*math.ceil(spec['counts']['train']/128)*proof['seconds_per_batch']/60)
    if proof['conservative_minutes'] != minutes or minutes > spec['minutes']['export_FUSION']:
        raise ValueError('Export exceeds six-hour allocation; preserve fit, no retraining')
    resources.check_memory(spec,proof['memory'])


def export_fit(spec, folder, log, device, begun):
    old = campaign.parent(spec)
    with log.phase('load_TRAIN_only_paired_views'):
        meta = views.metadata(old)
        if meta[0]['summaries'] != old['reuse']['u100']['summaries']:
            raise ValueError('Prepared summaries changed')
        tr = views.PairedCache(views.plain(old,'train','D000',meta),views.plain(old,'train','OFFLINE',meta))
        desc = views.describe(tr)
        if desc != load_json(Path(old['root'])/'teacher_FUSION/startup.json')['train']:
            raise ValueError('TRAIN inputs differ from finished teacher fit')
    refs = original.teacher_refs(old,'FUSION')
    with log.phase('load_original_selected_FUSION_weights'):
        net = model.new_model(old['nodes']['teacher_FUSION'],'teacher_FUSION').to(device)
        net.load_state_dict(torch.load(validate_file_ref(refs['checkpoint']),map_location=device,weights_only=True),strict=True)
        net.eval().requires_grad_(False)
    with log.phase('fresh_process_FP32_export_stress_and_timing'):
        with torch.inference_mode():
            for ids in (np.arange(min(128,len(tr))),startup.stress_ids(tr)):
                logits,h = teachers.forward(net,tr.batch(ids),device,bf16=False,name='FUSION',capture=True)
                if not torch.isfinite(logits).all() or not torch.isfinite(h).all():
                    raise ValueError('Nonfinite FP32 export stress')
                h.float().cpu().numpy()
            del logits,h
            rng = np.random.default_rng(73012); sync(device); tick = time.monotonic()
            for _ in range(16):
                logits,h = teachers.forward(net,tr.batch(rng.integers(0,len(tr),128)),device,
                    bf16=False,name='FUSION',capture=True)
                if not torch.isfinite(logits).all() or not torch.isfinite(h).all():
                    raise ValueError('Nonfinite FP32 export benchmark')
                h.float().cpu().numpy()
            sync(device); step = (time.monotonic()-tick)/16
            del logits,h
        prep = time.monotonic()-begun
        proof = artifact('EXPORT_CHECK',parents=dict(spec=spec['content_hash'],fit=spec['reuse']['report_hash']),
            selected_pass=spec['reuse']['selected_pass'],refs=refs,train=desc,
            precision='FP32_no_autocast',batch_size=128,stress_finite=True,benchmark_steps=16,
            seconds_per_batch=step,preparation_seconds=prep,memory=memory_snapshot(torch),
            conservative_minutes=math.ceil(prep/60+120+1.5*math.ceil(len(tr)/128)*step/60))
        # Persist diagnostics before enforcing allocation/memory guards.
        write_immutable_json(folder/'export_check.json',proof); check_export(spec,proof)
        print(f'FUSION export selected epoch={proof["selected_pass"]}; conservative_minutes={proof["conservative_minutes"]}',flush=True)
    with log.phase('export_original_selected_TRAIN_embeddings'):
        targets.export(folder/'bank',net,tr,'FUSION',refs,old,device)
        targets.Bank(folder/'bank',tr,'FUSION',refs,old)
    return spec['reuse']['report_hash']


def fit_student(spec, folder, log, device, begun):
    task = 'train_multi3'
    with log.phase('load_authenticated_HLT_views_and_three_banks'):
        tr,val,meta = views.load(spec,task); banks = load_banks(spec,tr)
    with log.phase('integrated_student_checks'):
        proof = startup.run(spec,task,tr,val,banks,meta,device,begun)
        startup.check(spec,task,proof); write_immutable_json(folder/'startup.json',proof)
    gc.collect()
    if torch.device(device).type == 'cuda': torch.cuda.empty_cache()
    parents = fit_parents(spec)
    def snapshot(epoch,row,state):
        target = folder/'epochs'/('%02d'%epoch); target.mkdir(parents=True,exist_ok=False)
        save_state(target/'state.pt',state)
        write_immutable_json(target/'receipt.json',scientific.artifact('EPOCH',
            parents=dict(parents,recipe=scientific.recipe()['content_hash']),row=row,state=file_ref(target/'state.pt')))
        from hlt_classification.luka_fullstat_multiview.campaign import storage
        storage(spec)
    with log.phase('twenty_scientific_student_epochs'):
        net,report,full = training.train(tr,val,banks=banks,node=spec['nodes'][task],task=task,
            parents=parents,device=device,save_epoch=snapshot)
    save_state(folder/'selected_full.pt',training.cpu_state(net.state_dict()))
    selected = folder/'epochs'/('%02d'%report['selected_pass'])/'state.pt'
    atomic_publish_bytes(folder/'selected_training_state.pt',selected.read_bytes())
    write_immutable_json(folder/'completed_training.json',report)
    training.validate_report(report,spec,task,parents)
    del banks,full
    deployed = model.deploy(net,task).to(device)
    save_state(folder/'selected.pt',training.cpu_state(deployed.state_dict()))
    restored = DelphesParticleTransformer().to(device)
    restored.load_state_dict(torch.load(folder/'selected.pt',map_location=device,weights_only=True),strict=True)
    del net,deployed; gc.collect()
    with log.phase('selected_HLT_deployment_readback'):
        p = training.predict(restored,val,device,name=None)
        if evaluate_probabilities(val.labels,p) != report['validation']:
            raise ValueError('Selected metric replay differs')
        atomic_publish_bytes(folder/'validation_predictions.npz',deterministic_npz_bytes(dict(
            identities=val.identities,labels=val.labels,probabilities=p)))
        write_immutable_json(folder/'training_report.json',report); validate_fit(spec)
    return report['content_hash']


def completed(spec, task, *, accounting=True):
    if task not in TASKS: raise PermissionError('Unregistered recovery task')
    root = Path(spec['root']); rec = load_json(root/task/'task_receipt.json')
    validate(rec,'TASK',dict(spec=spec['content_hash']))
    if (rec['task_id'] != task or rec['source_commit'] != spec['source_commit']
            or rec['job_id'] != submission.accepted_jobs(spec,partial=True).get(task)
            or [r['path'] for r in rec['outputs']] != output_names(task)):
        raise ValueError('Task identity/output inventory differs')
    for ref in rec['outputs']: validate_file_ref(ref,root=root)
    if task == 'export_FUSION':
        old = campaign.parent(spec); proof = load_json(root/task/'export_check.json'); check_export(spec,proof)
        m = load_json(fusion_bank(spec)/'manifest.json')
        scientific.validate(m,'BANK',targets.parents(old,'FUSION',original.teacher_refs(old,'FUSION')))
        if m['teacher'] != 'FUSION' or m['role'] != 'train' or m['rows'] != spec['counts']['train']:
            raise ValueError('Wrong exported population')
        validate_file_ref(m['payload'],root=fusion_bank(spec))
        report_hash = spec['reuse']['report_hash']
    else:
        completed(spec,'export_FUSION',accounting=accounting)
        report_hash = validate_fit(spec)[0]['content_hash']
    if rec['report_hash'] != report_hash: raise ValueError('Task report differs')
    if accounting: submission.successful(spec,task,rec['job_id'])
    return rec


def run(spec, task, *, device='cuda'):
    if task not in TASKS: raise PermissionError('Recovery cannot retrain either teacher')
    faulthandler.enable(all_threads=True)
    begun = time.monotonic(); campaign.check(spec); job = execution(spec,task)
    if submission.accepted_jobs(spec,partial=True).get(task) != job:
        raise PermissionError('Wrong exact submitted recovery job')
    folder = Path(spec['root'])/task; folder.mkdir(exist_ok=False)
    log = PhaseLog(folder,lambda:memory_snapshot(torch))
    report_hash = (export_fit if task == 'export_FUSION' else fit_student)(spec,folder,log,device,begun)
    resources.check_memory(spec,memory_snapshot(torch)); campaign.check(spec)
    if execution(spec,task) != job: raise PermissionError('Allocation changed')
    rec = artifact('TASK',parents=dict(spec=spec['content_hash']),task_id=task,job_id=job,
        source_commit=spec['source_commit'],report_hash=report_hash,
        outputs=[file_ref(Path(spec['root'])/p,root=spec['root']) for p in output_names(task)])
    write_immutable_json(folder/'task_receipt.json',rec)
    completed(spec,task,accounting=False)
    return rec


def results(spec):
    campaign.check(spec); old = campaign.parent(spec); rows = []
    items = [(n,*reference_arrays(spec,n)) for n in ('M0HLT','OFFLINE','U100_W5')]
    for name in ('CONCAT','FUSION'):
        items.append(('teacher_'+name,*original.validate_fit(old,'teacher_'+name)))
    pending = []
    if (Path(spec['root'])/'train_multi3/task_receipt.json').is_file():
        completed(spec,'train_multi3'); items.append(('train_multi3',*validate_fit(spec)))
    else: pending.append('train_multi3')
    ref = items[0][2]
    for name,report,a in items:
        if not np.array_equal(a['identities'],ref['identities']) or not np.array_equal(a['labels'],ref['labels']):
            raise ValueError('Comparison populations differ')
        rows.append(dict(model=name,selected_epoch=report['selected_pass'],validation=report['validation'],
            privileged=name == 'OFFLINE' or name.startswith('teacher_'),
            rejection_30=rejection(a['labels'],a['probabilities'],.3),
            rejection_50=rejection(a['labels'],a['probabilities'],.5)))
    lo,hi = rows[:2]
    for row in rows:
        row['recovery'] = recovery(row['validation'],lo['validation'],hi['validation'])
        row['rejection_recovery'] = {k:{c:None if v is None or lo[k][c] is None or hi[k][c] is None or hi[k][c] == lo[k][c]
            else 100*(v-lo[k][c])/(hi[k][c]-lo[k][c]) for c,v in row[k].items()} for k in ('rejection_30','rejection_50')}
    return artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=rows,pending=pending,
        historical_U100_reference=True,single_seed=True,adaptive_validation=True)
