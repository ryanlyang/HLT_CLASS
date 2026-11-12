"""Authenticated inputs, complete epoch states, native deployment and saved results."""
from pathlib import Path
import faulthandler
import gc
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities,recovery
from hlt_classification.luka_fullstat_readout.worker import reference_arrays
from hlt_classification.luka_fullstat_multiview import views
from hlt_classification.luka_fullstat_multiview.resources import check_memory
from hlt_classification.luka_fullstat_fine_tail.worker import rejection
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog,memory_snapshot
from . import campaign,submission,banks,model,training,startup,targets,kd_banks
from .execution import execution
from .sampling import CPUWorkers
from .contracts import artifact,validate,recipe,TASK

FIT_FILES=('selected.pt','selected_full.pt','selected_training_state.pt','completed_training.json',
           'training_report.json','validation_predictions.npz','startup.json','startup_measurements.json')


def parents(spec):
    ladder=spec['reuse']['ladder']
    return dict(spec=spec['content_hash'],prepared=spec['anchor']['evidence']['prepared'],
                subclasses=load_json(Path(spec['root'])/'subclasses/manifest.json')['content_hash'],
                **{n:spec['reuse']['banks'][n]['manifest_hash'] for n in ('OFFLINE','CONCAT','FUSION')},
                **{n:load_json(ladder['banks'][n]['embedding']['path'])['content_hash'] for n in ('UE100','DE066','DE033')},
                **{'kd_'+n:load_json(ladder['banks'][n]['kd']['path'])['content_hash'] for n in ladder['banks']},
                **{'kd_'+n:kd_banks.manifest(spec,n,verify_payload=False)['content_hash'] for n in kd_banks.NAMES})


def output_names():
    return ['logit_targets/'+t+'/'+n for t in kd_banks.NAMES for n in kd_banks.FILES]+['subclasses/'+n for n in targets.FILES]+[TASK+'/'+name for name in FIT_FILES]+[TASK+'/epochs/%02d/%s'%(e,n)
        for e in range(1,21) for n in ('state.pt','receipt.json')]


def epoch_records(spec,*,complete=True):
    targets.check_manifest(spec)
    records=[];folder=Path(spec['root'])/TASK
    for epoch in range(1,21):
        path=folder/'epochs'/('%02d'%epoch)/'receipt.json'
        if not path.is_file():
            if complete:raise ValueError('Missing completed epoch receipt')
            if any((folder/'epochs'/('%02d'%e)/'receipt.json').exists() for e in range(epoch+1,21)):
                raise ValueError('Noncontiguous completed epochs')
            break
        rec=load_json(path);validate(rec,'EPOCH',dict(parents(spec),recipe=recipe()['content_hash']))
        training.check_row(rec['row'],epoch,spec['counts'])
        if rec['state']!=file_ref(path.parent/'state.pt'):raise ValueError('Epoch state changed')
        records.append(rec)
    return records


def publish(spec,folder,net,report,val,device):
    training.validate_report(report,spec,parents(spec))
    save_state(folder/'selected_full.pt',training.cpu_state(net.state_dict()))
    chosen=folder/'epochs'/('%02d'%report['selected_pass'])/'state.pt'
    atomic_publish_bytes(folder/'selected_training_state.pt',chosen.read_bytes())
    write_immutable_json(folder/'completed_training.json',report)
    deployed=net.deployed().to(device)
    save_state(folder/'selected.pt',training.cpu_state(deployed.state_dict()))
    restored=DelphesParticleTransformer().to(device)
    restored.load_state_dict(torch.load(folder/'selected.pt',map_location=device,weights_only=True),strict=True)
    probabilities=training.predict(restored,val,device)
    if evaluate_probabilities(val.labels,probabilities)!=report['validation']:
        raise ValueError('Native selected checkpoint replay differs')
    atomic_publish_bytes(folder/'validation_predictions.npz',deterministic_npz_bytes(dict(
        identities=val.identities,labels=val.labels,probabilities=probabilities)))
    write_immutable_json(folder/'training_report.json',report)


def validate_fit(spec):
    folder=Path(spec['root'])/TASK
    startup.check(spec,load_json(folder/'startup.json'))
    report=load_json(folder/'training_report.json');training.validate_report(report,spec,parents(spec))
    if load_json(folder/'completed_training.json')!=report:raise ValueError('Completed report changed')
    records=epoch_records(spec)
    if [r['row'] for r in records]!=report['validation_history']:raise ValueError('Epoch/report history differs')
    selected=records[report['selected_pass']-1]
    if selected['state']['sha256']!=file_ref(folder/'selected_training_state.pt')['sha256']:
        raise ValueError('Selected full state differs')
    full=torch.load(validate_file_ref(selected['state']),map_location='cpu',weights_only=True)
    if full['epoch']!=report['selected_pass'] or full['update']!=selected['row']['update']:
        raise ValueError('Selected state cursor differs')
    saved=torch.load(folder/'selected_full.pt',map_location='cpu',weights_only=True)
    if saved.keys()!=full['student'].keys() or any(not torch.equal(v,full['student'][k]) for k,v in saved.items()):
        raise ValueError('Selected model differs from epoch state')
    net=model.new_model(spec['nodes'][TASK]);net.load_state_dict(saved,strict=True)
    native=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
    expected=net.deployed().state_dict()
    if expected.keys()!=native.keys() or any(not torch.equal(v,native[k]) for k,v in expected.items()):
        raise ValueError('Deployment differs from selected student')
    arrays=load_npz_arrays(folder/'validation_predictions.npz');_,reference=reference_arrays(spec,'M0HLT')
    if (not np.array_equal(arrays['identities'],reference['identities']) or not np.array_equal(arrays['labels'],reference['labels'])
            or evaluate_probabilities(arrays['labels'],arrays['probabilities'])!=report['validation']):
        raise ValueError('Published population/metrics differ')
    return report,arrays


def completed(spec,*,accounting=True):
    root=Path(spec['root']);rec=load_json(root/TASK/'task_receipt.json')
    validate(rec,'TASK',dict(spec=spec['content_hash']))
    if (rec['task_id']!=TASK or rec['job_id']!=submission.accepted_jobs(spec,partial=True)[TASK]
            or rec['source_commit']!=spec['source_commit'] or [r['path'] for r in rec['outputs']]!=output_names()):
        raise ValueError('Task identity/output inventory differs')
    for ref in rec['outputs']:validate_file_ref(ref,root=root)
    if validate_fit(spec)[0]['content_hash']!=rec['report_hash']:raise ValueError('Wrong published report')
    if accounting:submission.accounting(spec)
    return rec


def run(spec,*,device='cuda'):
    faulthandler.enable(all_threads=True);begun=time.monotonic()
    campaign.check(spec);job=execution(spec)
    if submission.accepted_jobs(spec,partial=True)[TASK]!=job:raise PermissionError('Wrong exact job')
    folder=Path(spec['root'])/TASK;folder.mkdir(exist_ok=False)
    log=PhaseLog(folder,lambda:memory_snapshot(torch))
    with log.phase('native_TRAIN_KD_exports_or_authenticated_recovery'):
        kd_banks.prepare(spec,device,begun)
    with log.phase('load_authenticated_HLT_and_six_TRAIN_banks'):
        tr,val,meta=views.load(spec,'train_multi3');targets,sampler=banks.load(spec,tr,device)
    def snapshot(epoch,row,state):
        d=folder/'epochs'/('%02d'%epoch);d.mkdir(parents=True,exist_ok=False)
        save_state(d/'state.pt',state)
        write_immutable_json(d/'receipt.json',artifact('EPOCH',parents=dict(parents(spec),recipe=recipe()['content_hash']),
            row=row,state=file_ref(d/'state.pt')))
        campaign.storage(spec)
    with CPUWorkers(sampler) as cpu:
        with log.phase('integrated_parity_stress_and_complete_update_timing'):
            proof=startup.run(spec,tr,val,targets,sampler,device,begun)
            write_immutable_json(folder/'startup.json',proof)
        gc.collect()
        if torch.device(device).type=='cuda':torch.cuda.empty_cache()
        with log.phase('twenty_scientific_epochs'):
            from .recovery import import_epochs
            resume=import_epochs(spec)
            net,report,full=training.train(tr,val,banks=targets,sampler=sampler,node=spec['nodes'][TASK],
                parents=parents(spec),device=device,save_epoch=snapshot,resume=resume)
            del resume
        total=memory_snapshot(torch);total['peak_rss_bytes']+=cpu.memory()
        check_memory(spec,total)
    del targets,sampler,full;gc.collect()
    with log.phase('selected_native_HLT_publication'):
        publish(spec,folder,net,report,val,device);validate_fit(spec)
    check_memory(spec,memory_snapshot(torch));campaign.check(spec)
    if execution(spec)!=job:raise PermissionError('Allocation changed')
    rec=artifact('TASK',parents=dict(spec=spec['content_hash']),task_id=TASK,job_id=job,
        source_commit=spec['source_commit'],report_hash=report['content_hash'],
        outputs=[file_ref(Path(spec['root'])/p,root=spec['root']) for p in output_names()])
    write_immutable_json(folder/'task_receipt.json',rec);completed(spec,accounting=False)
    return rec


def results(spec):
    campaign.check(spec);completed(spec)
    items=[(n,*reference_arrays(spec,n)) for n in ('M0HLT','OFFLINE','U100_W5','MULTI3','MULTI3_MIX','SUBCLASS_C03','SIX_TEACHER')]
    items.append(('SIX_TEACHER_S0',*validate_fit(spec)))
    rows=[];reference=items[0][2]
    for name,report,a in items:
        if not np.array_equal(a['identities'],reference['identities']) or not np.array_equal(a['labels'],reference['labels']):
            raise ValueError('Different validation populations')
        rows.append(dict(model=name,epoch=report['selected_pass'],validation=report['validation'],
            rejection_30=rejection(a['labels'],a['probabilities'],.3),rejection_50=rejection(a['labels'],a['probabilities'],.5)))
    lo,hi=rows[:2]
    for row in rows:
        row['recovery']=recovery(row['validation'],lo['validation'],hi['validation'])
        row['rejection_recovery']={k:{c:None if v is None or lo[k][c] is None or hi[k][c] is None or hi[k][c]==lo[k][c]
            else 100*(v-lo[k][c])/(hi[k][c]-lo[k][c]) for c,v in row[k].items()} for k in ('rejection_30','rejection_50')}
    return artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=rows,historical_references=True,single_seed=True)
