"""Production worker: verify, measure, fit, publish selected native teacher."""
from contextlib import nullcontext
from pathlib import Path
import time
import numpy as np
import torch
from hlt_classification.operations import fullstat_pt_metadata as meta
from hlt_classification.data.cache_contracts import deterministic_npz_bytes
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog,memory_snapshot
from hlt_classification.luka_u100_multiview.training import predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from . import contracts as c,campaign,source,population,views,banks,training,contrastive,artifacts,startup,execution,reporting


def fit_parents(spec,stage,dependencies):
    return dict(spec=spec['content_hash'],population=spec['population']['content_hash'],
        **{'teacher_'+t:r['receipt']['content_hash'] for t,r in dependencies.items()})


def run(spec,task,*,device='cuda:0'):
    campaign.check(spec)
    if task not in spec['tasks']:raise PermissionError('Unregistered task')
    stage=task.removeprefix('train_');job=execution.validate(spec,task);started=time.monotonic()
    folder=Path(spec['root'])/task;folder.mkdir(exist_ok=False)
    phases=PhaseLog(folder,lambda:memory_snapshot(torch));device=torch.device(device)
    try:
        with phases.phase('population_and_dependencies'):
            metadata=source.replay(spec['inputs'])
            pop,values,sampler=population.load(spec['population_root'],input_hash=spec['inputs']['content_hash'],
                prepared_hash=spec['inputs']['evidence']['prepared'])
            if pop!=spec['population']:raise ValueError('Population registration changed')
            dependencies={t:artifacts.completed(spec,t) for t in c.TEACHERS[stage]}
            parents=fit_parents(spec,stage,dependencies)
            meta.write_immutable_json(folder/'fit_binding.json',c.artifact('FIT_BINDING',
                parents=dict(spec=spec['content_hash']),stage=stage,fit_parents=parents))
        with phases.phase('views'):
            tr,val,proof=views.load(spec,stage,metadata,values)
            meta.write_immutable_json(folder/'view_build.json',proof)
        with phases.phase('fresh_teacher_banks'):
            targets={t:banks.Bank(r['folder']/'bank',tr,t,r['refs'],r['spec']) for t,r in dependencies.items()}
            matching=contrastive.Sampler(targets['OFFLINE'].keys(),tr.identities,tr.labels,device=device) if stage=='DE000' else None
            training.join(tr,val,targets,sampler,stage)
            continuation=spec['continuations'].get(stage);resume=None
            if continuation:
                resume=artifacts.restore(continuation['epochs'],continuation['epoch_parents'],stage,len(val),sampler)
        with phases.phase('integrated_acceptance'):
            accepted=startup.run(spec,stage,tr,val,targets,sampler,matching,device,folder,started,
                remaining_epochs=c.EPOCHS-(len(resume['history']) if resume else 0))
            meta.write_immutable_json(folder/'startup.json',accepted)
        inherited=list(continuation['epochs']) if continuation else []
        new_epochs=[]
        def save_epoch(epoch,row,state):
            artifacts.save_epoch(folder,epoch,row,state,parents)
            new_epochs.append(meta.file_ref(folder/'epochs'/f'{epoch:02d}'/'receipt.json'))
        with phases.phase('fit_or_finalize'):
            context=contrastive.CPUWorkers(matching,workers=spec['candidate_workers']) if matching else nullcontext()
            with context:
                net,report,best=training.train(tr,val,banks=targets,sampler=sampler,stage=stage,parents=parents,
                    device=device,save_epoch=save_epoch,matching=matching,resume=resume)
            training.validate_report(report,stage,parents,len(val),sampler)
            meta.write_immutable_json(folder/'training_report.json',report)
            native=net.deployed().to(device).eval();del net,best,resume
            artifacts.save_state(folder/'selected.pt',{k:v.cpu() for k,v in native.state_dict().items()})
        with phases.phase('selected_validation'):
            # Reload exported native checkpoint; a complete selected probability payload is required.
            native.load_state_dict(torch.load(folder/'selected.pt',map_location=device,weights_only=True),strict=True)
            probabilities=predict(native,val,device)
            measured=evaluate_probabilities(val.labels,probabilities)
            if measured!=report['validation']:raise ValueError('Selected native evaluation differs from checkpoint selection')
            payload=dict(identities=val.identities,labels=val.labels,probabilities=probabilities,pt=values['validation_pt'])
            meta.atomic_publish_bytes(folder/'validation_predictions.npz',deterministic_npz_bytes(payload))
            metrics=reporting.metrics(val.labels,probabilities,values['validation_pt'],dict(report=report['content_hash']))
            meta.write_immutable_json(folder/'validation_metrics.json',metrics)
        refs={k:meta.file_ref(folder/name) for k,name in dict(checkpoint='selected.pt',report='training_report.json').items()}
        if stage!='DE000':
            with phases.phase('selected_train_export'):
                banks.export(folder/'bank',native,tr,stage,refs,spec,device)
                banks.Bank(folder/'bank',tr,stage,refs,spec)
        with phases.phase('publication'):
            if source.current(spec['project_dir'],spec['source_commit'])!=spec['source']:raise ValueError('Source changed during task')
            files=dict(refs,predictions=meta.file_ref(folder/'validation_predictions.npz'),
                metrics=meta.file_ref(folder/'validation_metrics.json'),startup=meta.file_ref(folder/'startup.json'),
                view=meta.file_ref(folder/'view_build.json'),binding=meta.file_ref(folder/'fit_binding.json'))
            if stage!='DE000':files['bank']=meta.file_ref(folder/'bank/manifest.json')
            receipt=c.artifact('TASK_RECEIPT',parents=dict(spec=spec['content_hash'],report=report['content_hash']),
                population=pop['content_hash'],stage=stage,job_id=job,files=files,fit_parents=parents,
                inherited_epochs=inherited,new_epochs=new_epochs,selected_pass=report['selected_pass'])
            meta.write_immutable_json(folder/'task_receipt.json',receipt)
            print(f'BALANCED PUBLISHED stage={stage} epoch={report["selected_pass"]} AUC={measured["macro_ovr_auc"]:.8f}',flush=True)
        return receipt
    except BaseException as exc:
        meta.write_immutable_json(folder/'failure.json',c.artifact('FAILURE',parents=dict(spec=spec['content_hash']),
            stage=stage,job_id=job,error_type=type(exc).__name__,message=str(exc)))
        raise
