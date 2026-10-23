"""Shared export, five fresh scientific fits and saved-only partial reporting."""
from pathlib import Path
import gc
import math
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities,recovery
from hlt_classification.jetclass2_delphes.schema import CLASS_NAMES
from hlt_classification.luka_attention.worker import execution
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from hlt_classification.luka_fullsim import regular_cache,alljets_data
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog,memory_snapshot
from .contracts import artifact,validate,ARMS,TEACHERS
from . import campaign,banks,training,submission
from .model import new_model

FIT_FILES=('selected.pt','training_report.json','validation_predictions.npz')


def reference_files(spec):
    refs={n:spec['anchor']['evidence']['models'][n] for n in ('M0HLT','OFFLINE')}
    refs['OFFLINE_EMBED_W5']=spec['reference']['files']
    if spec['u100_reference'] is not None: refs['U100_EMBED_W5']=spec['u100_reference']['files']
    return refs


def output_names(spec,task):
    if task!='export_targets': return [task+'/'+n for n in (*FIT_FILES,'receipt.json')]
    return (['targets/'+t+'/'+n for t in TEACHERS for n in ('manifest.json','targets.npz')]
        +['targets/bundle.json','references/original50k.npz']
        +['references/'+n+'.npz' for n in reference_files(spec)])


def metadata(spec):
    meta=regular_cache.metadata(spec['anchor']['evidence']['prepared_root'])
    if meta[0]['content_hash']!=spec['anchor']['evidence']['cache'] or meta[1]['content_hash']!=spec['anchor']['evidence']['prepared']:
        raise ValueError('Prepared data changed')
    return meta


def load_native(ref,device):
    validate_file_ref(ref)
    model=DelphesParticleTransformer().to(device).eval()
    model.load_state_dict(torch.load(ref['path'],map_location='cpu',weights_only=True),strict=True)
    validate_file_ref(ref)
    return model


def export_targets(spec,device):
    root=Path(spec['root']); meta=metadata(spec)
    for name,co in TEACHERS.items():
        cache=banks.data(spec,'train',co,meta)
        teacher=load_native(spec['teachers'][name]['files']['checkpoint'],device)
        banks.export(root/'targets'/name,teacher,cache,spec['teachers'][name],device=device)
        del cache,teacher; gc.collect()
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
    tr=banks.data(spec,'train','D000',meta); bank=banks.load(spec,tr,publish=True)
    # This is an input-validity check for the registered sampler, not a benchmark.
    training.AnchorSampler(bank.labels)
    del tr,bank; gc.collect()
    val=banks.data(spec,'validation','D000',meta)
    folder=root/'references'; folder.mkdir(exist_ok=False)
    ix=alljets_data.comparison_indices(meta[2],meta[3],val.identities,val.labels)
    if len(ix)!=spec['original50k_rows']: raise ValueError('Report-only subset count differs')
    atomic_publish_bytes(folder/'original50k.npz',deterministic_npz_bytes(dict(indices=ix)))
    for name,refs in reference_files(spec).items():
        a=load_npz_arrays(validate_file_ref(refs['predictions'])); validate_predictions(a,len(val))
        report=load_json(validate_file_ref(refs['report']))
        if (not np.array_equal(a['identities'],val.identities) or not np.array_equal(a['labels'],val.labels)
                or evaluate_probabilities(a['labels'],a['probabilities'])!=report['validation']):
            raise ValueError('Reference selected predictions/population differ')
        atomic_publish_bytes(folder/(name+'.npz'),deterministic_npz_bytes(a))


def fit(directory,spec,arm,tr,val,bank,device):
    directory=Path(directory); directory.mkdir(exist_ok=False)
    model=new_model(spec['nodes'][arm])
    report,state=training.train(model,tr,val,bank=bank,node=spec['nodes'][arm],device=device)
    training.validate_report(report,spec,arm,bank.manifest['content_hash'])
    save_state(directory/'selected.pt',state)
    native=load_native(file_ref(directory/'selected.pt'),device)
    actual=predict(native,val,device=device,batch_size=128)
    expected=predict(model,val,device=device,batch_size=128)
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
    if evaluate_probabilities(val.labels,actual)!=report['validation']:
        raise ValueError('Native deployment metrics differ from selected checkpoint')
    atomic_publish_bytes(directory/'validation_predictions.npz',deterministic_npz_bytes(dict(
        identities=val.identities,labels=val.labels,probabilities=actual)))
    write_immutable_json(directory/'training_report.json',report)
    receipt=artifact('FIT',parents=dict(spec=spec['content_hash'],bank=bank.manifest['content_hash']),
        arm=arm,checkpoint_probability_roundtrip=True,outputs=[file_ref(directory/n,root=directory) for n in FIT_FILES])
    write_immutable_json(directory/'receipt.json',receipt)
    return receipt


def validate_fit(spec,arm,bank_hash):
    root=Path(spec['root'])/arm; r=load_json(root/'receipt.json')
    validate(r,'FIT',dict(spec=spec['content_hash'],bank=bank_hash))
    if r['arm']!=arm or r['checkpoint_probability_roundtrip'] is not True or [x['path'] for x in r['outputs']]!=list(FIT_FILES):
        raise ValueError('Fit publication differs')
    for ref in r['outputs']: validate_file_ref(ref,root=root)
    report=load_json(root/'training_report.json'); training.validate_report(report,spec,arm,bank_hash)
    a=load_npz_arrays(root/'validation_predictions.npz'); validate_predictions(a,spec['counts']['validation'])
    if evaluate_probabilities(a['labels'],a['probabilities'])!=report['validation']: raise ValueError('Saved metrics differ')
    return report,a


def completed(spec,task,*,seen=None):
    seen={} if seen is None else seen
    if task in seen: return seen[task]
    registered=next((t for t in spec['tasks'] if t['task_id']==task),None)
    if registered is None: raise ValueError('Unregistered task')
    root=Path(spec['root']); path=root/(task+'_receipt.json')
    if not path.is_file(): return None
    record=load_json(path); validate(record,'TASK',dict(spec=spec['content_hash']))
    deps={}
    for dep in registered['dependencies']:
        r=completed(spec,dep,seen=seen)
        if r is None: raise ValueError('Missing completed dependency')
        deps[dep]=r['content_hash']
    if (record['task_id']!=task or record['dependencies']!=deps or record['source_commit']!=spec['source_commit']
            or record['preflight_performed'] is not False
            or record['job_id']!=submission.accepted_jobs(spec).get(task)
            or [r['path'] for r in record['outputs']]!=output_names(spec,task)):
        raise ValueError('Completed task/source/job/output inventory differs')
    for ref in record['outputs']: validate_file_ref(ref,root=root)
    bundle=load_json(root/'targets/bundle.json')
    validate(bundle,'BUNDLE',dict(spec=spec['content_hash'],prepared=spec['anchor']['evidence']['prepared']))
    if bundle['role']!='train' or bundle['rows']!=spec['counts']['train'] or set(bundle['records'])!=set(TEACHERS):
        raise ValueError('Target bundle registration differs')
    for name,ref in bundle['records'].items():
        if ref!=file_ref(root/'targets'/name/'manifest.json'): raise ValueError('Bank manifest changed')
    if task!='export_targets': validate_fit(spec,task,bundle['content_hash'])
    seen[task]=record
    return record


def run(spec,task,*,device='cuda'):
    campaign.check(spec)
    if task not in ('export_targets',*ARMS): raise PermissionError('No unregistered operation')
    job=execution(spec,spec['resources'])
    if job!=submission.accepted_jobs(spec).get(task): raise PermissionError('Not the exact submitted allocation')
    deps={}
    if task!='export_targets':
        r=completed(spec,'export_targets')
        if r is None: raise ValueError('Shared target export incomplete')
        submission.successful_export(spec,r['job_id']); deps['export_targets']=r['content_hash']
    root=Path(spec['root']); attempt=root/(task+'_attempt'); attempt.mkdir(exist_ok=False)
    campaign.storage(spec); log=PhaseLog(attempt,lambda:memory_snapshot(torch))
    if task=='export_targets':
        with log.phase('export_actual_frozen_TRAIN_targets_no_preflight'): export_targets(spec,device)
    else:
        with log.phase('load_shared_targets_and_HLT'):
            meta=metadata(spec)
            tr=banks.data(spec,'train','D000',meta); val=banks.data(spec,'validation','D000',meta)
            bank=banks.load(spec,tr)
        with log.phase('fresh_thirty_epoch_science'): fit(root/task,spec,task,tr,val,bank,device)
    campaign.storage(spec); campaign.check(spec)
    if execution(spec,spec['resources'])!=job: raise PermissionError('Allocation changed')
    record=artifact('TASK',parents=dict(spec=spec['content_hash']),task_id=task,job_id=job,
        source_commit=spec['source_commit'],dependencies=deps,preflight_performed=False,
        outputs=[file_ref(root/p,root=root) for p in output_names(spec,task)])
    write_immutable_json(root/(task+'_receipt.json'),record)
    return record


def rejection(labels,probabilities,efficiency):
    if efficiency not in (.3,.5): raise ValueError('Unregistered signal efficiency')
    p=np.asarray(probabilities,np.float64); output={}
    for c,name in enumerate(CLASS_NAMES[1:],1):
        den=p[:,c]+p[:,0]; score=np.divide(p[:,c],den,out=np.zeros(len(p)),where=den>0)
        signal=np.sort(score[labels==c])[::-1]; qcd=score[labels==0]
        if not len(signal) or not len(qcd): raise ValueError('Missing rejection class')
        threshold=signal[math.ceil(efficiency*len(signal))-1]; passing=int(np.count_nonzero(qcd>=threshold))
        output[name]=dict(rejection=float(len(qcd)/passing) if passing else None,background_count=len(qcd),
            background_passing=passing,zero_fpr_censored=passing==0,threshold=float(threshold),
            achieved_signal_efficiency=float(np.mean(signal>=threshold)))
    return output


def results(spec):
    campaign.validate_spec(spec); root=Path(spec['root']); seen={}
    if completed(spec,'export_targets',seen=seen) is None: raise ValueError('Targets/reference snapshot not yet complete')
    original=load_npz_arrays(root/'references/M0HLT.npz')
    ix=load_npz_arrays(root/'references/original50k.npz')['indices']
    if (ix.dtype!=np.int64 or ix.shape!=(spec['original50k_rows'],) or len(np.unique(ix))!=len(ix)
            or np.any(ix<0) or np.any(ix>=len(original['labels']))): raise ValueError('Bad report-only subset')
    inputs=[(n,load_npz_arrays(root/'references'/(n+'.npz')),None) for n in reference_files(spec)]
    pending=[]; bundle=load_json(root/'targets/bundle.json')
    for arm in ARMS:
        if completed(spec,arm,seen=seen) is None: pending.append(arm)
        else:
            report,a=validate_fit(spec,arm,bundle['content_hash']); inputs.append((arm,a,report['selected_pass']))
    rows=[]
    for name,a,epoch in inputs:
        validate_predictions(a,spec['counts']['validation'])
        if not np.array_equal(a['identities'],original['identities']) or not np.array_equal(a['labels'],original['labels']):
            raise ValueError('Result population mismatch')
        row=dict(model=name,selected_epoch=epoch,historical=name not in ARMS)
        for role,indices in (('validation',np.arange(len(a['labels']))),('original50k',ix)):
            y,p=a['labels'][indices],a['probabilities'][indices]
            row[role]=evaluate_probabilities(y,p)
            row[role+'_r30']=rejection(y,p,.3); row[role+'_r50']=rejection(y,p,.5)
        rows.append(row)
    by={r['model']:r for r in rows}
    for row in rows:
        for role in ('validation','original50k'):
            row[role+'_recovery']=recovery(row[role],by['M0HLT'][role],by['OFFLINE'][role])
            row[role+'_delta_vs_multi_w5']=(row[role]['macro_ovr_auc']-by['multi_w5'][role]['macro_ovr_auc']
                                           if 'multi_w5' in by else None)
    return artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=rows,pending=pending,
        original50k_selection=False,single_seed=True,adaptive_validation=True,new_preflight_performed=False)


def print_results(result):
    print('Small ALLJETS multi-teacher screen: CE + representation prediction; NO logit KD.')
    print('Exploratory validation; no independent test. Original50k is report-only.')
    for role in ('validation','original50k'):
        print('\n'+role); print(f'{"model":23} {"pick":>5} {"accuracy":>10} {"AUC":>10} {"AUC rec%":>10}')
        for row in result['rows']:
            v=row[role]; rec=row[role+'_recovery']['macro_ovr_auc']; rec='n/a' if rec is None else f'{rec:.1f}'
            print(f'{row["model"]:23} {str(row["selected_epoch"] or "ref"):>5} {v["accuracy"]:10.6f} {v["macro_ovr_auc"]:10.6f} {rec:>10}')
            for efficiency in ('r30','r50'):
                fields=[]
                for c in ('X_bb','X_cc','X_gg','X_qq','X_ss'):
                    r=row[role+'_'+efficiency][c]; text='censored' if r['rejection'] is None else f'{r["rejection"]:.1f}'
                    fields.append(c+'='+text+f' ({r["background_passing"]} QCD)')
                print('  '+efficiency+': '+', '.join(fields))
    if result['pending']: print('PENDING/INCOMPLETE: '+', '.join(result['pending']))
