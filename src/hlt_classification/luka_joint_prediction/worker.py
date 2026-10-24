"""Two direct combined-loss fits with prior acceptance and scratch-only targets."""
from pathlib import Path
from tempfile import TemporaryDirectory
import gc
import shutil
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer,installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities,recovery
from hlt_classification.luka_attention.worker import data,execution
from hlt_classification.luka_attention.banks import counts
from hlt_classification.luka_u100_embedding.worker import teacher_readback,validate_predictions
from hlt_classification.luka_fullsim import regular_cache,alljets_data
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog,memory_snapshot
from .contracts import artifact,validate,recipe,ARMS
from . import banks,training,model,campaign,submission,source


def fit(directory,spec,arm,train,val,bank,*,device,technical=False):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    student=model.new_model(spec['nodes'][arm])
    report,state=training.train(student,train,val,bank,spec['nodes'][arm],device=device,technical=technical)
    training.validate_report(report,spec,arm,bank.manifest['content_hash'],technical=technical)
    expected=predict(student,val,device=device,batch_size=128)
    save_state(directory/'selected.pt',state)
    saved=torch.load(directory/'selected.pt',map_location='cpu',weights_only=True)
    if saved.keys()!=state.keys() or any(not torch.equal(saved[k],state[k]) for k in state):
        raise ValueError('Selected native state readback differs')
    del student
    native=DelphesParticleTransformer().to(device).eval(); native.load_state_dict(saved,strict=True)
    actual=predict(native,val,device=device,batch_size=128)
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
    if evaluate_probabilities(val.labels,actual)!=report['validation']:
        raise ValueError('Native selected validation metrics differ')
    atomic_publish_bytes(directory/'validation_predictions.npz',deterministic_npz_bytes(dict(
        identities=val.identities,labels=val.labels,probabilities=actual)))
    write_immutable_json(directory/'training_report.json',report)
    receipt=artifact('FIT',parents=dict(spec=spec['content_hash'],bank=bank.manifest['content_hash']),
        arm=arm,technical=technical,checkpoint_probability_roundtrip=True,
        outputs=[file_ref(directory/n,root=directory) for n in ('selected.pt','training_report.json','validation_predictions.npz')])
    write_immutable_json(directory/'receipt.json',receipt)
    del native; gc.collect()
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    return report,receipt


def validate_fit(directory,spec,arm,bank_hash,*,technical=False):
    directory=Path(directory); receipt=load_json(directory/'receipt.json')
    validate(receipt,'FIT',dict(spec=spec['content_hash'],bank=bank_hash))
    if (receipt['arm']!=arm or receipt['technical'] is not technical or not receipt['checkpoint_probability_roundtrip']
            or [r['path'] for r in receipt['outputs']]!=['selected.pt','training_report.json','validation_predictions.npz']):
        raise ValueError('Fit publication differs')
    for ref in receipt['outputs']: validate_file_ref(ref,root=directory)
    report=load_json(directory/'training_report.json')
    training.validate_report(report,spec,arm,bank_hash,technical=technical)
    arrays=load_npz_arrays(directory/'validation_predictions.npz'); validate_predictions(arrays,spec['counts']['validation'])
    if evaluate_probabilities(arrays['labels'],arrays['probabilities'])!=report['validation']:
        raise ValueError('Saved metrics differ')
    return report,arrays


def snapshot_references(root,spec,val,meta):
    root=Path(root); root.mkdir(exist_ok=False)
    refs={n:spec['anchor']['evidence']['models'][n]['predictions'] for n in ('M0HLT','OFFLINE')}
    refs.update(COARSE_U100=spec['teacher']['files']['predictions'],
        U100_EMBED_W5=spec['reference']['files']['predictions'],
        OFFLINE_EMBED_W5=spec['reference']['reference']['files']['predictions'])
    refs.update(spec['acceptance']['predictions'])
    for name,ref in refs.items():
        validate_file_ref(ref); a=load_npz_arrays(ref['path']); validate_predictions(a,len(val))
        if not np.array_equal(a['identities'],val.identities) or not np.array_equal(a['labels'],val.labels):
            raise ValueError('Reference population differs')
        atomic_publish_bytes(root/(name+'.npz'),deterministic_npz_bytes(a)); validate_file_ref(ref)
    ix=alljets_data.comparison_indices(meta[2],meta[3],val.identities,val.labels)
    atomic_publish_bytes(root/'original50k.npz',deterministic_npz_bytes(dict(indices=ix)))


def scratch_directory(spec):
    """Explicit registered disk scratch; never infer a fallback from environment."""
    base=Path(spec['scratch_base'])
    if not base.is_absolute() or not base.is_dir() or base.absolute()!=base.resolve(strict=True):
        raise PermissionError('Registered canonical node-local scratch is unavailable')
    if (base.is_relative_to(Path(spec['root']).parent)
            or base.is_relative_to(Path(spec['anchor']['evidence']['raw_container']))):
        raise PermissionError('Scratch must not be inside shared checkpoints/dataset')
    # Adapt the established SPORC attention-weight scratch mount check. tmpfs is
    # deliberately excluded: these large target banks must not consume RAM quota.
    mounts=[]
    for line in mountinfo().splitlines():
        left,right=line.split(' - ',1)
        mount=Path(left.split()[4].replace('\\040',' '))
        if base.is_relative_to(mount): mounts.append((len(mount.parts),right.split()[0]))
    if not mounts or max(mounts)[1] not in ('ext4','xfs','btrfs','overlay'):
        raise PermissionError('Token-bank scratch must be node-local disk, not Ceph/NFS/tmpfs')
    return base


def mountinfo():
    return Path('/proc/self/mountinfo').read_text()


def retained_bank(root,spec):
    """Authenticate retained descriptors, not the explicitly retired scratch payload."""
    root=Path(root); m=load_json(root/'bank_manifest.json')
    validate(m,'BANK',dict(recipe=recipe()['content_hash'],teacher=spec['teacher']['content_hash'],
                         prepared=spec['anchor']['evidence']['prepared']))
    if ((m['role'],m['coordinate'],m['width'],m['dtype'],m['rows'])!=('train','U100',128,'float32',spec['counts']['train'])
            or m['checkpoint']!=spec['teacher']['files']['checkpoint']
            or [r['path'] for r in m['outputs']]!=['tokens.npy','index.npz','jets.npy']):
        raise ValueError('Retained bank semantics differ')
    actual=file_ref(root/'bank_index.npz')
    if any(actual[k]!=m['outputs'][1][k] for k in ('bytes','sha256')):
        raise ValueError('Retained bank index hash differs')
    a=load_npz_arrays(root/'bank_index.npz'); n=spec['counts']['train']
    if (a['offsets'].dtype!=np.int64 or a['offsets'].shape!=(n+1,) or a['offsets'][0]!=0
            or a['offsets'][-1]!=m['tokens'] or np.any(np.diff(a['offsets'])<1) or np.any(np.diff(a['offsets'])>512)
            or a['identities'].dtype!=np.uint8 or a['identities'].shape!=(n,32)
            or len(np.unique(a['identities'],axis=0))!=n or a['labels'].dtype!=np.int64 or a['labels'].shape!=(n,)
            or any(a[k].dtype!=np.float64 or a[k].shape!=(128,) or not np.isfinite(a[k]).all() for k in ('mean','scale','jet_mean','jet_scale'))
            or np.any(a['scale']<1e-3) or np.any(a['jet_scale']<1e-3)
            or np.any(a['labels']<0) or np.any(a['labels']>=11)):
        raise ValueError('Retained bank index/moments differ')
    expected=spec['acceptance']['bank']
    if m['outputs'][0]!=expected['outputs'][0] or m['alignment']!=expected['alignment']:
        raise ValueError('TRAIN tokens differ from original accepted targets')
    return m


REFERENCE_NAMES=('M0HLT','OFFLINE','OFFLINE_EMBED_W5','COARSE_U100','U100_EMBED_W5','TOKEN_ALIGNED','TOKEN_SET')


def run(spec,task):
    if task not in ARMS: raise ValueError('Only two direct science tasks; no preflight')
    campaign.check(spec); campaign.storage(spec)
    job=execution(spec,spec['acceptance']['preflight'])
    if installed_environment()!=spec['anchor']['evidence']['environment']:
        raise ValueError('Original model environment changed')
    if submission.accepted_jobs(spec).get(task)!=job: raise ValueError('Worker is not the accepted exact job')
    root=Path(spec['root'])/task; root.mkdir(exist_ok=False)
    write_immutable_json(root/'claim.json',artifact('RUN_CLAIM',parents=dict(spec=spec['content_hash']),task=task,job_id=job))
    local=scratch_directory(spec)
    log=PhaseLog(root,lambda:memory_snapshot(torch)); torch.cuda.reset_peak_memory_stats()
    with log.phase('authenticate_inputs_and_teacher'):
        meta=regular_cache.metadata(spec['anchor']['evidence']['prepared_root'])
        train=data(spec,'train','D000',meta); val=data(spec,'validation','D000',meta)
        rich=data(spec,'train','U100',meta); banks.check_pair(rich,train)
        required=(int(counts(train).sum())+len(train))*128*4+512*2**20
        if shutil.disk_usage(local).free<required: raise OSError(f'Need {required/2**30:.2f}GiB job-local scratch')
        teacher=DelphesParticleTransformer().to('cuda').eval()
        ref=spec['teacher']['files']['checkpoint']; validate_file_ref(ref)
        teacher.load_state_dict(torch.load(ref['path'],map_location='cpu',weights_only=True),strict=True)
        for p in teacher.parameters(): p.requires_grad_(False)
        richval=data(spec,'validation','U100',meta)
        proof=teacher_readback(teacher,richval,spec['teacher'],'cuda')
        write_immutable_json(root/'teacher_readback.json',proof); del richval
        snapshot_references(root/'references',spec,val,meta)
    # No technical epoch, benchmark or new preflight; teacher export and saved
    # probability readback are input construction/authentication for this fit.
    with TemporaryDirectory(prefix='lukajp-'+job+'-',dir=local) as temporary:
        bank=None
        try:
            target_root=Path(temporary)/'targets'
            with log.phase('export_TRAIN_jet_and_block8_targets_once'):
                record=banks.export(target_root,teacher,rich,train,spec['teacher'],device='cuda')
                write_immutable_json(root/'bank_manifest.json',record)
                atomic_publish_bytes(root/'bank_index.npz',(target_root/'index.npz').read_bytes())
                retained_bank(root,spec)
                del teacher,rich; gc.collect(); torch.cuda.empty_cache()
                bank=banks.Bank(target_root,train,spec['teacher'])
            with log.phase('science_fixed30_'+task):
                fit(root/'fit',spec,task,train,val,bank,device='cuda')
        finally:
            if bank is not None: bank.close()
    print('Owned jet/token banks retired; native models, predictions and histories retained.',flush=True)
    if source.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']:
        raise ValueError('Source changed during worker')
    campaign.storage(spec)
    files=output_names()
    receipt=artifact('SCIENCE',parents=dict(spec=spec['content_hash'],acceptance=spec['acceptance']['content_hash']),
        arm=task,job_id=job,scratch_payloads_retired=True,new_preflight_performed=False,
        environment=installed_environment(),gpu=gpu_identity(),memory=memory_snapshot(torch),
        outputs=[file_ref(root/f,root=root) for f in sorted(files)])
    write_immutable_json(root/'receipt.json',receipt)
    return receipt


def output_names():
    return ({'teacher_readback.json','bank_manifest.json','bank_index.npz'}
        | {'references/'+n+'.npz' for n in (*REFERENCE_NAMES,'original50k')}
        | {'fit/'+n for n in ('selected.pt','training_report.json','validation_predictions.npz','receipt.json')})


def results(spec):
    campaign.validate_spec(spec)
    root=Path(spec['root']); arrays={}; pending=[]; first_bank=None; refroot=None
    for arm in ARMS:
        path=root/arm/'receipt.json'
        if not path.is_file(): pending.append(arm); continue
        receipt=load_json(path)
        validate(receipt,'SCIENCE',dict(spec=spec['content_hash'],acceptance=spec['acceptance']['content_hash']))
        if (receipt['arm']!=arm or receipt['scratch_payloads_retired'] is not True
                or receipt['new_preflight_performed'] is not False
                or receipt['environment']!=spec['acceptance']['preflight']['environment']
                or receipt['gpu']!=spec['acceptance']['preflight']['gpu']
                or len(receipt['outputs'])!=len(output_names())
                or {r['path'] for r in receipt['outputs']}!=output_names()):
            raise ValueError('Science publication coverage/runtime differs')
        for ref in receipt['outputs']: validate_file_ref(ref,root=root/arm)
        bank=retained_bank(root/arm,spec)
        if first_bank is None: first_bank=bank
        elif bank!=first_bank: raise ValueError('Arms do not share identical jet AND token targets')
        current=root/arm/'references'
        if refroot is None:
            refroot=current
            arrays={n:load_npz_arrays(current/(n+'.npz')) for n in REFERENCE_NAMES}
        else:
            for n in (*REFERENCE_NAMES,'original50k'):
                if file_ref(current/(n+'.npz'))['sha256']!=file_ref(refroot/(n+'.npz'))['sha256']:
                    raise ValueError('Reference snapshots differ')
        _,arrays['JOINT_'+arm.upper()+'_J4_T3']=validate_fit(root/arm/'fit',spec,arm,bank['content_hash'])
    if refroot is None:
        return artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=[],pending=pending)
    ix=load_npz_arrays(refroot/'original50k.npz')['indices']; baseline=arrays['M0HLT']
    if ix.dtype!=np.int64 or ix.shape!=(spec['original50k_rows'],) or len(np.unique(ix))!=len(ix) or np.any(ix<0) or np.any(ix>=len(baseline['labels'])):
        raise ValueError('Report-only original50k selection differs')
    rows=[]
    for name,a in arrays.items():
        validate_predictions(a,spec['counts']['validation'])
        if not np.array_equal(a['identities'],baseline['identities']) or not np.array_equal(a['labels'],baseline['labels']):
            raise ValueError('Comparison population/order differs')
        rows.append(dict(model=name,validation=evaluate_probabilities(a['labels'],a['probabilities']),
            original50k=evaluate_probabilities(a['labels'][ix],a['probabilities'][ix])))
    reference=next(r for r in rows if r['model']=='U100_EMBED_W5')
    for row in rows:
        for role in ('validation','original50k'):
            row[role+'_recovery']=recovery(row[role],rows[0][role],rows[1][role])
            row[role+'_delta_U100_w5']=row[role]['macro_ovr_auc']-reference[role]['macro_ovr_auc']
    return artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=rows,pending=pending)


def print_results(value):
    print('U100 joint prediction: CE+4*jet+3*tokens; NO KD. No independent test; original50k report-only.')
    for role in ('validation','original50k'):
        print('\n'+role); print(f'{"model":26} {"accuracy":>10} {"AUC":>10} {"recovery%":>10} {"delta U100-w5":>14}')
        for row in value['rows']:
            rec=row[role+'_recovery']['macro_ovr_auc']; formatted='n/a' if rec is None else f'{rec:.1f}'
            print(f'{row["model"]:26} {row[role]["accuracy"]:10.6f} {row[role]["macro_ovr_auc"]:10.6f} {formatted:>10} {row[role+"_delta_U100_w5"]:+14.8f}')
    print('Pending:',', '.join(value['pending']) or 'none')
