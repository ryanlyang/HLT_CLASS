"""Compact native TRAIN targets with exact teacher, branch and identity joins."""
from pathlib import Path
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_embedding_aux.model import head_input
from hlt_classification.luka_embedding_aux.targets import standardize
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.luka_fullstat_hybrid_embedding.kd_banks import validate_probabilities
from .contracts import artifact,validate,ARMS,STAGES,TEACHERS,teachers,kd_teacher,export_kinds,task_id


def directory(spec,arm,teacher):
    if teacher not in TEACHERS or arm not in ARMS: raise ValueError('Unknown teacher/arm')
    scope='targets' if teacher in ('OFFLINE','M0HLT') else task_id(arm,teacher)
    return Path(spec['root'])/scope/'banks'/teacher


def teacher_record(spec,arm,teacher,*,payloads=True):
    directory(spec,arm,teacher)
    if teacher in ('OFFLINE','M0HLT'):
        refs={k:spec['anchor']['evidence']['models'][teacher][k] for k in ('checkpoint','report')}; scope='shared'
    else:
        root=Path(spec['root'])/task_id(arm,teacher)/'fit'; scope=arm
        refs=dict(checkpoint=file_ref(root/'selected.pt'),report=file_ref(root/'training_report.json'))
        report=load_json(refs['report']['path']); validate(report,'TRAINING_REPORT')
        if (report['node']!=spec['nodes'][arm][teacher] or report['technical'] is not False
                or report['scientific_fit'] is not True or report['selected_state']!='native_backbone_only'):
            raise ValueError('New teacher is not this branch selected native model')
    if payloads:
        for ref in refs.values(): validate_file_ref(ref)
    return artifact('TEACHER',parents=dict(spec=spec['content_hash'],prepared=spec['anchor']['evidence']['prepared'],
        **{k:v['sha256'] for k,v in refs.items()}),scope=scope,teacher=teacher,coordinate=TEACHERS[teacher],files=refs)


def check_selected(native,record):
    validate(record,'TEACHER')
    for ref in record['files'].values(): validate_file_ref(ref)
    state=torch.load(record['files']['checkpoint']['path'],map_location='cpu',weights_only=True)
    actual=native.state_dict()
    if (state.keys()!=actual.keys() or any(k.startswith(('predictor.','predictors.')) for k in actual)
            or any(not torch.equal(state[k],actual[k].detach().cpu()) for k in state)):
        raise ValueError('Target export must use exact selected native checkpoint, no predictor')


def export(folder,native,cache,record,*,device):
    check_selected(native,record); teacher=record['teacher']; kinds=export_kinds(teacher)
    if (cache.role!='train' or cache.coordinate_name!=record['coordinate']
            or cache.foundation_sha256!=record['parents']['prepared'] or not kinds):
        raise PermissionError('Only registered TRAIN teacher view exports')
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
    native.to(device).eval().requires_grad_(False)
    arrays=dict(identities=cache.identities,labels=cache.labels); moments=None
    if 'embedding' in kinds:
        values=np.empty((len(cache),128),np.float32)
        with torch.inference_mode(),torch.autocast(torch.device(device).type,enabled=False):
            for start in range(0,len(cache),128):
                raw=cache.batch(np.arange(start,min(start+128,len(cache))))
                _,z=head_input(native,**{k:torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')})
                if not torch.isfinite(z).all(): raise ValueError('Nonfinite native embedding')
                values[start:start+len(z)]=z.float().cpu().numpy()
                if start%(128*128)==0: print(f'TRIAL_HYBRID export teacher={teacher} rows={start+len(z)}/{len(cache)}',flush=True)
        targets,mean,scale=standardize(values); arrays.update(targets=targets,mean=mean,scale=scale)
        moments=dict(mean=mean.tolist(),scale=scale.tolist())
    if 'kd' in kinds:
        arrays['probabilities']=predict(native,cache,device=device,temperature=2.,batch_size=128)
        validate_probabilities(arrays['probabilities'],len(cache))
    atomic_publish_bytes(folder/'targets.npz',deterministic_npz_bytes(arrays))
    check_selected(native,record)
    manifest=artifact('BANK',parents=dict(teacher=record['content_hash'],prepared=cache.foundation_sha256),
        teacher=record,role='train',coordinate=record['coordinate'],kinds=kinds,rows=len(cache),width=128,temperature=2.,
        normalization='TRAIN_population_FP64_floor1e-3',moments=moments,payload=file_ref(folder/'targets.npz',root=folder))
    write_immutable_json(folder/'manifest.json',manifest)
    restored=load(folder,cache,record)[1]
    if any(not np.array_equal(v,restored[k]) for k,v in arrays.items()): raise ValueError('Native bank readback differs')
    return manifest


def manifest(folder,record,prepared,rows,*,payload=True):
    folder=Path(folder); m=load_json(folder/'manifest.json')
    validate(m,'BANK',dict(teacher=record['content_hash'],prepared=prepared))
    if (m['teacher']!=record or m['role']!='train' or m['coordinate']!=record['coordinate'] or m['rows']!=rows
            or m['kinds']!=export_kinds(record['teacher']) or m['width']!=128 or m['temperature']!=2.
            or m['normalization']!='TRAIN_population_FP64_floor1e-3' or m['payload']['path']!='targets.npz'):
        raise ValueError('Target bank semantics/branch differs')
    if 'embedding' in m['kinds']:
        if (set(m['moments'])!={'mean','scale'} or any(np.asarray(m['moments'][k]).shape!=(128,) for k in m['moments'])
                or not np.isfinite(list(m['moments'].values())).all() or min(m['moments']['scale'])<1e-3):
            raise ValueError('Retained normalization invalid')
    elif m['moments'] is not None: raise ValueError('Unexpected embedding moments')
    if payload: validate_file_ref(m['payload'],root=folder)
    return m


def load(folder,cache,record):
    if cache.role!='train' or cache.coordinate_name not in ('OFFLINE',*STAGES.values()):
        raise PermissionError('Only joined TRAIN roles can read targets')
    m=manifest(folder,record,cache.foundation_sha256,len(cache)); a=load_npz_arrays(Path(folder)/'targets.npz')
    keys={'identities','labels'}|({'targets','mean','scale'} if 'embedding' in m['kinds'] else set())|({'probabilities'} if 'kd' in m['kinds'] else set())
    if (set(a)!=keys or a['identities'].dtype!=np.uint8 or a['identities'].shape!=(len(cache),32)
            or len(np.unique(a['identities'],axis=0))!=len(cache) or not np.array_equal(a['identities'],cache.identities)
            or a['labels'].dtype!=np.int64 or a['labels'].shape!=(len(cache),) or not np.array_equal(a['labels'],cache.labels)):
        raise ValueError('Exact ordered TRAIN identity/label join failed')
    if 'targets' in a:
        if (a['targets'].dtype!=np.float32 or a['targets'].shape!=(len(cache),128)
                or any(a[k].dtype!=np.float64 or a[k].shape!=(128,) for k in ('mean','scale'))
                or any(not np.isfinite(a[k]).all() for k in ('targets','mean','scale'))
                or any(a[k].tolist()!=m['moments'][k] for k in ('mean','scale'))):
            raise ValueError('Invalid standardized native targets')
    if 'probabilities' in a: validate_probabilities(a['probabilities'],len(cache))
    return m,a


class Bundle:
    def __init__(self,spec,arm,stage,cache):
        if cache.coordinate_name!=STAGES[stage]: raise ValueError('Student view differs')
        self.targets={}; records={}; self.identities=cache.identities; self.labels=cache.labels
        for t in dict.fromkeys(teachers(stage)+[kd_teacher(stage)]):
            teacher=teacher_record(spec,arm,t)
            m,a=load(directory(spec,arm,t),cache,teacher); records[t]=m
            if t in teachers(stage): self.targets[t]=a['targets']
            if t==kd_teacher(stage): self.probabilities=a['probabilities']
        self.manifest=artifact('BUNDLE',parents=dict(spec=spec['content_hash'],**{t:m['content_hash'] for t,m in records.items()}),
            arm=arm,stage=stage,teacher_order=teachers(stage),kd_teacher=kd_teacher(stage),banks=records)


def validate_bundle(value,spec,arm,stage,*,payloads):
    expected={}
    for t in dict.fromkeys(teachers(stage)+[kd_teacher(stage)]):
        expected[t]=manifest(directory(spec,arm,t),teacher_record(spec,arm,t,payloads=payloads),spec['anchor']['evidence']['prepared'],
                             spec['counts']['train'],payload=payloads)
    validate(value,'BUNDLE',dict(spec=spec['content_hash'],**{t:m['content_hash'] for t,m in expected.items()}))
    if (value['arm']!=arm or value['stage']!=stage or value['teacher_order']!=teachers(stage)
            or value['kd_teacher']!=kd_teacher(stage) or value['banks']!=expected):
        raise ValueError('Incoming teacher bundle differs')
