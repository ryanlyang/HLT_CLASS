"""Native selected CONCAT/FUSION TRAIN logit exports, once, with exact joins."""
from pathlib import Path
import gc
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,atomic_publish_bytes,
    deterministic_npz_bytes,write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_fullstat_multiview import views
from hlt_classification.luka_fullstat_multiview.model import new_model as new_teacher
from hlt_classification.luka_fullstat_multiview.startup import stress_ids
from hlt_classification.luka_fullstat_multiview.resources import check_memory
from hlt_classification.luka_fullstat_fine_tail.kd_banks import validate_probabilities
from hlt_classification.luka_fullstat_retained.worker import sync
from hlt_classification.luka_fullsim.preflight_v2 import memory_snapshot
from hlt_classification.luka_u100_multiview import teachers
from .contracts import artifact,validate,TASK

NAMES=('CONCAT','FUSION')
FILES=('probabilities.npz','manifest.json')


def export_recipe():
    return artifact('KD_EXPORT_RECIPE',role='train',temperature=2.,classes=11,batch=128,
                    forward='native_BF16_cuda_FP32_cpu',softmax='FP32_logits_div_T',teachers=list(NAMES))


def parents(spec,name):
    item=spec['reuse']['banks'][name]
    owner=load_json(validate_file_ref(item['owner']))
    return dict(recipe=export_recipe()['content_hash'],source=spec['source_commit'],
        prepared=spec['anchor']['evidence']['prepared'],owner=owner['content_hash'],
        checkpoint=item['refs']['checkpoint']['sha256'],report=item['refs']['report']['sha256'])


def folder(spec,name):
    if name not in NAMES:raise ValueError('Unregistered new logit export')
    return Path(spec['root'])/'logit_targets'/name


def manifest(spec,name,*,verify_payload=True):
    f=folder(spec,name);m=load_json(f/'manifest.json')
    validate(m,'KD_BANK',parents(spec,name))
    if (m['teacher']!=name or m['role']!='train' or m['rows']!=spec['counts']['train']
            or m['temperature']!=2. or m['class_count']!=11 or m['coordinate']!=name
            or m['payload']['path']!='probabilities.npz' or m['export_batch']!=128):
        raise ValueError('Native KD bank semantics differ')
    if verify_payload:validate_file_ref(m['payload'],root=f)
    return m


def arrays(spec,name,cache):
    m=manifest(spec,name)
    if (cache.role!='train' or cache.coordinate_name!='D000'
            or cache.foundation_sha256!=spec['anchor']['evidence']['prepared']):
        raise PermissionError('Only registered HLT TRAIN may consume KD')
    a=load_npz_arrays(validate_file_ref(m['payload'],root=folder(spec,name)))
    if (set(a)!={'identities','labels','probabilities'} or len(cache)!=m['rows']
            or a['identities'].dtype!=np.uint8 or a['identities'].shape!=(len(cache),32)
            or a['labels'].dtype!=np.int64 or not np.array_equal(a['identities'],cache.identities)
            or not np.array_equal(a['labels'],cache.labels)):
        raise ValueError('Native KD ordered TRAIN join differs')
    validate_probabilities(a['probabilities'],len(cache))
    for v in a.values():v.flags.writeable=False
    return m,a


def export(spec,name,train,net,device,started):
    if (name not in NAMES or train.role!='train' or train.coordinate_name!=name
            or len(train)!=spec['counts']['train'] or train.foundation_sha256!=spec['anchor']['evidence']['prepared']):
        raise PermissionError('Native KD requires exact teacher TRAIN view')
    refs=spec['reuse']['banks'][name]['refs']
    state=torch.load(validate_file_ref(refs['checkpoint']),map_location='cpu',weights_only=True)
    if state.keys()!=net.state_dict().keys() or any(not torch.equal(v.cpu(),state[k]) for k,v in net.state_dict().items()):
        raise ValueError('Export model is not selected frozen teacher')
    del state
    net=net.to(device).eval().requires_grad_(False)
    f=folder(spec,name);f.mkdir(parents=True,exist_ok=False)
    with torch.inference_mode():
        stress=teachers.forward(net,train.batch(stress_ids(train)),device,
            bf16=torch.device(device).type=='cuda',name=name)
        if not torch.isfinite(stress).all():raise ValueError('Nonfinite native teacher stress')
        del stress
    p=np.empty((len(train),11),np.float32);steps=math.ceil(len(train)/128)
    sync(device);tick=time.monotonic()
    with torch.inference_mode():
        for batch,start in enumerate(range(0,len(train),128),1):
            ids=np.arange(start,min(start+128,len(train)))
            logits=teachers.forward(net,train.batch(ids),device,bf16=torch.device(device).type=='cuda',name=name)
            batch_p=(logits.float()/2.).softmax(1).cpu().numpy()
            if batch_p.shape!=(len(ids),11) or not np.isfinite(batch_p).all():
                raise ValueError('Invalid native teacher probabilities')
            p[ids]=batch_p
            if batch==min(64,steps):
                sync(device);seconds=(time.monotonic()-tick)/batch
                # Reserve 12 hours for the full student/validation/publication.
                estimate=time.monotonic()-started+1.5*max(0,steps-batch)*seconds+12*3600
                write_immutable_json(f/'timing.json',artifact('EXPORT_TIMING',parents=parents(spec,name),
                    seconds_per_batch=seconds,measured_batches=batch,remaining_estimate_seconds=estimate,
                    full_export_measured=False))
                if estimate>60*spec['minutes'][TASK]:raise ValueError('Native export plus student reserve exceeds allocation')
            if batch%256==0:
                print(f'SIX TEACHER native KD export={name} rows={start+len(ids)}/{len(train)}',flush=True)
    validate_probabilities(p,len(train));check_memory(spec,memory_snapshot(torch))
    for key in ('checkpoint','report'):validate_file_ref(refs[key])
    atomic_publish_bytes(f/'probabilities.npz',deterministic_npz_bytes(dict(
        identities=train.identities,labels=train.labels,probabilities=p)))
    m=artifact('KD_BANK',parents=parents(spec,name),teacher=name,role='train',coordinate=name,
        rows=len(train),temperature=2.,class_count=11,export_batch=128,
        payload=file_ref(f/'probabilities.npz',root=f))
    write_immutable_json(f/'manifest.json',m)
    manifest(spec,name)
    return m


def prepare(spec,device,started):
    """Serial view/model lifetimes avoid keeping two privileged caches resident."""
    for name in NAMES:
        if spec.get('resume') is not None and name in spec['resume']['exports']:
            old=load_json(validate_file_ref(spec['resume']['spec']))
            manifest(old,name);dest=folder(spec,name);dest.mkdir(parents=True,exist_ok=False)
            for n in FILES:
                atomic_publish_bytes(dest/n,(folder(old,name)/n).read_bytes())
            manifest(spec,name)
            continue
        item=spec['reuse']['banks'][name]
        owner=load_json(validate_file_ref(item['owner']))
        net=new_teacher(owner['nodes']['teacher_'+name],'teacher_'+name)
        net.load_state_dict(torch.load(validate_file_ref(item['refs']['checkpoint']),map_location='cpu',weights_only=True),strict=True)
        # The established loader authenticates both development roles; only
        # TRAIN is forwarded/exported and the validation view is freed here.
        train,val,_=views.load(spec,'teacher_'+name)
        del val
        export(spec,name,train,net,device,started)
        del train,net
        gc.collect()
        if torch.device(device).type=='cuda':torch.cuda.empty_cache()
