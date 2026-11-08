"""Actual production miniatures and representative A6000 qualification."""
from pathlib import Path
from types import SimpleNamespace
import time
import math
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.runner import predict, _forward
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_fullsim.preflight_v2 import memory_snapshot
from hlt_classification.luka_u100_contrastive.resources import Subset
from hlt_classification.luka_selective_teacher_hydra.resources import free, mini_indices
from hlt_classification.luka_u100_contrastive.training import optimizer
from hlt_classification.luka_u100_multiview.views import lengths
from .contracts import FITS, TEACHERS, STUDENTS, artifact, validate
from .student_model import Native, new_model
from . import inputs, teacher_training as tt, student_training as training, embedding_banks


def storage_inventory(spec,physicals,structures,stats,maximum):
    """Concrete serialized checkpoint sizes plus complete array inventories."""
    import io
    import shutil
    from .contracts import STRUCTURES
    physical_bytes=sum(p.values.nbytes+p.offsets.nbytes+p.identities.nbytes+p.labels.nbytes for p in physicals.values())
    structure_bytes=sum(sum(x.nbytes for x in s.values.values())+sum(x.nbytes for x in s.valid.values()) for s in structures.values())
    embedding_bytes=16*(spec['counts']['train']*(128*4+32+8)+128*16) # eight technical and eight science banks
    checkpoints=0; sizes={}
    for task in FITS:
        with torch.random.fork_rng(devices=[]):
            model=(tt.new_model(spec,task,maximum,stats,structures['train','OFFLINE'].normalizers) if task in TEACHERS
                   else new_model(spec,task,structures['train','HLT' if task=='S08' else 'OFFLINE'].normalizers))
        total=0
        for state in (model.state_dict(),model.native_state()):
            b=io.BytesIO(); torch.save(state,b); total+=b.tell(); b.close()
        # Scientific + setup miniature, plus actual-target startup for students.
        sizes[task]=total; checkpoints+=total*(2 if task in TEACHERS else 3)
        del model
    predictions=22*(spec['counts']['validation']*(32+8+11*4)+1024)*3
    # Two GiB covers JSON epoch metrics/receipts/logs and transient publication.
    total=physical_bytes+structure_bytes+embedding_bytes+checkpoints+predictions+2*2**30
    used=sum(p.stat().st_size for p in Path(spec['root']).rglob('*') if p.is_file())
    if total>spec['storage_budget_bytes'] or shutil.disk_usage(spec['root']).free<max(0,total-used)+2*2**30:
        raise OSError('Concrete output/temporary inventory exceeds registered space')
    value=artifact('OUTPUT_INVENTORY',parents=dict(spec=spec['content_hash']),physical_bytes=physical_bytes,
        structure_bytes=structure_bytes,embedding_bytes=embedding_bytes,checkpoint_bytes=checkpoints,
        serialized_checkpoint_pairs=sizes,prediction_allowance_bytes=predictions,metadata_temporary_allowance_bytes=2*2**30,
        required_bytes=total,registered_bytes=spec['storage_budget_bytes'],headroom_bytes=2*2**30)
    write_immutable_json(Path(spec['root'])/'setup/output_inventory.json',value)
    return value

def replay(spec, val, device):
    bundle = inputs.Bundle(spec['bundle_root'])
    saved = load_npz_arrays(bundle.path('reference/matched/predictions.npz'))
    if any(not np.array_equal(saved[k], getattr(val, k)) for k in ('identities', 'labels')):
        raise ValueError('Historical native replay identity join differs')
    model = Native(128).to(device).eval()
    ref = file_ref(bundle.path('reference/matched/checkpoint.pt'))
    model.load_state_dict(torch.load(ref['path'], map_location='cpu', weights_only=True), strict=True)
    actual = predict(model, val, device=device, batch_size=128)
    folder = Path(spec['root'])/'setup'
    atomic_publish_bytes(folder/'replay_predictions.npz', deterministic_npz_bytes(dict(
        identities=val.identities, labels=val.labels, probabilities=actual)))
    old_auc = evaluate_probabilities(val.labels, saved['probabilities'])['macro_ovr_auc']
    new_auc = evaluate_probabilities(val.labels, actual)['macro_ovr_auc']
    value = artifact('REPLAY', parents=dict(spec=spec['content_hash']), checkpoint=ref,
        predictions=file_ref(folder/'replay_predictions.npz', root=folder), historical_auc=old_auc,
        actual_auc=new_auc, auc_abs=abs(new_auc-old_auc), tolerance=5e-5,
        precision='original_BF16_path', rows=len(val))
    write_immutable_json(folder/'replay.json', value)
    if value['auc_abs'] > value['tolerance']:
        raise ValueError('Historical native numerical replay differs; preserve diagnostics')
    del model; free()
    return value


def check_replay(spec):
    folder = Path(spec['root'])/'setup'; value = load_json(folder/'replay.json')
    validate(value, 'REPLAY', dict(spec=spec['content_hash']))
    b = inputs.Bundle(spec['bundle_root']); expected = load_npz_arrays(b.path('reference/matched/predictions.npz'))
    if value['checkpoint'] != file_ref(b.path('reference/matched/checkpoint.pt')) or value['predictions']['path'] != 'replay_predictions.npz':
        raise ValueError('Replay ownership differs')
    validate_file_ref(value['predictions'], root=folder)
    actual = load_npz_arrays(folder/'replay_predictions.npz')
    if any(not np.array_equal(actual[k], expected[k]) for k in ('identities', 'labels')):
        raise ValueError('Replay population differs')
    a, e = [evaluate_probabilities(x['labels'], x['probabilities'])['macro_ovr_auc'] for x in (actual, expected)]
    if (value['actual_auc'] != a or value['historical_auc'] != e or value['auc_abs'] != abs(a-e)
            or value['tolerance'] != 5e-5 or value['precision'] != 'original_BF16_path'
            or abs(a-e) > 5e-5 or value['rows'] != spec['counts']['validation']):
        raise ValueError('Replay numerical evidence differs')
    return value


class PhysicalSubset:
    def __init__(self,full,indices):
        self.full=full; self.indices=np.asarray(indices); self.role=full.role; self.source=full.source
        self.manifest=full.manifest; self.identities=full.identities[indices]; self.labels=full.labels[indices]
    def __len__(self): return len(self.indices)
    def jet(self,i): return self.full.jet(self.indices[i])


def subset_structure(full,ix):
    obj=SimpleNamespace(manifest=full.manifest,role=full.role,normalizers=full.normalizers,
        values={n:x[ix] for n,x in full.values.items()},valid={n:x[ix] for n,x in full.valid.items()},
        identities=full.identities[ix],labels=full.labels[ix])
    obj.target=lambda i,name: full.target(ix[i],name)
    return obj


def subset_banks(banks,ix):
    return {k:subset_structure(b,ix) if k=='structure' else SimpleNamespace(manifest=b.manifest,
        targets=b.targets[ix],identities=b.identities[ix],labels=b.labels[ix]) for k,b in banks.items()}


def miniature(folder,spec,task,train,val,banks,device):
    ix,vi=mini_indices(train),mini_indices(val)
    model,report=training.fit(folder,spec,task,Subset(train,ix),Subset(val,vi),subset_banks(banks,ix),device=device,technical=True)
    del model; free()
    return report


def timed(fn,device):
    training.synchronize(device); start=time.perf_counter(); fn(); training.synchronize(device)
    return time.perf_counter()-start


def measure(spec,device='cuda'):
    if torch.device(device).type!='cuda': raise PermissionError('Genuine GPU qualification required')
    if installed_environment()!=spec['reference']['environment']: raise ValueError('Installed environment differs')
    folder=Path(spec['root'])/'setup'; profiles={}; status='failed'; error=None
    try:
        began=time.perf_counter(); train,val=inputs.views(spec); physicals,structures,stats,prep=inputs.prepare(spec,train,val)
        preparation=time.perf_counter()-began; replay(spec,val,device)
        storage_inventory(spec,physicals,structures,stats,prep['maximum'])
        ptrain,pval=physicals['train','OFFLINE'],physicals['validation','OFFLINE']
        strain,sval=structures['train','OFFLINE'],structures['validation','OFFLINE']
        ix,vi=mini_indices(train),mini_indices(val)
        pt,pv=PhysicalSubset(ptrain,ix),PhysicalSubset(pval,vi)
        st,sv=subset_structure(strain,ix),subset_structure(sval,vi)
        order=np.random.default_rng(2026110809).permutation(len(train))
        # Validate reconstructed clean features against historical native OFFLINE.
        from hlt_classification.luka_teacher_sweep_hydra.transfer import Delta
        from .corruption import batch
        old=Delta(spec['offline_root'],spec['offline_hash'],spec['bundle_root']).load(spec,'train')
        physical_batch=batch([ptrain.jet(i) for i in ix[:128]])
        expected=old.batch(ix[:128])
        for key in ('features','vectors','mask'): np.testing.assert_array_equal(physical_batch[key],expected[key])
        del old
        for task in (*TEACHERS,*STUDENTS):
            print('PRETEXT setup qualifying '+task,flush=True); free(); torch.cuda.reset_peak_memory_stats(device)
            export_seconds=0.
            if task in TEACHERS:
                technical_root=folder/('technical_'+task); technical_root.mkdir()
                model,report=tt.fit(technical_root/'fit',spec,task,pt,pv,st,sv,stats,prep['maximum'],device=device,technical=True)
                # These real miniature-selected embeddings are only technical
                # targets, never published as a scientific teacher result.
                export_seconds=timed(lambda:embedding_banks.export(technical_root/'bank',model,ptrain,spec,task,
                    file_ref(technical_root/'fit/selected.pt'),device=device,technical=True),device)
                del model; free()
                model=tt.new_model(spec,task,prep['maximum'],stats,strain.normalizers).to(device); opt=optimizer(model)
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(spec['nodes'][task]['initialization_seed'])
                    factory=Native(128).to(device).eval()
                model.eval()
                from hlt_classification.luka_embedding_aux.model import head_input
                raw=batch([ptrain.jet(i) for i in ix[:128]])
                with torch.no_grad():
                    tensors={k:torch.as_tensor(v,device=device) for k,v in raw.items()}
                    _,expected_h=head_input(factory,**tensors)
                    torch.testing.assert_close(model.embedding(raw,device,bf16=False),expected_h,rtol=0,atol=0)
                del factory,expected_h,tensors
                fn=lambda rows,diagnose=False,phases=None:tt.step(model,opt,ptrain,strain,rows,3.,1,device,diagnose=diagnose,phases=phases)
                stress=np.argsort(np.diff(ptrain.offsets),kind='stable')[-128:]
                validation=lambda:tt.evaluate(model,pval,sval,device)
            else:
                banks=inputs.banks(spec,task,train,technical=True)
                miniature(folder/('mini_'+task),spec,task,train,val,banks,device)
                structure=banks.get('structure'); model=new_model(spec,task,None if structure is None else structure.normalizers).to(device)
                # Installed factory parity on the exact native state and input.
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(spec['nodes'][task]['initialization_seed']); factory=Native(128).to(device).eval()
                model.eval(); model.capture=False
                with torch.no_grad():
                    torch.testing.assert_close(_forward(model,train.batch(ix[:128]),device,bf16=False),
                        _forward(factory,train.batch(ix[:128]),device,bf16=False),rtol=0,atol=0)
                del factory; opt=optimizer(model)
                fn=lambda rows,diagnose=False,phases=None:training.step(model,task,opt,train.batch(rows),banks,rows,3.,device,diagnose=diagnose,phases=phases)
                stress=np.argsort(lengths(train),kind='stable')[-128:]
                def validation():
                    model.eval(); model.capture=False; model.predictions=None; model.structures=None
                    return predict(model,val,device=device,batch_size=128)
            timed(lambda:fn(stress),device); timed(lambda:fn(order[:11]),device)
            for b in range(2): timed(lambda:fn(order[b*128:(b+1)*128]),device)
            samples=[timed(lambda:fn(order[b*128:(b+1)*128]),device) for b in range(2,14)]
            phase_samples=[]
            for b in range(2,5):
                phases={}; timed(lambda:fn(order[b*128:(b+1)*128],phases=phases),device); phase_samples.append(phases)
            probe=timed(lambda:fn(order[:128],True),device)
            if task in STUDENTS:
                probe+=timed(lambda:training.target_probe(model,task,train,banks,device),device)
            validation_seconds=timed(validation,device)
            profiles[task]=dict(initial_factory_fp32_parity=True,clean_frontend_parity=True,
                miniature_complete=True,selected_reload=True,largest_support_backward=True,partial_batch_backward=True,
                technical_targets=task not in TEACHERS,train_rows=len(train),validation_rows=len(val),
                step_samples=samples,phase_samples=phase_samples,step_seconds=float(np.mean(samples)),validation_seconds=validation_seconds,
                export_seconds=export_seconds,probe_seconds=probe,load_seconds=preparation,memory=memory_snapshot(torch))
            print(f'PRETEXT profile {task}: step={np.mean(samples):.3f}s validation={validation_seconds:.1f}s export={export_seconds:.1f}s',flush=True)
            del model,opt
            if task in STUDENTS: del banks
            free()
        status='complete'
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'; raise
    finally:
        diagnostic=artifact('RESOURCES',parents=dict(spec=spec['content_hash']),status=status,error=error,
            profiles=profiles,environment=installed_environment(),gpu=gpu_identity())
        write_immutable_json(folder/'resource_diagnostic.json',diagnostic)
    acceptance=accept(spec,diagnostic); write_immutable_json(folder/'acceptance.json',acceptance)
    return acceptance


def accept(spec,diagnostic):
    validate(diagnostic,'RESOURCES',dict(spec=spec['content_hash']))
    if (diagnostic['status']!='complete' or diagnostic['error'] is not None or set(diagnostic['profiles'])!=set(FITS)
            or diagnostic['environment']!=spec['reference']['environment'] or diagnostic['gpu']['name']!='NVIDIA RTX A6000'
            or diagnostic['gpu']['compute_capability']!=[8,6]): raise ValueError('Incomplete/incorrect real setup')
    minutes={}
    for task,p in diagnostic['profiles'].items():
        phase_names=({'input_corruption','native_forward','auxiliary_forward_assignment','diagnostics','backward_optimizer'}
                     if task in TEACHERS else {'native_forward','targets_and_auxiliary','diagnostics','backward_optimizer'})
        if (not all(p[k] is True for k in ('initial_factory_fp32_parity','clean_frontend_parity','miniature_complete',
                'selected_reload','largest_support_backward','partial_batch_backward')) or len(p['step_samples'])!=12
                or not np.isfinite(p['step_samples']).all() or min(p['step_samples'])<=0
                or p['step_seconds']!=float(np.mean(p['step_samples']))
                or p['train_rows']!=spec['counts']['train'] or p['validation_rows']!=spec['counts']['validation']):
            raise ValueError('Measurement coverage differs')
        if (len(p.get('phase_samples',[]))!=3 or any(set(row)!=phase_names
                or not np.isfinite(list(row.values())).all() or min(row.values())<0 or sum(row.values())<=0
                for row in p['phase_samples'])):
            raise ValueError('Resource phase coverage differs')
        memory=p['memory']
        if any(not isinstance(memory.get(k),(int,float)) or not math.isfinite(memory[k]) or memory[k]<=0 for k in ('peak_rss_bytes','gpu_peak_bytes')):
            raise ValueError('Missing/nonfinite memory measurement')
        if memory['peak_rss_bytes']>=.85*spec['memory_mb']*2**20 or memory['gpu_peak_bytes']>=.88*diagnostic['gpu']['total_memory_bytes']:
            raise MemoryError(task+' exceeds registered memory envelope')
        timings=[p[k] for k in ('validation_seconds','probe_seconds','load_seconds','export_seconds')]
        if not np.isfinite(timings).all() or min(timings)<0 or p['validation_seconds']<=0: raise ValueError('Invalid timings')
        seconds=30*(math.ceil(p['train_rows']/128)*p['step_seconds']+p['validation_seconds'])
        # Startup miniature, diagnostics, reads and bank export are included.
        seconds+=5*p['probe_seconds']+p['load_seconds']+p['export_seconds']+8*p['step_seconds']+2*p['validation_seconds']
        minutes[task]=max(60,math.ceil((2*seconds+900)/60))+15
        if minutes[task]>spec['fit_max_minutes']: raise RuntimeError(f'{task} needs {minutes[task]}min; preserve diagnostics')
    return artifact('ACCEPTANCE',parents=dict(spec=spec['content_hash'],diagnostic=diagnostic['content_hash']),minutes=minutes,
        gpu=diagnostic['gpu'],environment=diagnostic['environment'],technical_weights_discarded=True)


def accepted(spec):
    folder=Path(spec['root'])/'setup'; saved=load_json(folder/'acceptance.json')
    if saved!=accept(spec,load_json(folder/'resource_diagnostic.json')): raise ValueError('Acceptance differs from measurements')
    return saved
