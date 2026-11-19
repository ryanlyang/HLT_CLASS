"""Authenticate selected Oscar closure, preserve original bytes, resolve locally.

Epoch state payloads are authenticated at export and deliberately not transferred.
Original records are never rewritten to masquerade as a new Oscar execution.
"""
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import re
import numpy as np
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, sha256_file
from hlt_classification.provenance import capture_source_snapshot
from hlt_classification.luka_sixclass_balanced_oscar import (
    campaign as original, artifacts, population, banks, training, contracts as s)
from . import contracts as c


def relative(name):
    if (not isinstance(name,str) or not name or '\\' in name or ':' in name
            or any(x in ('','.','..') for x in name.split('/'))):
        raise ValueError('Unsafe bundle path')
    return name


def local(root,name):
    root=Path(root).absolute();relative(name)
    if any(p.is_symlink() for p in (root,*root.parents)):
        raise ValueError('Bundle ancestor is a symlink')
    p=root
    for part in name.split('/'):
        p=p/part
        if p.is_symlink():raise ValueError('Bundle symlink')
    return p


def oscar_root(root):
    if not root.is_relative_to(Path('/oscar/home/rlyang/hlt_classification/checkpoints')):
        raise ValueError('Expected original Oscar checkpoint root')


def export(spec,project,output):
    """Read-only on the campaign; write one small manifest, never submit jobs."""
    from .campaign import donor_sources, source
    original.check(spec,live=False)
    donor_sources(project)
    snapshot=capture_source_snapshot(project,require_clean=True)
    source(project,snapshot['git_commit'])
    if capture_source_snapshot(spec['project'],require_clean=True)!=spec['source']:
        raise ValueError('Original source snapshot differs')
    pins=donor_sources(spec['project'])
    if original.runtime()!=spec['environment']:
        raise ValueError('Use the original registered Oscar environment for export')
    root=Path(spec['root']).resolve(strict=True)
    oscar_root(root)
    if load_json(root/'campaign_spec.json')!=spec:raise ValueError('Noncanonical original spec')
    accepted=original.accepted_jobs(spec)
    loaded=population.load(spec['dataset'],spec['population_root'])
    if loaded[0]!=spec['population']:raise ValueError('Population differs')
    a=loaded[4];cache=SimpleNamespace(role='train',identities=a['train_identities'],labels=a['train_labels'])
    # Bank only needs len, role, identities and labels for authentication.
    class Rows:
        role='train';identities=cache.identities;labels=cache.labels
        def __len__(self):return len(self.labels)
    files={};memo={};teacher_hashes={};accounting={}
    def add(path):
        path=Path(path).absolute()
        name=path.relative_to(root).as_posix();local(root,name)
        if not path.is_file():raise ValueError('Missing export file: '+str(path))
        files[name]=dict(path=name,bytes=path.stat().st_size,sha256=sha256_file(path))
    add(root/'campaign_spec.json')
    # Population metadata includes protected role masks, never test predictions.
    for path in sorted((root/'population').rglob('*')):
        if path.is_symlink():raise ValueError('Population symlink')
        if path.is_file():add(path)
    for path in sorted((root/'submission').rglob('*.json')):add(path)
    for stage in c.IMPORTED:
        print('Authenticating completed Oscar teacher:',stage,flush=True)
        r=artifacts.completed(spec,stage,memo=memo)
        if r['receipt']['job_id']!=c.JOBS[stage]:raise ValueError('Wrong original teacher job')
        row=original.terminal(c.JOBS[stage]);accounting[stage]=row
        if row[1]!='COMPLETED' or row[2]!='0:0':raise ValueError('Original teacher did not complete successfully')
        if not any(x['job']==c.JOBS[stage] and x['task']=='train_'+stage for x in accepted):
            raise ValueError('Original submission binding differs')
        bank=banks.Bank(r['folder']/'bank',Rows(),stage,r['refs'],spec)
        teacher_hashes[stage]=r['receipt']['content_hash']
        add(root/('train_'+stage)/'result.json')
        for ref in r['receipt']['files'].values():add(ref['path'])
        for ref in bank.manifest['payloads'].values():add(r['folder']/'bank'/ref['path'])
        for path in sorted((root/('train_'+stage)/'epochs').glob('*.json')):add(path)
        del bank
    result=c.artifact('BUNDLE',parents=dict(oscar_spec=spec['content_hash'],population=s.POPULATION_HASH),
        source_root=str(root),source_snapshot=spec['source'],export_source=snapshot,
        donor_sources=pins,teachers=teacher_hashes,jobs=c.JOBS,accounting=accounting,
        source_epoch_states_authenticated=True,epoch_states_transferred=False,
        files=[files[k] for k in sorted(files)],bytes=sum(r['bytes'] for r in files.values()))
    # Recheck original source after reading all payloads; immutable manifest last.
    if capture_source_snapshot(spec['project'],require_clean=True)!=spec['source']:
        raise ValueError('Original source drift during export')
    write_immutable_json(output,result)
    print(f'BUNDLE: {output}\nHASH: {result["content_hash"]}\nPAYLOAD: {result["bytes"]/2**30:.2f} GiB',flush=True)
    return result


class Import:
    """Resolver for original refs; no dependency on original absolute paths."""
    def __init__(self,root,expected,*,verify_payloads=False):
        self.root=Path(root).absolute()
        self.manifest=load_json(local(self.root,'bundle.json'));c.validate(self.manifest,'BUNDLE')
        m=self.manifest
        if (m['content_hash']!=expected or m['jobs']!=c.JOBS or set(m['teachers'])!=set(c.IMPORTED)
                or m.get('source_epoch_states_authenticated') is not True or m.get('epoch_states_transferred') is not False
                or any(m['accounting'].get(t)!=[c.JOBS[t],'COMPLETED','0:0'] for t in c.IMPORTED)):
            raise ValueError('Import campaign/teacher/export binding differs')
        self.source=PurePosixPath(m['source_root'])
        if not self.source.is_absolute() or '..' in self.source.parts:raise ValueError('Invalid original root')
        names=[relative(r['path']) for r in m['files']]
        if names!=sorted(set(names)) or sum(r['bytes'] for r in m['files'])!=m['bytes']:
            raise ValueError('Bundle inventory differs')
        self.files={r['path']:r for r in m['files']}
        self.verified=set()
        done=load_json(local(self.root,'transfer_complete.json'))
        if done!=dict(contract=c.PREFIX+'_TRANSFER_COMPLETE/v1',bundle_sha256=expected,
                      files=len(names),bytes=m['bytes'],all_files_verified=True):
            raise ValueError('Transfer incomplete or different')
        self.spec=load_json(self.path('campaign_spec.json'))
        original.check(self.spec,live=False)
        if (self.spec['root']!=str(self.source) or m['source_snapshot']!=self.spec['source']
                or m['parents']!=dict(oscar_spec=self.spec['content_hash'],population=s.POPULATION_HASH)):
            raise ValueError('Original scientific identity differs')
        self.jobs=self.accepted_jobs()
        self.memo={}
        for t in c.IMPORTED:self.completed(t)
        if verify_payloads:
            for name in names:self.path(name)

    def path(self,name):
        relative(name)
        r=self.files.get(name)
        if r is None:raise ValueError('Unregistered imported payload: '+name)
        p=local(self.root,name)
        if not p.is_file() or p.stat().st_size!=r['bytes']:raise ValueError('Imported size differs: '+name)
        if name not in self.verified:
            if sha256_file(p)!=r['sha256']:raise ValueError('Imported checksum differs: '+name)
            self.verified.add(name)
        return p

    def ref(self,ref):
        original_path=PurePosixPath(ref['path'])
        try:name=original_path.relative_to(self.source).as_posix()
        except ValueError:raise ValueError('Ref escapes original campaign') from None
        if self.files.get(name)!=dict(path=name,bytes=ref['bytes'],sha256=ref['sha256']):
            raise ValueError('Original reference and bundle differ')
        return self.path(name)

    def accepted_jobs(self):
        result=[]
        for name in self.files:
            if name.startswith('submission/') and name.endswith('.intent.json'):
                intent=load_json(self.path(name));s.validate(intent,'SUBMISSION_INTENT',dict(spec=self.spec['content_hash']))
                receipt=name.replace('.intent.json','.submitted.json')
                # Never tolerate an ambiguous original submission, even a later stage.
                r=load_json(self.path(receipt));s.validate(r,'SUBMITTED',dict(spec=self.spec['content_hash']))
                if (any(r[k]!=intent[k] for k in ('task','plan','command')) or r['task'] not in s.tasks()
                        or not re.fullmatch('[1-9][0-9]*',r['job'])):
                    raise ValueError('Original job intent differs')
                result.append(r)
        if len({r['job'] for r in result})!=len(result):raise ValueError('Duplicate original jobs')
        return result

    def completed(self,stage):
        if stage not in c.IMPORTED:raise PermissionError('Only four authorized Oscar teachers')
        if stage in self.memo:return self.memo[stage]
        r=load_json(self.path('train_'+stage+'/result.json'));s.validate(r,'RESULT')
        deps={t:self.completed(t) for t in s.TEACHERS[stage]}
        parents=artifacts.fit_parents(self.spec,stage,deps)
        if (r['stage']!=stage or r['job_id']!=c.JOBS[stage] or r['fit_parents']!=parents
                or r['content_hash']!=self.manifest['teachers'][stage]
                or not any(x['job']==r['job_id'] and x['task']=='train_'+stage for x in self.jobs)):
            raise ValueError('Imported teacher lineage/job differs')
        if set(r['files'])!={'checkpoint','report','predictions','outside_predictions','metrics','startup','view','bank'}:
            raise ValueError('Original result inventory differs')
        report=load_json(self.ref(r['files']['report']))
        training.validate_report(report,stage,parents,s.COUNTS['validation'])
        if r['parents']!=dict(spec=self.spec['content_hash'],report=report['content_hash']) or r['selected_epoch']!=report['selected_pass']:
            raise ValueError('Imported selection binding differs')
        rows=[]
        for epoch in range(1,s.EPOCHS+1):
            e=load_json(self.path(f'train_{stage}/epochs/{epoch:02d}.json'));s.validate(e,'EPOCH_COMMIT',parents)
            if e['stage']!=stage or e['epoch']!=epoch or e['job_id']!=r['job_id']:
                raise ValueError('Imported epoch binding differs')
            rows.append(e['row'])
        if rows!=report['validation_history']:raise ValueError('Imported history differs')
        start=load_json(self.ref(r['files']['startup']));s.validate(start,'STARTUP',dict(spec=self.spec['content_hash']))
        flags=('installed_weaver_parity','finite_connected_gradients','classifier_objective_verified',
               'native_deployment_parity','full_support','probe_discarded','integrated_two_epoch_miniature')
        if (start['stage']!=stage or any(start.get(k) is not True for k in flags)
                or start.get('main_head')!=s.main_policy(stage) or start.get('matching_serial_parity') is not False):
            raise ValueError('Imported acceptance differs')
        result=dict(spec=self.spec,receipt=r,report=report,
                    refs={k:r['files'][k] for k in ('checkpoint','report')},imported=self)
        self.memo[stage]=result
        return result

    def authenticate_population(self,dataset):
        loaded=population.load(dataset,self.path('population/population.json').parent)
        if loaded[0]!=self.spec['population']:raise ValueError('Transferred population differs')
        return loaded

    def bank(self,stage,cache):
        return ImportedBank(self,stage,cache)


class ImportedBank(banks.Bank):
    """Same FP32 bank checks and keys as Oscar, with an explicit ref resolver."""
    def __init__(self,imported,stage,cache):
        if cache.role!='train':raise PermissionError('TRAIN-only target access')
        result=imported.completed(stage);refs=result['refs'];spec=imported.spec
        m=load_json(imported.ref(result['receipt']['files']['bank']))
        s.validate(m,'BANK',banks.parents(spec,stage,refs))
        expected=dict(stage=stage,coordinate=s.COORDINATES[stage],role='train',rows=len(cache),width=128,
            temperature=2.,refs=refs,normalization=s.recipe()['normalization'],
            embedding_export=s.recipe()['embedding_export'],probability_export=s.recipe()['probability_export'])
        if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Imported bank semantics differ')
        for ref in refs.values():imported.ref(ref)
        folder=PurePosixPath(result['receipt']['files']['bank']['path']).parent
        shapes=dict(identities=((len(cache),32),np.uint8),labels=((len(cache),),np.int64),
            targets=((len(cache),128),np.float32),probabilities=((len(cache),6),np.float32),
            mean=((128,),np.float64),scale=((128,),np.float64))
        if set(m['payloads'])!=set(shapes):raise ValueError('Imported bank inventory differs')
        values={}
        for key,(shape,dtype) in shapes.items():
            ref=m['payloads'][key]
            if ref['path']!=key+'.npy':raise ValueError('Imported bank filename differs')
            value=np.load(imported.ref(dict(ref,path=str(folder/ref['path']))),allow_pickle=False)
            if value.shape!=shape or value.dtype!=dtype or not np.isfinite(value).all():
                raise ValueError('Imported bank shape/dtype/finiteness differs: '+key)
            value.flags.writeable=False;values[key]=value
        if (not np.array_equal(values['identities'],cache.identities) or not np.array_equal(values['labels'],cache.labels)
                or np.any(values['scale']<.001) or np.any(values['probabilities']<0)
                or not np.allclose(values['probabilities'].sum(1),1.,atol=2e-6,rtol=0)
                or np.max(np.abs(values['targets'].mean(0,dtype=np.float64)))>1e-5
                or np.max(values['targets'].std(0,dtype=np.float64))>1.00001):
            raise ValueError('Imported bank rows/normalization/probabilities differ')
        self.__dict__.update(values);self.manifest=m
