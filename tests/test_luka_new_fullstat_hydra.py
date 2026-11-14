"""New snapshot semantics, sealed reads and portable transfer; no remote acceptance claim."""
import importlib.util
import io
import json
import hashlib
from pathlib import Path
import tarfile
import numpy as np
import pytest
import uproot
from hlt_classification.data.cache_contracts import sha256_file
from hlt_classification.jetclass2_delphes.schema import SIGNAL_IDS
from hlt_classification.luka_new_fullstat_hydra import contracts as c, population as p, prepare
from test_luka_fullstat import write_root


@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
    root=tmp_path_factory.mktemp('new_fullstat');lines=[]
    samples=[('qcd','jetclass2sb/QCD_PT-170to300/261028_081242/0000/ntuple_1.root',[161]*160),
             ('qcd','jetclass2sb/QCD_PT-300to470/261028_081422/0000/ntuple_2.root',[187]*80),
             ('nonqcd','jetclass2sb/train_higgs2p/merged/ntuple_merged_0.root',list(SIGNAL_IDS)*16+[161]*16+[206]*5)]
    for side,rel,labels in samples:
        path=root/'raw'/side/rel;write_root(path,labels)
        prefix=next(k for k,v in c.SOURCES.items() if v==side)
        lines.append(f'{sha256_file(path)}  {prefix+rel}\n')
    (root/'source.sha256').write_text(''.join(lines),encoding='utf-8-sig')
    f=p.build(root,root/'foundation',production=False,acknowledge_row_split=True,
              counts=dict(train=264,validation=66,final_test=66,unused=4))
    return root,f


def test_known_pool_quarantine_and_replay(dataset):
    root,f=dataset;again,arrays=p.load(root/'foundation',production=False)
    assert again==f and f['raw_rows']==421
    assert sum(r['cutflow']['selected'] for r in f['files'])==400
    assert sum(r['cutflow']['quarantined206'] for r in f['files'])==5
    for a in arrays:
        assert np.all(a['roles'][a['jet_label']==206]==255)
        assert np.all(a['labels'][a['jet_label']==206]==-1)
    assert f['policy']['additional_pt_cut'] is None  # Fixture pT=199 remains eligible.
    assert all(sum(f['quotas'][role].values())==f['counts'][role] for role in c.ROLES)
    with pytest.raises(ValueError):p.load(root/'foundation')
    with pytest.raises(FileExistsError):p.build(root,root/'foundation',acknowledge_row_split=True,production=False)


def test_unknown_codes_fail_and_snapshot_domain_differs(dataset):
    root,f=dataset;_,arrays=p.load(root/'foundation',production=False)
    a={k:v.copy() for k,v in arrays[0].items() if k in c.SCALARS};a['jet_label'][0]=207
    with pytest.raises(ValueError,match='Unregistered raw label'):p.select(a,'signal')
    from hlt_classification.luka_fullstat.contracts import identity
    assert p.identity(f['files'][0],0)!=identity(f['files'][0],0)


def test_preparation_seals_test_and_reuses_exact_products(dataset,tmp_path,monkeypatch):
    root,f=dataset;_,members=p.load(root/'foundation',production=False)
    original=uproot.behaviors.TBranch.HasBranches.arrays;calls=[]
    lookup={r['path']:a for r,a in zip(f['files'],members)}
    def guarded(tree,expressions,*args,**kwargs):
        name=Path(tree.file.file_path).relative_to(root).as_posix()
        roles=lookup[name]['roles'][kwargs['entry_start']:kwargs['entry_stop']]
        assert not np.isin(roles,[2,3]).any()
        assert all(k=='jet_nparticles' or k.startswith('part_') for k in expressions)
        calls.append(name);return original(tree,expressions,*args,**kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches,'arrays',guarded)
    products=[prepare.prepare_file((root,root/'foundation',tmp_path,i,r,f['content_hash'])) for i,r in enumerate(f['files'])]
    assert calls and sum(b['rows'] for x in products for b in x['blocks'])==330
    assert prepare.prepare_file((root,root/'foundation',tmp_path,0,f['files'][0],f['content_hash']))==products[0]
    from hlt_classification.data.cache_contracts import write_immutable_json
    from hlt_classification.luka_fullstat.cache import conventions
    write_immutable_json(tmp_path/'cache.json',c.artifact('CACHE',parent=f['content_hash'],coordinate='OFFLINE',
        conventions=conventions(f['content_hash']),blocks=[b for x in products for b in x['blocks']]))
    tr=prepare.load(tmp_path,f,members,role='train',max_ram_bytes=2**30)
    assert len(tr)==264 and set(tr.labels)==set(range(11))
    with pytest.raises(PermissionError):prepare.load(tmp_path,f,members,role='final_test',max_ram_bytes=2**30)


def transfer_module():
    spec=importlib.util.spec_from_file_location('new_fullstat_transfer',Path(__file__).resolve().parents[1]/'scripts/transfer_luka_new_fullstat.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def stream_bundle(module,files):
    manifest=dict(contract='LUKA_NEW_FULLSTAT_TRANSFER/v1',files=[dict(path=k,bytes=len(v),sha256=hashlib.sha256(v).hexdigest())
                   for k,v in sorted(files.items())],bytes=sum(map(len,files.values())))
    raw=module.canonical(manifest);out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w') as tar:
        for name,data in [('transfer_manifest.json',raw),*sorted(files.items())]:
            info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
    out.seek(0);return hashlib.sha256(raw).hexdigest(),out


def test_transfer_stream_roundtrip_and_authenticated_retry(tmp_path):
    m=transfer_module();files={'raw/a.root':b'\x00\xffabc','foundation/f.json':b'{}'}
    sha,stream=stream_bundle(m,files);m.receive(tmp_path,sha,stream)
    for name,value in files.items():assert (tmp_path/name).read_bytes()==value
    sha,stream=stream_bundle(m,files);m.receive(tmp_path,sha,stream)
    (tmp_path/'raw/a.root').write_bytes(b'broken')
    sha,stream=stream_bundle(m,files)
    with pytest.raises(ValueError,match='not overwriting'):m.receive(tmp_path,sha,stream)


@pytest.mark.parametrize('name',['../escape','/absolute','x/../../escape','C:/escape','x\\escape'])
def test_transfer_unsafe_paths(tmp_path,name):
    m=transfer_module()
    with pytest.raises(ValueError):m.safe(tmp_path,name)


def test_transfer_explicit_identity_is_forwarded_without_reading(tmp_path,monkeypatch):
    m=transfer_module();key=tmp_path/'key with spaces';key.write_bytes(b'not a real key')
    def forbidden(*args,**kwargs):raise AssertionError('Private key contents must not be opened')
    monkeypatch.setattr(Path,'open',forbidden)
    args=m.ssh_command('rlyang@ssh.cs.brown.edu',key)
    assert args[args.index('-i')+1]==str(key.absolute())
    assert '-T' in args and args[-1]=='rlyang@ssh.cs.brown.edu'
    assert '-i' not in m.ssh_command('rlyang@ssh.cs.brown.edu')
    with pytest.raises(ValueError,match='identity file not found'):
        m.ssh_command('rlyang@ssh.cs.brown.edu',tmp_path/'missing')


@pytest.mark.parametrize('close_error',[BrokenPipeError(32,'Broken pipe'),OSError(22,'Invalid argument')])
def test_transfer_ssh_rejection_has_clear_error(tmp_path,monkeypatch,close_error):
    from types import SimpleNamespace
    m=transfer_module();key=tmp_path/'brown_key';key.write_bytes(b'placeholder')
    monkeypatch.setattr(m,'inventory',lambda root:({'bytes':0,'files':[]},'a'*64))
    class ClosedConnection:
        closed=False
        def write(self,data):raise BrokenPipeError(32,'Broken pipe')
        def close(self):self.closed=True;raise close_error
    process=SimpleNamespace(stdin=ClosedConnection(),wait=lambda:255)
    calls=[]
    def popen(args,**kwargs):calls.append(args);return process
    monkeypatch.setattr(m.subprocess,'Popen',popen)
    with pytest.raises(RuntimeError,match='SSH failed.*identity-file'):
        m.send(tmp_path,'rlyang@ssh.cs.brown.edu',
               '/cs/home/rlyang/hlt_u100_pairs/datasets/luka_qcdrun2com_higgs2p_sb_v1',identity_file=key)
    assert calls[0][calls[0].index('-i')+1]==str(key.absolute())


def test_original_twenty_epoch_schedule():
    from hlt_classification.luka_fullstat.recipe import recipe,learning_rate
    assert c.recipe()['base']==recipe(128) and c.recipe()['ce_weight']==1.
    assert learning_rate(2)==3e-4 and learning_rate(20)==1.5e-5
    assert c.node()['coordinate']=='OFFLINE' and c.node()['teacher'] is None


def toy_cache(role):
    from hlt_classification.jetclass2_delphes.cache import RamBlock,RamCache
    rng=np.random.default_rng(3);n=33
    b=RamBlock(0,np.arange(n+1,dtype=np.int64)*2,rng.normal(size=(n*2,17)).astype(np.float32),
        np.tile(np.array([1,0,0,2],np.float32),(n*2,1)),np.arange(n*32,dtype=np.uint8).reshape(n,32),
        np.tile(np.arange(11,dtype=np.int64),3))
    return RamCache([b],role=role,foundation_sha256='a'*64,coordinate_name='OFFLINE')


def toy_model():
    import torch
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__();self.dropout=torch.nn.Dropout(.2);self.linear=torch.nn.Linear(17,11)
        def forward(self,features,vectors,mask):
            return self.linear(self.dropout((features*mask).sum(-1)/mask.sum(-1)))
        def no_weight_decay(self):return set()
    torch.manual_seed(c.node()['initialization_seed']);return Net()


def test_checkpoint_continuation_reproduces_uninterrupted_updates(tmp_path):
    import torch
    from hlt_classification.luka_new_fullstat_hydra import training as t
    torch.set_num_threads(1);tr,val=toy_cache('train'),toy_cache('validation')
    one,h1,b1=t.run(tr,val,tmp_path/'one','1',spec_hash='a'*64,device='cpu',model_factory=toy_model,epochs=3,batch=16)
    t.run(tr,val,tmp_path/'resumed','2',spec_hash='a'*64,device='cpu',model_factory=toy_model,epochs=1,batch=16)
    two,h2,b2=t.run(tr,val,tmp_path/'resumed','3',spec_hash='a'*64,device='cpu',model_factory=toy_model,epochs=3,batch=16)
    assert [x['validation'] for x in h1]==[x['validation'] for x in h2]
    assert b1['pass_number']==b2['pass_number']
    for key,value in one.state_dict().items():torch.testing.assert_close(value,two.state_dict()[key],rtol=0,atol=0)
    state_path=next((tmp_path/'resumed/epochs').glob('*/state.pt'));state_path.write_bytes(b'broken')
    with pytest.raises(ValueError):t.committed(tmp_path/'resumed','a'*64,3)


def test_training_matches_original_ce_kernel(tmp_path):
    import torch
    from hlt_classification.jetclass2_delphes.runner import train_kernel
    from hlt_classification.luka_new_fullstat_hydra import training as t
    torch.set_num_threads(1);tr,val=toy_cache('train'),toy_cache('validation')
    report,state=train_kernel(toy_model(),tr,val,node=c.node(),device='cpu',
        acceptance_passes=2,training_recipe=c.recipe()['base'])
    net,history,best=t.run(tr,val,tmp_path,'4',spec_hash='a'*64,device='cpu',model_factory=toy_model,epochs=2)
    assert [r['validation'] for r in report['validation_history']]==[r['validation'] for r in history]
    assert report['selected_pass']==best['pass_number']
    for k,v in state.items():torch.testing.assert_close(v,net.state_dict()[k],atol=0,rtol=0)


def test_rejection50_matches_original_and_ties():
    from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
    from hlt_classification.luka_new_fullstat_hydra.training import rejection
    labels=np.tile(np.arange(11),10);prob=np.full((110,11),1/11,np.float32)
    old=evaluate_probabilities(labels,prob);new=rejection(labels,prob,.5)
    for name,row in new.items():
        assert row['rejection']==old['per_class'][name]['rejection_at_50pct']==1.
        assert row['background_passing']==10 and row['achieved_signal_efficiency']==1.
    assert rejection(labels,prob,.3)==new


def test_submission_plan_is_single_offline_fit():
    from hlt_classification.luka_new_fullstat_hydra.campaign import commands
    s=dict(root='/r',project='/p',content_hash='a'*64,prepare_cpus=4,cpus=6,prepare_memory_mb=16000,
           memory_mb=64000,prepare_minutes=1440,minutes=2880,gpu='l40s')
    plan=commands(s)
    assert [r['task'] for r in plan]==['prepare','train']
    assert '--gres=gpu:nvidia_l40s:1' in plan[1]['command']
    assert '--dependency=afterok:{prepare}' in plan[1]['command']
    assert '--partition=compute' in plan[0]['command']
    assert [r['task'] for r in commands(s,resume=True)]==['train']


def test_submission_binds_dependencies_and_blocks_duplicate_attempts(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from hlt_classification.luka_new_fullstat_hydra import campaign
    s=dict(root=str(tmp_path),project='/p',content_hash='a'*64,prepare_cpus=4,cpus=6,prepare_memory_mb=16000,
           memory_mb=40000,prepare_minutes=1440,minutes=2880,gpu='l40s')
    monkeypatch.setattr(campaign,'check',lambda s:None);monkeypatch.setattr(campaign,'site',lambda p:None)
    calls=[]
    def submit(args,**kwargs):
        task=args[-1]
        assert list((tmp_path/'submission').glob(f'attempt_*/{task}.intent.json'))
        calls.append(args)
        return SimpleNamespace(stdout=str(100+len(calls))+'\n',stderr='',returncode=0)
    monkeypatch.setattr(campaign.subprocess,'run',submit)
    plan=campaign.queue(s);assert not calls and not (tmp_path/'submission').exists()
    jobs=campaign.queue(s,execute=True,authorization=campaign.AUTH)
    assert jobs==dict(prepare='101',train='102')
    assert '--dependency=afterok:101' in calls[1]
    with pytest.raises(FileExistsError):campaign.queue(s,execute=True,authorization=campaign.AUTH)
    assert len(calls)==2


def test_submission_ambiguity_preserves_intent(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from hlt_classification.luka_new_fullstat_hydra import campaign
    s=dict(root=str(tmp_path),project='/p',content_hash='a'*64,prepare_cpus=4,cpus=6,prepare_memory_mb=16000,
           memory_mb=40000,prepare_minutes=1440,minutes=2880,gpu='l40s')
    monkeypatch.setattr(campaign,'check',lambda s:None);monkeypatch.setattr(campaign,'site',lambda p:None)
    monkeypatch.setattr(campaign.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='not a job ID',stderr='',returncode=0))
    with pytest.raises(RuntimeError,match='Ambiguous'):campaign.queue(s,execute=True,authorization=campaign.AUTH)
    assert (tmp_path/'submission/attempt_001/prepare.intent.json').is_file()
    assert not list((tmp_path/'submission').glob('**/*.submitted.json'))
    with pytest.raises(FileExistsError):campaign.queue(s,execute=True,authorization=campaign.AUTH)


def test_transfer_wrong_digest_and_foreign_directory(tmp_path):
    m=transfer_module();sha,stream=stream_bundle(m,{'raw/file':b'hello'})
    with pytest.raises(ValueError,match='identity differs'):m.receive(tmp_path/'wrong', '0'*64,stream)
    target=tmp_path/'occupied';target.mkdir();(target/'user_file').write_text('keep')
    sha,stream=stream_bundle(m,{'raw/file':b'hello'})
    with pytest.raises(FileExistsError):m.receive(target,sha,stream)
    assert (target/'user_file').read_text()=='keep'


def test_allocation_rejects_wrong_gpu_and_memory(tmp_path,monkeypatch):
    import sys,torch
    from types import SimpleNamespace
    from hlt_classification.luka_new_fullstat_hydra import worker,campaign
    s=dict(root=str(tmp_path),cpus=6,memory_mb=40000,environment_dir=sys.prefix,environment={'v':1},gpu='l40s')
    monkeypatch.setenv('SLURM_JOB_ID','101');monkeypatch.setenv('PYTHONNOUSERSITE','1')
    monkeypatch.setenv('SLURM_NTASKS','1');monkeypatch.setenv('SLURM_CPUS_PER_TASK','6')
    monkeypatch.setattr(campaign,'job_binding',lambda *a:None);monkeypatch.setattr(campaign,'runtime',lambda:{'v':1})
    fields='JobId=101 JobState=RUNNING Account=default QOS=normal Partition=gpus NumCPUs=6 NumNodes=1 NumTasks=N/A CPUs/Task=N/A AllocTRES=mem=40000M'
    def run(args,**kw):return SimpleNamespace(stdout='ClusterName = hydra' if args[-1]=='config' else fields)
    monkeypatch.setattr(worker.subprocess,'run',run)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True);monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True);monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'NVIDIA L40S')
    assert worker.allocation(s,'train')=='101'
    s['memory_mb']=32000
    with pytest.raises(PermissionError,match='Memory'):worker.allocation(s,'train')
    s['memory_mb']=40000;s['gpu']='a6000'
    with pytest.raises(PermissionError,match='GPU model'):worker.allocation(s,'train')
