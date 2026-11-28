"""Standalone authenticated metadata; importing this module loads no science."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

PREFIX='LUKA_D20_INPUT_RECOVERY'
PARENT_COMMIT='5af66449d2b56ae37b01a642c7246027db94b097'
BASE=Path('/cs/home/rlyang/hlt_u100_pairs')
PARENT=BASE/'runs/d20_full_5af66449_r1'
AUTH='AUTHORIZE_D20_TARGETED_INPUT_RECOVERY_V1'
BAD_FILE='raw/qcd/jetclass2sb/QCD_PT-2400to3200/261028_082640/0000/ntuple_382.root'
FLAGS=dict(final_test_accessed=False,final_test_sealed=True,evaluation_scope='development_validation')


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def artifact(kind,*,parents=None,**fields):
    value=dict(contract=PREFIX+'_'+kind+'/v1',schema_version=1,parents=dict(parents or {}),**FLAGS,**fields)
    return dict(value,content_hash=hashlib.sha256(canonical(value)).hexdigest())


def read(path):
    value=json.loads(Path(path).read_bytes());body={k:v for k,v in value.items() if k!='content_hash'}
    require(value.get('content_hash')==hashlib.sha256(canonical(body)).hexdigest(),'Content hash differs: '+str(path))
    return value


def validate(value,kind):
    require(value.get('contract')==PREFIX+'_'+kind+'/v1' and value.get('schema_version')==1
        and all(value.get(k)==v for k,v in FLAGS.items()),'Unexpected repair schema/scope')
    require(value.get('content_hash')==hashlib.sha256(canonical({k:v for k,v in value.items() if k!='content_hash'})).hexdigest(),
        'Repair content hash differs')
    return value


def ref(path):
    path=Path(path).resolve(strict=True)
    return dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size)


def referenced(row):
    path=Path(row['path'])
    require(path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes']
        and digest(path)==row['sha256'],'Referenced payload differs: '+str(path))
    return path


def bootstrap_receipt(spec,phase):
    folder=Path(spec['root'])/'submission/bootstrap'
    row=validate(read(folder/(phase+'.submitted.json')),'SUBMITTED')
    intent=validate(read(folder/(phase+'.intent.json')),'SUBMISSION_INTENT')
    plan=validate(read(folder/'plan.json'),'PLAN')
    expected=dict(repair=spec['content_hash'])
    require(row['parents']==intent['parents']==plan['parents']==expected
        and row['task']==intent['task']==phase and row['plan']==intent['plan']==plan['content_hash']
        and row['execution_recovery']==intent['execution_recovery']==plan['execution_recovery']==spec['content_hash']
        and row['execution_spec']==intent['execution_spec']==ref(Path(spec['root'])/'repair_spec.json')
        and row['command']==intent['command'],'Bootstrap submission provenance differs')
    choices=[r for r in plan['commands'] if r['task']==phase]
    require(len(choices)==1,'Bootstrap phase absent or duplicated in plan')
    command=choices[0]['command']
    if phase=='release':
        probe=bootstrap_receipt(spec,'probe')
        command=[v.replace('{probe}',probe['job']) for v in command]
    require(row['command']==command,'Bootstrap accepted command differs from plan')
    return row


def publish(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
        os.link(name,path)
    finally:Path(name).unlink(missing_ok=True)


def policy():
    return artifact('POLICY',reader='uproot.MultithreadedFileSource',reader_threads=False,reader_handles=1,
        temporary_cache_bytes=128*2**20,logical_spans='original_authorized_role_spans',
        retry='fresh_owned_pool_replay_unaccepted_batches_in_original_order',max_pool_restarts=2,
        task_timeout_seconds=600,cleanup_timeout_seconds=15,retry_errno=[5,110,116,104,103,113,100,101],
        retry_checksum='worker_before_each_replayed_batch',source_final_checksum='original_unchanged',
        invalid_inputs='fail_without_retry',science_source='original_immutable_parent',
        qualification='reuse_only_after_actual_reader_equivalence_probe',
        running_jobs='preserve',completed_segments='preserve',
        cancel='exact_bound_pending_descendants_only_with_state_filter',
        restart_state='original_authenticated_last_committed_epoch',persistent_particle_cache=False)
