"""Linux /proc races simulated without hiding actual monitoring errors."""
from pathlib import Path
import errno

import pytest

from hlt_classification.cms2jc2_response import measurement as m


@pytest.fixture
def proc(monkeypatch):
    monkeypatch.setattr(m.platform,'system',lambda:'Linux')
    monkeypatch.setattr(m.os,'getpid',lambda:100)
    monkeypatch.setattr(m.os,'sysconf',lambda k:100 if k=='SC_CLK_TCK' else 4096,raising=False)
    errors={}
    def read(path,*a,**kw):
        name=path.as_posix()
        if name in errors: raise errors[name]
        if name.endswith('/stat'):
            fields=['0']*22
            fields[11]='20'; fields[12]='10'; fields[19]='123'; fields[21]='7'
            return 'pid (name with spaces) '+ ' '.join(fields)
        return '200' if name=='/proc/100/task/101/children' else ''
    def dirs(path):
        name=path.as_posix()
        if name in errors: raise errors[name]
        return iter([Path(name)/'100',Path(name)/'101']) if name=='/proc/100/task' else iter([Path(name)/'200'])
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setattr(Path,'iterdir',dirs)
    return m.Measurement(),errors


@pytest.mark.parametrize('error',[FileNotFoundError(errno.ENOENT,'gone'),ProcessLookupError(errno.ESRCH,'gone')])
@pytest.mark.parametrize('where',['/proc/200/stat','/proc/200/task','/proc/200/task/200/children','/proc/100/task/100/children'])
def test_disappearing_children_or_threads_keep_sampling(proc,error,where):
    sampler,errors=proc
    errors[where]=error
    sampler.sample()
    assert sampler.samples==1 and sampler.peak>=7*4096
    errors.clear()
    sampler.sample()
    assert sampler.samples==2 and sampler.peak==14*4096
    assert len(sampler.cpu)==2 and sampler.error is None


@pytest.mark.parametrize('error',[ProcessLookupError(errno.ESRCH,'gone'),FileNotFoundError(errno.ENOENT,'gone'),PermissionError('denied'),OSError(errno.EIO,'io')])
def test_root_process_must_be_measurable(proc,error):
    sampler,errors=proc
    errors['/proc/100/stat']=error
    with pytest.raises(type(error)): sampler.sample()


@pytest.mark.parametrize('error',[PermissionError('denied'),ValueError('malformed'),OSError(errno.EIO,'io')])
def test_genuine_child_monitoring_errors_remain_fatal(proc,error):
    sampler,errors=proc
    errors['/proc/200/stat']=error
    with pytest.raises(type(error)): sampler.sample()


def test_monitor_thread_survives_exit_race_but_reports_real_failure(proc,monkeypatch):
    sampler,errors=proc
    steps=iter([False,False,True])
    monkeypatch.setattr(sampler.stop,'wait',lambda _:next(steps))
    errors['/proc/200/stat']=ProcessLookupError(errno.ESRCH,'gone')
    sampler._loop()
    assert sampler.samples==2 and sampler.error is None
    monkeypatch.setattr(sampler.stop,'wait',lambda _:False)
    errors['/proc/200/stat']=PermissionError('denied')
    sampler._loop()
    assert isinstance(sampler.error,PermissionError)
    errors.clear()
    sampler.thread=type('Thread',(),{'join':lambda self:None})()
    sampler.started=0
    with pytest.raises(PermissionError): sampler.__exit__(None,None,None)
