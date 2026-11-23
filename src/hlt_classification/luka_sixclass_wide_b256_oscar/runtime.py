"""Persistent runtime bootstrap, without submitting the old eleven-class run."""
import math
import os
from pathlib import Path
from hlt_classification.operations import oscar_runtime_restore as repair
from .contracts import RESULT_GIB


def ensure(spec_path, project, conda_exe, available_gib):
    """Use healthy base Python when absent; verify exact bytes when already built."""
    target = repair.TARGET
    records = Path(str(target)+'.restore')
    existed = os.path.lexists(target) or os.path.lexists(records)
    if not existed:
        repair.require(bool(conda_exe), 'A healthy base Conda executable is required to build HOME runtime')
        repair.restore(spec_path, project, conda_exe, available_gib)
    complete = repair.read_hashed(records/'complete.json')
    plan = repair.read_hashed(records/'plan.json')
    expected = repair.read_hashed(spec_path)['environment']
    repair.require(complete.get('contract') == 'LUKA_OSCAR_RUNTIME_RESTORED/v1'
        and complete.get('environment_dir') == str(target)
        and complete.get('parents') == dict(plan=plan['content_hash'], environment=expected['content_hash'])
        and plan['expected_environment'] == expected,
        'HOME runtime completion/plan/fingerprint differs; inspect existing restoration records')
    if existed:
        repair.verify_files(target, plan)
        env = dict(os.environ)
        for key in ('PYTHONHOME','PYTHONPATH','LD_PRELOAD'):
            env.pop(key, None)
        env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(target/'lib'),
                   OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        repair.verify_runtime(target, project, spec_path, env)
    # The caller's quota is current free space, not the old restoration quota.
    remaining = available_gib if existed else available_gib-math.ceil(complete['installed_bytes']/repair.GIB)-2
    repair.require(remaining >= RESULT_GIB, f'Need {RESULT_GIB} GiB current HOME headroom for campaign results')
    return remaining


def previous_terminal(spec_path):
    """An explicitly named old campaign must be terminal; never cancel anything."""
    from hlt_classification.luka_new_balanced_oscar import campaign as old
    spec = repair.read_hashed(spec_path)
    repair.require(spec['contract'] == 'LUKA_NEW_BALANCED_OSCAR_SPEC/v1'
                   and Path(spec['root']) == Path(spec_path).parent, 'Wrong previous campaign')
    jobs = old.accepted_jobs(spec)
    repair.require(bool(jobs), 'Previous submission records are missing')
    return dict(contract='LUKA_SIXCLASS_WIDE_B256_OSCAR_PRIOR_TERMINAL/v1',schema_version=1,
                parents=dict(previous_spec=spec['content_hash']),
                jobs={row['job']:old.terminal(row['job']) for row in jobs},jobs_changed=False)
