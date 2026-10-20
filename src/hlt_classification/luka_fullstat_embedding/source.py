"""Pinned original FullStat replay; no dependency on attention science."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_fullstat_attention.source import authenticate as parent_authenticate
from hlt_classification.jetclass2_delphes.contracts import validate as kernel_validate
from .contracts import artifact, TEACHER

DONOR = '5ed50964085e9bf364030764bbbc755179c445ae'
MODULES = ('__init__', 'contracts', 'source', 'banks', 'training', 'checkpoints',
           'campaign', 'worker', 'preflight', 'submission')
RUNTIME = {f'src/hlt_classification/luka_fullstat_embedding/{n}.py' for n in MODULES} | {
    'scripts/luka_fullstat_embedding.py', 'scripts/queue_luka_fullstat_embedding.sh',
    'sbatch/run_luka_fullstat_embedding.sh'}
REQUIRED = RUNTIME | {'tests/test_luka_fullstat_embedding.py',
    'docs/plans/LUKA_FULLSTAT_LAST_EMBEDDING_PLAN.md',
    'docs/contracts/LUKA_FULLSTAT_LAST_EMBEDDING.md', 'docs/LUKA_FULLSTAT_LAST_EMBEDDING_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True,
                              capture_output=True, text=True).stdout.splitlines()
    if (not REQUIRED <= set(git('ls-files'))
            or not set(git('diff', '--name-only', DONOR, commit, '--', 'src', 'scripts', 'sbatch')) <= RUNTIME
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_fullstat_embedding/source.py'):
        raise ValueError('Commit complete additive embedding source; existing runtime must remain frozen')
    return snapshot


def authenticate(path, project):
    original, parent = parent_authenticate(path, project)
    root = Path(parent['root'])/('reduce_'+TEACHER)/'train'
    manifest = load_json(root/'manifest.json')
    report = load_json(validate_file_ref(original['evidence']['models'][TEACHER]['report']))
    kernel_validate(manifest, 'PROBABILITY_BANK')
    expected = dict(foundation_sha256=parent['prepared_sha256'], teacher_node=TEACHER,
        teacher_report_sha256=report['content_hash'], role='train', temperature=2.,
        rows=parent['counts']['train'], class_count=11, dtype='float32', final_test_accessed=False)
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Original D033 TRAIN probability bank differs')
    # Original parent replay has verified the reducer receipt and every shard.
    return artifact('ANCHOR', parents=dict(original=original['content_hash']),
        original=original, spec=original['spec'], evidence=original['evidence'],
        logits=file_ref(root/'manifest.json')), parent
