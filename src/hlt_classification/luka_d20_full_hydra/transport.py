"""Campaign-local GPU transport selected by the bounded Hydra diagnostic."""
import os
from . import contracts as c

ENVIRONMENT={'NCCL_P2P_DISABLE':'1'}


def policy():
    return c.artifact('GPU_TRANSPORT_POLICY',control_backend='gloo',tensor_backend='nccl',
        environment=ENVIRONMENT,automatic_fallback=False,
        evidence=dict(kind='operator_supplied_diagnostic_status_and_accounting',
            helper_sha256='4a19da6c85172343141bbfbc43fa494bad59482439be6edee2f8435c56bf0466',
            source_commit='a2b634c7a4406f24f6a11422f24fdb76b5a1c7ca',node='gpu1905',
            original_job='219050',gloo_job='219051',passing_job='219052',
            passing_profile='gloo_no_p2p',state='COMPLETED',exit_code='0:0',elapsed='00:01:27',
            scope='known_tensor_collectives_and_OFFLINE_s10_synthetic_four_rank_parity'),
        admission='actual_per_allocation_early_GPU_checks_before_decoding')


def check():
    if any(os.environ.get(k)!=v for k,v in ENVIRONMENT.items()):
        raise PermissionError('Use the qualified Hydra transport: NCCL_P2P_DISABLE=1 before GPU initialization')
