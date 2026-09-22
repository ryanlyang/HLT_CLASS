"""K2-only scheduling portability; other campaigns retain strict site checks."""
from __future__ import annotations

import os

from .contracts import artifact
from .execution import execution_site

PARTITIONS = {"tier3": "sporc_a100", "debug": "sporc_a100_debug"}


def site_for_partition(partition):
    if partition not in PARTITIONS:
        raise ValueError("K2 partition must be exactly tier3 or debug")
    return execution_site(PARTITIONS[partition])


def execution_policy():
    return artifact("CONCAT_K2_EXECUTION_POLICY",
        allowed_sites=[site_for_partition(p) for p in PARTITIONS],
        mutable_scheduler_fields=["Partition"], pending_jobs_only=True,
        maximum_time_minutes=1440, unchanged_resources_required=True,
        identical_gpu_and_environment_required=True,
        original_submission_ledger_is_immutable=True)


def admit_site(spec, partition):
    """Canonical actual site, independent of the initial submission partition."""
    registration = spec.get("registration", spec)
    requested = registration["execution_site"]
    if (registration.get("execution_policy") != execution_policy()
            or requested != site_for_partition(requested["partition"])):
        raise ValueError("K2 execution policy or requested site differs")
    actual = site_for_partition(partition)
    scheduling = {"name", "partition", "content_hash"}
    if ({k: v for k, v in actual.items() if k not in scheduling}
            != {k: v for k, v in requested.items() if k not in scheduling}):
        raise ValueError("K2 partition transfer changes hardware/environment contract")
    return actual


def runtime_site(spec):
    # allocation() independently compares this environment with scontrol.
    return admit_site(spec, os.environ.get("SLURM_JOB_PARTITION", ""))


def validate_acceptance_site(spec, acceptance):
    actual = acceptance["site"]
    if (actual != admit_site(spec, actual["partition"])
            or acceptance.get("requested_site") != spec["execution_site"]
            or acceptance.get("execution_policy_sha256") != spec["execution_policy"]["content_hash"]):
        raise ValueError("K2 acceptance execution policy/site differs")
