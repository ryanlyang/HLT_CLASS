"""K2-only scheduling portability; other campaigns retain strict site checks."""
from __future__ import annotations

import math
from numbers import Real
import os

from .contracts import artifact
from .execution import execution_site

PARTITIONS = {"tier3": "sporc_a100", "debug": "sporc_a100_debug"}
PORTABLE_MINUTES = 1440
TRAINING_MINUTES = 5760
WALLTIME_RESERVE_MINUTES = 60


def site_for_partition(partition):
    if partition not in PARTITIONS:
        raise ValueError("K2 partition must be exactly tier3 or debug")
    return execution_site(PARTITIONS[partition])


def execution_policy():
    return artifact("CONCAT_K2_EXECUTION_POLICY", version=2,
        allowed_sites=[site_for_partition(p) for p in PARTITIONS],
        mutable_scheduler_fields=["Partition"], pending_jobs_only=True,
        maximum_time_minutes=TRAINING_MINUTES,
        portable_maximum_time_minutes=PORTABLE_MINUTES,
        long_job_partition="tier3", unchanged_resources_required=True,
        identical_gpu_and_environment_required=True,
        original_submission_ledger_is_immutable=True)


def admit_site(spec, partition, *, resource=None):
    """Canonical actual site, independent of the initial submission partition."""
    registration = spec.get("registration", spec)
    requested = registration["execution_site"]
    if (registration.get("execution_policy") != execution_policy()
            or requested != site_for_partition(requested["partition"])):
        raise ValueError("K2 execution policy or requested site differs")
    actual = site_for_partition(partition)
    if resource is not None:
        minutes = resource["minutes"]
        if (type(minutes) is not int or not 0 < minutes <= TRAINING_MINUTES
                or (partition == "debug" and minutes > PORTABLE_MINUTES)):
            raise ValueError("K2 long-walltime jobs require tier3; debug is limited to 24h")
    scheduling = {"name", "partition", "content_hash"}
    if ({k: v for k, v in actual.items() if k not in scheduling}
            != {k: v for k, v in requested.items() if k not in scheduling}):
        raise ValueError("K2 partition transfer changes hardware/environment contract")
    return actual


def submission_site(spec, resource):
    """Long jobs never inherit a short launcher's requested/actual debug site."""
    registered = spec.get("registration", spec)
    partition = ("tier3" if resource["minutes"] > PORTABLE_MINUTES
                 else registered["execution_site"]["partition"])
    return admit_site(spec, partition, resource=resource)


def runtime_projection_limit_seconds(spec):
    registered = spec.get("registration", spec)
    if (registered["resources"]["train"]["minutes"] != TRAINING_MINUTES
            or registered.get("runtime_walltime_reserve_minutes") != WALLTIME_RESERVE_MINUTES):
        raise ValueError("K2 runtime acceptance budget differs from registration")
    return (TRAINING_MINUTES - WALLTIME_RESERVE_MINUTES) * 60


def validate_runtime_projection(spec, seconds):
    limit = runtime_projection_limit_seconds(spec)
    if (isinstance(seconds, bool) or not isinstance(seconds, Real)
            or not math.isfinite(seconds) or not 0 < seconds <= limit):
        raise ValueError(f"K2 runtime acceptance requires a finite positive projected fit <= {limit/3600:g}h; "
                         f"measured projection seconds={seconds!r}")


def runtime_site(spec):
    # allocation() independently compares this environment with scontrol.
    return admit_site(spec, os.environ.get("SLURM_JOB_PARTITION", ""))


def validate_acceptance_site(spec, acceptance):
    actual = acceptance["site"]
    if (actual != admit_site(spec, actual["partition"], resource=spec["resources"]["preflight"])
            or acceptance.get("requested_site") != spec["execution_site"]
            or acceptance.get("execution_policy_sha256") != spec["execution_policy"]["content_hash"]):
        raise ValueError("K2 acceptance execution policy/site differs")
