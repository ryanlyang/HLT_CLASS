"""96h tier3 fits; unchanged short-job portability and finite 95h acceptance."""
import pytest

from hlt_classification.jetclass2_delphes import (
    concat_k2_campaign as campaign, concat_k2_execution as execution,
    concat_k2_submit as scheduler,
)
from test_jetclass2_concat_k2 import spec_at
from test_jetclass2_concat_k2_execution import worker
from test_jetclass2_dzfix_fusion_chain import rehash


@pytest.mark.parametrize("seconds", [1., 24*3600, 262478.2203352421, 95*3600])
def test_runtime_accepts_measured_73h_with_one_hour_reserved(seconds):
    spec = campaign.registration()
    assert execution.runtime_projection_limit_seconds(spec) == 95*3600
    assert spec["runtime_projection_margin"] == 1.30
    assert spec["gpu_peak_fraction_limit"] == .90
    assert spec["cpu_peak_fraction_limit"] == .80
    execution.validate_runtime_projection(spec, seconds)


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), True, "100", None,
                                     95*3600+.01, 96*3600])
def test_runtime_rejects_nonfinite_invalid_and_insufficient_headroom(seconds):
    with pytest.raises(ValueError, match="runtime acceptance"):
        execution.validate_runtime_projection(campaign.registration(), seconds)


@pytest.mark.parametrize("fault", ["reserve", "walltime", "policy"])
def test_rehashed_changes_cannot_silently_relax_runtime_policy(tmp_path, fault):
    spec = spec_at(tmp_path)
    if fault == "reserve":
        spec["runtime_walltime_reserve_minutes"] = 0
    elif fault == "walltime":
        spec["resources"]["train"]["minutes"] = 7200
    else:
        spec["execution_policy"]["portable_maximum_time_minutes"] = 5760
        spec["execution_policy"] = rehash(spec["execution_policy"])
    with pytest.raises(ValueError, match="registration differs"):
        campaign.validate_campaign(rehash(spec), check_source=False)


@pytest.mark.parametrize("requested", ["tier3", "debug"])
def test_all_ten_fits_request_tier3_96h_even_when_launcher_uses_debug(tmp_path, requested):
    spec = rehash(spec_at(tmp_path), **campaign.registration(requested))
    # Parent-process scheduling cannot change canonical downstream commands.
    full = scheduler.plan(spec, "full")
    fits = [r for r in full["commands"] if r["task_id"].startswith("train_")]
    assert len(fits) == 10
    for row in full["commands"]:
        command = row["command"]
        fit = row in fits
        assert "--partition=" + ("tier3" if fit else requested) in command
        assert "--qos=qos_tier3" in command
        if fit:
            for flag in ("--time=5760", "--cpus-per-task=4", "--mem=320000M", "--gres=gpu:a100:1"):
                assert flag in command
        else:
            assert int(next(t.split("=", 1)[1] for t in command if t.startswith("--time="))) <= 1440
    assert spec["training"]["batch_size"] == spec["inference_batch_size"] == 128


@pytest.mark.parametrize("limit", ["4-00:00:00", "96:00:00"])
def test_authentication_accepts_slurm_day_or_hour_96h_format(tmp_path, monkeypatch, limit):
    name = "train_CONCAT_K2_D100"
    spec, fields, _, _, _ = worker(tmp_path, monkeypatch, "debug", "tier3", name)
    fields["TimeLimit"] = limit
    assert scheduler.authenticate_job(spec, name) == "456"


@pytest.mark.parametrize("limit", ["1-00:00:00", "3-23:00:00", "5-00:00:00", "UNLIMITED"])
def test_shortening_or_extending_a_registered_fit_fails_authentication(tmp_path, monkeypatch, limit):
    name = "train_CONCAT_K2_D100"
    spec, fields, _, root, _ = worker(tmp_path, monkeypatch, "tier3", "tier3", name)
    fields["TimeLimit"] = limit
    with pytest.raises(PermissionError):
        scheduler.authenticate_job(spec, name)
    assert not (root / "execution").exists()


def test_old_execution_policy_cannot_authorize_long_jobs():
    spec = campaign.registration()
    old = spec["execution_policy"]
    spec["execution_policy"] = rehash(old, schema_version=1,
        contract=old["contract"].replace("/v2", "/v1"))
    with pytest.raises(ValueError, match="policy"):
        execution.submission_site(spec, spec["resources"]["train"])
