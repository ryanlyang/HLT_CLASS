"""Operational recovery tests; CPU/mocked CUDA checks are not site admission."""
from copy import deepcopy
from pathlib import Path
import subprocess

import numpy as np
import pytest
import uproot

from test_luka_fullsim_stage2 import dataset, prepared, bind
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.provenance import capture_source_snapshot
from hlt_classification.luka_fullsim import preflight_cache as cache, preflight_reuse as reuse, preflight_v2 as gpu
from hlt_classification.luka_fullsim.stage2_cache import prepare_cache
from hlt_classification.context_fusion.worker import largest_raw


@pytest.mark.parametrize("role", ["train", "validation"])
def test_bundle_exact_v1_replay_single_reader(prepared, monkeypatch, role):
    root, p = bind(prepared, monkeypatch)
    calls = []
    reader = cache.ParticleReader
    def counted(*args, **kwargs):
        calls.append(kwargs)
        return reader(*args, **kwargs)
    monkeypatch.setattr(cache, "ParticleReader", counted)
    bundle = cache.build_caches(root, role=role, max_ram_bytes=2**30)
    assert len(calls) == 1
    for c in cache.COORDINATES:
        original = prepare_cache(root, role=role, coordinate=c, max_ram_bytes=2**30)
        assert bundle[c].nbytes == original.nbytes
        for key, value in original.batch(np.arange(len(original))).items():
            assert np.array_equal(value, bundle[c].batch(np.arange(len(original)))[key])
    paired = gpu._view(bundle, gpu._node(3))
    assert paired.primary is paired.context is bundle["D000"]


def test_bundle_d000_never_requests_offline_and_budget_early(prepared, monkeypatch):
    root, _ = bind(prepared, monkeypatch)
    original = uproot.behaviors.TBranch.HasBranches.arrays
    calls = []
    def guarded(tree, expressions, *args, **kwargs):
        assert not any(s.startswith("part_") or s == "jet_nparticles" for s in expressions)
        calls.append(expressions)
        return original(tree, expressions, *args, **kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, "arrays", guarded)
    with pytest.raises(MemoryError):
        cache.build_caches(root, role="train", max_ram_bytes=1)
    assert not calls
    cache.build_caches(root, role="train", coordinates=("D000",), max_ram_bytes=2**30)
    assert calls
    with pytest.raises(PermissionError):
        cache.build_caches("absent", role="final_test", max_ram_bytes=2**30)


def test_small_witnesses_exact_longest_and_first_four(prepared, monkeypatch):
    root, p = bind(prepared, monkeypatch)
    _, _, _, inventory, splits = prepared
    allowed = {inventory["files"][m["file_index"]]["path"]: set(sum(m["entries_by_class"], []))
               for m in splits["memberships"]["train"]}
    original = uproot.behaviors.TBranch.HasBranches.arrays
    def guarded(tree, expressions, *args, **kwargs):
        path = Path(tree.file.file_path).as_posix().split("/raw/")[1]
        assert path in allowed  # not even scalar access to validation/test files
        if any("part_" in s for s in expressions):
            assert set(range(kwargs["entry_start"], kwargs["entry_stop"])) <= allowed[path]
        return original(tree, expressions, *args, **kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, "arrays", guarded)
    small, report = cache.probe_caches(root)
    bundle = cache.build_caches(root, role="train", max_ram_bytes=2**30)
    assert report["retained_training_rows"] < len(bundle["U000"])
    assert not report["validation_particles_accessed"] and not report["final_test_accessed"]
    for c in cache.COORDINATES:
        assert np.array_equal(small[c].identities[:4], bundle[c].identities[:4])
        assert report["witnesses"][c]["particles"] == p["summaries"]["train"][c]["maximum"]
        for key, value in largest_raw(bundle[c]).items():
            assert np.array_equal(value, largest_raw(small[c])[key])


def test_probe_source_tamper_and_missing_maximum(prepared, monkeypatch):
    root, p = bind(prepared, monkeypatch)
    with monkeypatch.context() as m:
        m.setattr(cache, "sha256_file", lambda _: "0"*64)
        with pytest.raises(ValueError, match="checksum"):
            cache.probe_caches(root)
    bad = deepcopy(p)
    bad["summaries"]["train"]["U050"]["maximum"] = 511
    mdata = (bad, *cache.s.load_prepared(root)[1:])
    monkeypatch.setattr(cache.s, "load_prepared", lambda _: mdata)
    with pytest.raises(ValueError, match="Missing.*witness"):
        cache.probe_caches(root)


def test_bundle_fingerprint_mismatch_rejected(prepared, monkeypatch):
    root, p = bind(prepared, monkeypatch)
    bad = deepcopy(p)
    bad["summaries"]["train"]["U050"]["input_sha256"] = "0"*64
    mdata = (bad, *cache.s.load_prepared(root)[1:])
    monkeypatch.setattr(cache.s, "load_prepared", lambda _: mdata)
    with pytest.raises(ValueError, match="input replay"):
        cache.build_caches(root, role="train", max_ram_bytes=2**30)


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _commit(root):
    _git(root, "add", ".")
    _git(root, "-c", "user.name=Fixture", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    return capture_source_snapshot(root, require_clean=True)


def test_dual_source_all_old_scientific_bytes_still_required(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    _git(old, "init")
    _git(old, "config", "core.autocrlf", "false")
    (old / "src").mkdir()
    (old / "src/model.py").write_bytes(b"# unchanged science\n")
    source = _commit(old)
    _git(tmp_path, "-c", "core.autocrlf=false", "clone", str(old), str(new))
    for name in reuse.ADDITIONS:
        path = new / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# registered runtime addition\n")
    active = _commit(new)
    p = dict(content_hash="a"*64, source_snapshot=source)
    r = reuse.authenticate(p, active, project=new, prepared_project=old)
    assert r["parents"]["preparation_source"] != r["parents"]["execution_source"]
    assert r["identical_source_files"] and r["added_execution_modules"]
    (new / "src/model.py").write_bytes(b"# forbidden science change\n")
    with pytest.raises(ValueError, match="dirty"):
        reuse.authenticate(p, active, project=new, prepared_project=old)
    active = _commit(new)
    with pytest.raises(ValueError, match="scientific source changed"):
        reuse.authenticate(p, active, project=new, prepared_project=old)
    (new / "src/model.py").write_bytes((old / "src/model.py").read_bytes())
    (new / "src/unknown.py").write_bytes(b"# unreviewed addition\n")
    active = _commit(new)
    with pytest.raises(ValueError, match="additions/deletions"):
        reuse.authenticate(p, active, project=new, prepared_project=old)


def test_phase_failure_keeps_evidence_not_completion(tmp_path, capsys):
    log = gpu.PhaseLog(tmp_path, lambda: dict(rss_bytes=123, peak_rss_bytes=456, gpu_peak_bytes=789))
    with pytest.raises(MemoryError):
        with log.phase("fusion_stress"):
            raise MemoryError("injected")
    assert load_json(tmp_path / "phases/00-start.json")["status"] == "running"
    assert load_json(tmp_path / "phases/00-end.json")["status"] == "failed"
    assert not (tmp_path / "preflight.json").exists()
    assert "fusion_stress" in capsys.readouterr().out


def test_backend_and_memory_checks():
    with pytest.MonkeyPatch.context() as m:
        m.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        with pytest.raises(ValueError, match="before starting"):
            gpu.require_backend()
        m.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        gpu.require_backend()
    gpu.check_headroom(dict(peak_rss_bytes=100, gpu_peak_bytes=100),
                       memory_mb=1000, future_cache_bytes=100, gpu_bytes=1000)
    with pytest.raises(MemoryError, match="host RAM"):
        gpu.check_headroom(dict(peak_rss_bytes=850*2**20, gpu_peak_bytes=100),
                           memory_mb=1000, future_cache_bytes=1, gpu_bytes=1000)
    with pytest.raises(MemoryError, match="GPU"):
        gpu.check_headroom(dict(peak_rss_bytes=100, gpu_peak_bytes=900),
                           memory_mb=1000, future_cache_bytes=1, gpu_bytes=1000)


def test_worker_and_versioned_cli():
    root = Path(__file__).resolve().parents[1]
    worker = (root / "sbatch/run_luka_fullsim_stage2.sh").read_text()
    assert worker.index("export CUBLAS_WORKSPACE_CONFIG=:4096:8") < worker.index("exec python")
    assert "preflight-v2" in worker
    assert gpu.MEMORY_MB == 256000
    text = (root / "src/hlt_classification/luka_fullsim/preflight_v2.py").read_text()
    assert text.index("stats = stress(model, train)") < text.index("bundles[role] = build_caches")
    assert "acceptance_passes=1" in text and "GPU_PREFLIGHT/v2" in text
    assert "production_submission_authorized=False" in text


@pytest.mark.parametrize("fail_stress", [True, False])
def test_runtime_order_roundtrips_and_early_failure(prepared, monkeypatch, tmp_path, fail_stress):
    """Exercise actual v2 orchestration; GPU/Slurm/math are deliberately mocked."""
    import torch
    from hlt_classification.cms2jc2_response import measurement
    from hlt_classification.context_fusion import inputs, worker
    from hlt_classification.jetclass2_delphes import acceptance, execution, model, runner, dzfix_fusion_model
    from hlt_classification.data.cache_contracts import with_content_hash
    root, p = bind(prepared, monkeypatch)
    full = {role: cache.build_caches(root, role=role, max_ram_bytes=2**30) for role in ("train", "validation")}
    small, witnesses = cache.probe_caches(root)
    loaded = cache.s.load_prepared(root)
    p = deepcopy(p)
    # Only the orchestrator's real-population guard is mocked, not a production artifact.
    p["counts"] = dict(train=100000, validation=50000)
    events = []
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(gpu.s, "source", lambda *a: p["source_snapshot"])
    monkeypatch.setattr(gpu.s, "load_prepared", lambda *a: (p, *loaded[1:]))
    monkeypatch.setattr(gpu, "authenticate", lambda *a, **kw: dict(content_hash="c"*64))
    monkeypatch.setattr(gpu, "probe_caches", lambda *a: (small, witnesses))
    monkeypatch.setattr(gpu, "memory_snapshot", lambda *a: dict(rss_bytes=1, peak_rss_bytes=2, gpu_peak_bytes=3))
    def build(*a, role, **kw):
        events.append("cache_"+role)
        return full[role]
    monkeypatch.setattr(gpu, "build_caches", build)
    monkeypatch.setattr(execution, "allocation", lambda *a: ("123", 6, gpu.MEMORY_MB))
    monkeypatch.setattr(execution, "gpu_identity", lambda: dict(total_memory_bytes=100000))
    monkeypatch.setattr(model, "installed_environment", lambda: dict(test_only=True))
    monkeypatch.setattr(acceptance, "installed_parity", lambda *a, **kw: dict(passed=True))
    monkeypatch.setattr(dzfix_fusion_model, "native_mask_parity", lambda *a, **kw: dict(passed=True))
    monkeypatch.setattr(dzfix_fusion_model, "native_offload_parity", lambda *a, **kw: dict(passed=True))
    monkeypatch.setattr(dzfix_fusion_model, "validate_offload_stats", lambda *a, **kw: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(worker, "clear", lambda: None)
    monkeypatch.setattr(inputs, "new_model", lambda *a: torch.nn.Linear(1, 1))
    def stress(*a):
        events.append("stress")
        if fail_stress:
            raise MemoryError("injected GPU/host stress failure")
        return None
    monkeypatch.setattr(worker, "stress", stress)
    def train(m, train, val, **kwargs):
        events.append("train")
        assert kwargs["acceptance_passes"] == 1
        return with_content_hash(dict(runtime_seconds=1.)), m.state_dict()
    monkeypatch.setattr(runner, "train_kernel", train)
    monkeypatch.setattr(runner, "predict", lambda m, train, **kw: np.full((len(train), 11), 1/11, np.float32))
    class Measured:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def report(self):
            return dict(sampled_peak_tree_rss_bytes=3)
    monkeypatch.setattr(measurement, "Measurement", Measured)
    output = tmp_path / "runtime"
    kwargs = dict(project=Path(__file__).resolve().parents[1], expected_commit="a"*40,
                  prepared_project=tmp_path / "old_project", site_name="sporc_a100_debug")
    if fail_stress:
        with pytest.raises(MemoryError, match="injected"):
            gpu.run(root, output, **kwargs)
        assert events == ["stress"]
        assert not (output / "preflight.json").exists()
    else:
        r = gpu.run(root, output, **kwargs)
        assert events == ["stress"]*4 + ["cache_train", "cache_validation"] + ["train"]*4
        assert r["contract"] == "LUKA_FULLSIM_GPU_PREFLIGHT/v2"
        assert r["checkpoint_probability_roundtrip"] and r["technical_checks_passed"]
        assert not r["final_test_accessed"] and not r["production_submission_authorized"]
        assert (output / "preflight.json").is_file()


def test_downloaded_train_rows_local_diagnostic(monkeypatch, tmp_path):
    """Optional 128 real TRAIN jets; ephemeral test parents, never admission.

    Exercise real ROOT ranges, mappings and varied-length old/new cache bytes.
    We do not manufacture a production foundation or prepared.json here.
    """
    from test_luka_fullsim_stage2 import config
    from hlt_classification.data.cache_contracts import (
        validate_content_hash, sha256_file, atomic_publish_bytes, deterministic_npz_bytes,
    )
    from hlt_classification.luka_fullsim.particles import _matrix, _ranges, RawJet, FIELDS, physical
    from hlt_classification.luka_fullsim.contracts import row_identity, source_path
    from hlt_classification.luka_fullsim import stage2, stage2_cache
    local = Path(__file__).resolve().parents[1] / "artifacts/luka_fullsim_local"
    inv_path = local / "inspection_r1/diagnostic_inventory.json"
    if not inv_path.exists():
        pytest.skip("Downloaded real-data diagnostic fixture is not installed")
    inventory = load_json(inv_path)
    splits = load_json(local / "inspection_r1/diagnostic_splits.json")
    validate_content_hash(inventory, expected_contract="LUKA_FULLSIM_INVENTORY/v1")
    validate_content_hash(splits, expected_contract="LUKA_FULLSIM_SPLITS/v1")
    assert splits["parents"]["inventory"] == inventory["content_hash"]
    raw = local / "cms_fullsim_630k_trial_luka_v1/raw"
    jets, members, refs = [], [], []
    meters = {c: stage2.ViewMeter() for c in cache.COORDINATES}
    cfg = config()
    for m in splits["memberships"]["train"][:8]:
        index = m["file_index"]
        record = inventory["files"][index]
        path, _ = source_path(raw, record["path"])
        assert sha256_file(path) == record["sha256"]
        entries = dict(sorted((entry, label) for label, es in enumerate(m["entries_by_class"]) for entry in es)[:16])
        selected = [[] for _ in range(11)]
        offsets, mappings, identities, labels = [0], [], [], []
        branches = ["hlt_jet_nparticles", "jet_nparticles"] + [
            prefix+field for prefix in ("hlt_part_", "part_") for field in FIELDS]
        with uproot.open(path) as handle:
            for start, stop in _ranges(sorted(entries), 512):
                arrays = handle[record["tree_key"]].arrays(branches, entry_start=start, entry_stop=stop,
                                                          library="ak", how=dict)
                for i, entry in enumerate(range(start, stop)):
                    selected[entries[entry]].append(entry)
                    jet = RawJet(row_identity(record, entry), entries[entry], index, entry,
                        _matrix(arrays, "hlt_part_", i, int(arrays["hlt_jet_nparticles"][i])),
                        _matrix(arrays, "part_", i, int(arrays["jet_nparticles"][i])))
                    jets.append(jet)
                    hlt, off = physical(jet.hlt, cfg, "hlt"), physical(jet.offline, cfg, "offline")
                    mapping = stage2.views.match_particles(hlt, off)
                    mappings.append(mapping)
                    offsets.append(offsets[-1]+len(mapping))
                    identities.append(np.frombuffer(bytes.fromhex(jet.identity), np.uint8))
                    labels.append(jet.label)
                    for c, meter in meters.items():
                        value = stage2.build_inputs(stage2.views.build_view(identity=jet.identity,
                            proxy=hlt, offline=off, coordinate=c, mapping=mapping), capacity=512)
                        meter.add(jet, value)
        assert sha256_file(path) == record["sha256"]
        members.append(dict(file_index=index, entries_by_class=selected))
        path = tmp_path / f"assignment_{index}.npz"
        atomic_publish_bytes(path, deterministic_npz_bytes(dict(offsets=np.asarray(offsets, np.int64),
            mapping=np.concatenate(mappings).astype(np.int32), identities=np.asarray(identities, np.uint8),
            labels=np.asarray(labels, np.int64))))
        refs.append(dict(file_index=index, rows=len(labels), path=path.name,
                         bytes=path.stat().st_size, sha256=sha256_file(path)))
    assert len(jets) == 128
    p = dict(content_hash="a"*64, conventions=cfg, inputs=dict(capacity=512),
             summaries=dict(train={c: meter.report() for c, meter in meters.items()}),
             assignments=dict(train=refs), counts=dict(train=len(jets)))
    diagnostic = (p, dict(input_container=str(raw.parent)), inventory, dict(memberships=dict(train=members)))
    monkeypatch.setattr(stage2, "load_prepared", lambda *a: diagnostic)
    monkeypatch.setattr(cache, "ParticleReader", lambda *a, **kw: iter(jets))
    monkeypatch.setattr(stage2_cache, "ParticleReader", lambda *a, **kw: iter(jets))
    bundle = cache.build_caches(tmp_path, role="train", max_ram_bytes=2**30)
    small, evidence = cache.probe_caches(tmp_path)
    for c in cache.COORDINATES:
        original = prepare_cache(tmp_path, role="train", coordinate=c, max_ram_bytes=2**30)
        for k, value in original.batch(np.arange(128)).items():
            assert np.array_equal(bundle[c].batch(np.arange(128))[k], value)
        for k, value in largest_raw(original).items():
            assert np.array_equal(largest_raw(small[c])[k], value)
    print("Real TRAIN-only diagnostic:", len(jets), "jets; maxima:",
          {c: r["particles"] for c, r in evidence["witnesses"].items()},
          "; exact v1/v2 inputs and batch256 witnesses; NOT production admission.")
