"""Early unchanged worst-case stress, dual-source reuse and one-pass RAM caches."""
from contextlib import contextmanager
from pathlib import Path
import json
import math
import os
import threading
import time

import numpy as np

from hlt_classification.data.cache_contracts import sha256_file, with_content_hash, write_immutable_json
from . import stage2 as s
from .preflight_cache import build_caches, probe_caches, resident_bound
from .preflight_reuse import authenticate, ADDITIONS

MEMORY_MB = 256000  # Explicit new host-RAM envelope, not a measured sufficiency claim.
CASES = (("U000_probe", "U000", None), ("DIRECT_probe", "D000", None),
         ("FUSION_probe", "U050", "U000"), ("HLT_PAIR_probe", "D000", "D000"))


def require_backend():
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python")


def memory_snapshot(torch):
    import resource
    rss = int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
    return dict(rss_bytes=rss, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                gpu_peak_bytes=torch.cuda.max_memory_allocated())


class PhaseLog:
    """Flush phase boundaries/15-second heartbeats; preserve failure evidence."""
    def __init__(self, output, sample):
        self.output, self.sample, self.rows = Path(output), sample, []

    @contextmanager
    def phase(self, name):
        index, started, stop = len(self.rows), time.monotonic(), threading.Event()
        row = dict(phase=name, status="running", start=self.sample())
        self.rows.append(row)
        write_immutable_json(self.output / "phases" / f"{index:02d}-start.json", row)
        print("LUKA " + json.dumps(row, sort_keys=True), flush=True)
        def heartbeat():
            while not stop.wait(15):
                print("LUKA " + json.dumps(dict(phase=name, status="working",
                    seconds=time.monotonic()-started, **self.sample()), sort_keys=True), flush=True)
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            yield row
            row["status"] = "completed"
        except BaseException as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            stop.set()
            thread.join()
            row.update(seconds=time.monotonic()-started, end=self.sample())
            write_immutable_json(self.output / "phases" / f"{index:02d}-end.json", row)
            print("LUKA " + json.dumps(row, sort_keys=True), flush=True)


def check_headroom(snapshot, *, memory_mb, future_cache_bytes, gpu_bytes):
    if snapshot["peak_rss_bytes"] + future_cache_bytes >= .85*memory_mb*1024**2:
        raise MemoryError("Worst-case stress plus reserved full caches exceeds 85% host RAM; "
                          "stop before population loading. Do not weaken batch/support stress.")
    if snapshot["gpu_peak_bytes"] >= .90*gpu_bytes:
        raise MemoryError("Worst-case stress exceeds 90% GPU memory")


def _node(index):
    from hlt_classification.jetclass2_delphes.campaign import paired_seed
    name, coordinate, context = CASES[index]
    return dict(node_id=name, coordinate=coordinate, context_coordinate=context,
        initialization_seed=paired_seed(coordinate, "initialization"), context_initialization_seed=192031,
        sampler_seed=paired_seed(coordinate, "sampler"), teacher=None if index == 0 else "U000_probe")


def _view(caches, node):
    from hlt_classification.context_fusion.inputs import PairedCache
    primary = caches[node["coordinate"]]
    context = node["context_coordinate"]
    return primary if context is None else PairedCache(primary, caches[context])


def run(prepared_root, output, *, project, expected_commit, prepared_project, site_name):
    require_backend()  # Before source hashing, ROOT loading or CUDA initialization.
    import torch
    from hlt_classification.cms2jc2_response.measurement import Measurement
    from hlt_classification.context_fusion.inputs import new_model
    from hlt_classification.context_fusion.worker import clear, save_state, stress
    from hlt_classification.jetclass2_delphes.acceptance import installed_parity
    from hlt_classification.jetclass2_delphes.banks import publish_bank, load_bank
    from hlt_classification.jetclass2_delphes.campaign import recipe
    from hlt_classification.jetclass2_delphes.dzfix_fusion_model import (
        native_mask_parity, native_offload_parity, validate_offload_stats,
    )
    from hlt_classification.jetclass2_delphes.execution import execution_site, allocation, gpu_identity
    from hlt_classification.jetclass2_delphes.model import installed_environment
    from hlt_classification.jetclass2_delphes.runner import train_kernel, predict

    print("LUKA phase=authenticate prepared-v1 and both clean source worktrees", flush=True)
    if Path(__file__).resolve() != Path(project).resolve() / "src/hlt_classification/luka_fullsim/preflight_v2.py":
        raise ValueError("Preflight-v2 import escapes pinned project")
    pinned = s.source(project, expected_commit)
    if any(not (Path(project) / name).is_file() for name in ADDITIONS):
        raise ValueError("Incomplete preflight-v2 source")
    p, f, _, _ = s.load_prepared(prepared_root)
    reuse = authenticate(p, pinned, project=project, prepared_project=prepared_project)
    if p["counts"] != dict(train=100000, validation=50000):
        raise ValueError("Real preflight requires the frozen full 100k/50k population")
    if site_name not in ("sporc_a100", "sporc_a100_debug"):
        raise PermissionError("Only registered SPORC A100 sites")
    site = execution_site(site_name)
    job, cpus, memory = allocation(site)
    if (cpus, memory) != (6, MEMORY_MB):
        raise PermissionError(f"Preflight-v2 requires exactly 6 CPUs and {MEMORY_MB} MiB")
    environment, gpu = installed_environment(), gpu_identity()
    output = s.fresh(output, prepared_root, p["foundation_root"], f["input_container"], project, prepared_project)
    write_immutable_json(output / "prepared_reuse.json", reuse)
    write_immutable_json(output / "runtime.json", dict(job_id=job, cpus=cpus, memory_mb=memory,
        environment=environment, gpu=gpu, cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        cudnn_v8_disabled=os.environ.get("TORCH_CUDNN_V8_API_DISABLED", "unset"),
        scientific_fit=False, final_test_accessed=False))
    bounds = {role: resident_bound(p, role) for role in ("train", "validation")}
    if sum(bounds.values()) > .15*memory*1024**2:
        raise MemoryError("Full RAM cache reservation exceeds registered 15% host envelope")
    log = PhaseLog(output, lambda: memory_snapshot(torch))
    parities, stress_results, measurements = {}, {}, []
    teacher_args = teacher_ids = None
    with Measurement() as measured:
        torch.cuda.reset_peak_memory_stats()
        with log.phase("small_train_witnesses"):
            small, witnesses = probe_caches(prepared_root)
        with log.phase("installed_and_native_mask_parity"):
            parities["installed"] = installed_parity(small["U000"], device="cuda")
            parities["native_mask"] = native_mask_parity(small["U000"].batch(np.arange(4)), device="cuda")
            clear()
        # Every parity and original three-step batch-256 stress precedes full caches.
        for index in range(len(CASES)):
            node = _node(index)
            name = node["node_id"]
            train = _view(small, node)
            if node["context_coordinate"] is not None:
                for bf16 in (False, True):
                    with log.phase(f"{name}_offload_parity_bf16_{bf16}"):
                        parities[f"{name}_{bf16}"] = native_offload_parity(
                            train.primary.batch(np.arange(4)), train.context.batch(np.arange(4)),
                            device="cuda", bf16=bf16)
                        clear()
            with log.phase(f"{name}_stress_batch256_steps3"):
                model = new_model(node)
                stats = stress(model, train)
                if node["context_coordinate"] is not None:
                    validate_offload_stats(stats, calls=3)
                del model
                clear()
                stress_results[name] = stats
                check_headroom(memory_snapshot(torch), memory_mb=memory,
                               future_cache_bytes=sum(bounds.values()), gpu_bytes=gpu["total_memory_bytes"])
        del train, small
        clear()
        bundles, cache_seconds = {}, {}
        for role in ("train", "validation"):
            with log.phase(f"{role}_all_views_once") as phase:
                bundles[role] = build_caches(prepared_root, role=role, max_ram_bytes=bounds[role])
            cache_seconds[role] = phase["seconds"]
        cache_peak = sum(cache.nbytes for bundle in bundles.values() for cache in bundle.values())
        for index in range(len(CASES)):
            node = _node(index)
            name = node["node_id"]
            train, val = (_view(bundles[r], node) for r in ("train", "validation"))
            kwargs = {}
            if index:
                if not np.array_equal(teacher_ids, train.identities):
                    raise ValueError("Probe teacher population differs")
                kwargs = dict(teacher_probabilities=load_bank(output / "probe_train_bank",
                    expected_identities=train.identities, **teacher_args), teacher_identities=train.identities)
            with log.phase(f"{name}_full_population_pass"):
                model = new_model(node)
                training, state = train_kernel(model, train, val, node=node, device="cuda",
                                               acceptance_passes=1, **kwargs)
            with log.phase(f"{name}_checkpoint_and_train_bank") as phase:
                path = output / (name + ".pt")
                save_state(path, state)
                replay = torch.load(path, map_location="cpu", weights_only=True)
                if state.keys() != replay.keys() or any(not torch.equal(state[k], replay[k]) for k in state):
                    raise ValueError("Selected technical weights readback differs")
                model.load_state_dict(replay, strict=True)
                probabilities = predict(model, train, device="cuda", temperature=2.)
                bank_args = dict(foundation_sha256=p["content_hash"], teacher_report_sha256=training["content_hash"],
                                 teacher_node=name, role="train")
                bank_root = output / ("probe_train_bank" if index == 0 else name + "_train_bank")
                publish_bank(bank_root, identities=train.identities, probabilities=probabilities, **bank_args)
                replay_bank = load_bank(bank_root, expected_identities=train.identities, **bank_args)
                if not np.array_equal(replay_bank, probabilities):
                    raise ValueError("Technical probability readback differs")
                if index == 0:
                    teacher_args, teacher_ids = bank_args, train.identities.copy()
            measurements.append(dict(name=name, node=node, training=training,
                cache_seconds=sum(cache_seconds.values()), cache_shared=True, inference_seconds=phase["seconds"],
                stress_steps=3, stress_batch=256, stress_offload=stress_results[name]))
            del model, state, replay, probabilities, replay_bank, kwargs, train, val
            clear()
    train_minutes = math.ceil(max(1.75*(r["cache_seconds"]+100*r["training"]["runtime_seconds"])
                                  for r in measurements)/60)
    reduce_minutes = math.ceil(max(2*(r["cache_seconds"]+r["inference_seconds"])
                                   for r in measurements)/60)
    snapshot = memory_snapshot(torch)
    peak_rss = max(snapshot["peak_rss_bytes"], measured.report()["sampled_peak_tree_rss_bytes"])
    if authenticate(p, pinned, project=project, prepared_project=prepared_project) != reuse:
        raise ValueError("Prepared source compatibility changed during preflight")
    if s.load_prepared(prepared_root)[0] != p or installed_environment() != environment:
        raise ValueError("Prepared artifacts/installed environment changed during preflight")
    outputs = {path.relative_to(output).as_posix(): dict(sha256=sha256_file(path), bytes=path.stat().st_size)
               for path in sorted(output.rglob("*")) if path.is_file()}
    result = with_content_hash(dict(contract="LUKA_FULLSIM_GPU_PREFLIGHT/v2", schema_version=2,
        parents=dict(prepared=p["content_hash"], source=pinned["content_hash"], reuse=reuse["content_hash"]),
        source_snapshot=pinned, preparation_source_snapshot=p["source_snapshot"], counts=p["counts"],
        site=site, job_id=job, cpus=cpus, memory_mb=memory, environment=environment, gpu=gpu,
        parities=parities, stress_witnesses=witnesses, measurements=measurements, phases=log.rows,
        outputs=outputs, recipe=recipe(), resource_measurement=measured.report(), peak_rss_bytes=peak_rss,
        gpu_peak_bytes=snapshot["gpu_peak_bytes"], cache_peak_bytes=cache_peak, cache_build_seconds=cache_seconds,
        projected_train_minutes=train_minutes, projected_reduce_minutes=reduce_minutes,
        resource_envelope_ok=bool(peak_rss < .85*memory*1024**2 and snapshot["gpu_peak_bytes"] < .90*gpu["total_memory_bytes"]),
        fits_debug_24h=bool(max(train_minutes, reduce_minutes) <= 1440),
        checkpoint_probability_roundtrip=True, technical_checks_passed=True,
        scientific_fit=False, scientific_metrics_for_selection=False, final_test_accessed=False,
        production_submission_authorized=False, automatic_followup=False))
    write_immutable_json(output / "preflight.json", result)
    return result
