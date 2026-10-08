"""Real SPORC technical passes only; no scientific weights or campaign launch."""
from pathlib import Path
import math
import time

import numpy as np

from hlt_classification.data.cache_contracts import sha256_file, write_immutable_json
from hlt_classification.provenance import validate_source_snapshot
from .contracts import artifact
from . import stage2 as s


def run(prepared_root, output, *, project, expected_commit, site_name):
    import resource
    import torch
    from hlt_classification.cms2jc2_response.measurement import Measurement
    from hlt_classification.context_fusion.inputs import PairedCache, new_model
    from hlt_classification.context_fusion.worker import clear, save_state, stress
    from hlt_classification.jetclass2_delphes.acceptance import installed_parity
    from hlt_classification.jetclass2_delphes.banks import publish_bank, load_bank
    from hlt_classification.jetclass2_delphes.campaign import recipe, paired_seed
    from hlt_classification.jetclass2_delphes.dzfix_fusion_model import (
        native_mask_parity, native_offload_parity, validate_offload_stats,
    )
    from hlt_classification.jetclass2_delphes.execution import execution_site, allocation, gpu_identity
    from hlt_classification.jetclass2_delphes.model import installed_environment
    from hlt_classification.jetclass2_delphes.runner import train_kernel, predict
    from .stage2_cache import prepare_cache

    pinned = s.source(project, expected_commit)
    p, f, _, _ = s.load_prepared(prepared_root)
    # Preparation and GPU execution must use identical committed source.
    validate_source_snapshot(p["source_snapshot"], repository=project, require_clean=True)
    if p["counts"] != dict(train=100000, validation=50000):
        raise ValueError("Real preflight requires the full frozen 100k/50k population")
    if site_name not in ("sporc_a100", "sporc_a100_debug"):
        raise PermissionError("Only explicit SPORC A100 profiles are registered")
    site = execution_site(site_name)
    job, cpus, memory = allocation(site)
    if cpus != 6 or memory != 90000:
        raise PermissionError("Preflight requires exactly 6 CPUs and 90000 MiB")
    output = s.fresh(output, prepared_root, p["foundation_root"], f["input_container"], project)
    budget = int(.65*memory*1024**2/4)
    cases = (("U000_probe", "U000", None), ("DIRECT_probe", "D000", None),
             ("FUSION_probe", "U050", "U000"), ("HLT_PAIR_probe", "D000", "D000"))
    measurements, parities, teacher_args, teacher_ids = [], {}, None, None
    gpu_peak = cache_peak = 0
    environment, gpu = installed_environment(), gpu_identity()
    with Measurement() as measured:
        torch.cuda.reset_peak_memory_stats()
        for index, (name, coordinate, context) in enumerate(cases):
            print(f"LUKA technical preflight={name}; NONSCIENTIFIC one full pass", flush=True)
            started = time.monotonic()
            def cache(role):
                primary = prepare_cache(prepared_root, role=role, coordinate=coordinate, max_ram_bytes=budget)
                if context is None:
                    return primary
                other = primary if context == coordinate else prepare_cache(
                    prepared_root, role=role, coordinate=context, max_ram_bytes=budget)
                return PairedCache(primary, other)
            train, val = cache("train"), cache("validation")
            cache_seconds = time.monotonic()-started
            cache_peak = max(cache_peak, train.nbytes+val.nbytes)
            if index == 0:
                parities["installed"] = installed_parity(train, device="cuda")
                parities["native_mask"] = native_mask_parity(train.batch(np.arange(4)), device="cuda")
            if context is not None:
                for bf16 in (False, True):
                    parities[f"{name}_{bf16}"] = native_offload_parity(
                        train.primary.batch(np.arange(4)), train.context.batch(np.arange(4)),
                        device="cuda", bf16=bf16)
                    clear()
            node = dict(node_id=name, coordinate=coordinate, context_coordinate=context,
                initialization_seed=paired_seed(coordinate, "initialization"),
                context_initialization_seed=192031, sampler_seed=paired_seed(coordinate, "sampler"),
                teacher=None if index == 0 else "U000_probe")
            model = new_model(node)
            stress_stats = stress(model, train)
            if context is not None:
                validate_offload_stats(stress_stats, calls=3)
            del model
            clear()
            kwargs = {}
            if index:
                if not np.array_equal(teacher_ids, train.identities):
                    raise ValueError("Probe teacher population differs")
                kwargs = dict(teacher_probabilities=load_bank(output / "probe_train_bank",
                    expected_identities=train.identities, **teacher_args), teacher_identities=train.identities)
            model = new_model(node)
            training, state = train_kernel(model, train, val, node=node, device="cuda",
                                           acceptance_passes=1, **kwargs)
            path = output / (name + ".pt")
            save_state(path, state)
            replay = torch.load(path, map_location="cpu", weights_only=True)
            if state.keys() != replay.keys() or any(not torch.equal(state[k], replay[k]) for k in state):
                raise ValueError("Selected technical weights readback differs")
            model.load_state_dict(replay, strict=True)
            started = time.monotonic()
            probabilities = predict(model, train, device="cuda", temperature=2.)
            bank_args = dict(foundation_sha256=p["content_hash"], teacher_report_sha256=training["content_hash"],
                             teacher_node=name, role="train")
            bank_root = output / ("probe_train_bank" if index == 0 else name + "_train_bank")
            publish_bank(bank_root, identities=train.identities, probabilities=probabilities, **bank_args)
            bank_replay = load_bank(bank_root, expected_identities=train.identities, **bank_args)
            if not np.array_equal(bank_replay, probabilities):
                raise ValueError("Technical probability readback differs")
            if index == 0:
                teacher_args, teacher_ids = bank_args, train.identities.copy()
            inference_seconds = time.monotonic()-started
            measurements.append(dict(name=name, node=node, training=training, cache_seconds=cache_seconds,
                inference_seconds=inference_seconds, stress_steps=3, stress_batch=256, stress_offload=stress_stats))
            gpu_peak = max(gpu_peak, torch.cuda.max_memory_allocated())
            del train, val, kwargs, model, state, replay, probabilities, bank_replay
            clear()
    train_minutes = math.ceil(max(1.75*(r["cache_seconds"]+100*r["training"]["runtime_seconds"])
                                  for r in measurements)/60)
    reduce_minutes = math.ceil(max(2*(r["cache_seconds"]+r["inference_seconds"])
                                   for r in measurements)/60)
    peak_rss = max(measured.report()["sampled_peak_tree_rss_bytes"],
                   resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    envelope = peak_rss < .85*memory*1024**2 and gpu_peak < .90*gpu["total_memory_bytes"]
    validate_source_snapshot(pinned, repository=project, require_clean=True)
    if installed_environment() != environment:
        raise ValueError("Installed environment changed during preflight")
    outputs = {path.relative_to(output).as_posix(): dict(sha256=sha256_file(path), bytes=path.stat().st_size)
               for path in sorted(output.rglob("*")) if path.is_file()}
    result = artifact("GPU_PREFLIGHT", parents=dict(prepared=p["content_hash"], source=pinned["content_hash"]),
        source_snapshot=pinned, counts=p["counts"], site=site, job_id=job, cpus=cpus, memory_mb=memory,
        environment=environment, gpu=gpu, parities=parities, measurements=measurements, outputs=outputs,
        recipe=recipe(), resource_measurement=measured.report(), peak_rss_bytes=peak_rss,
        gpu_peak_bytes=gpu_peak, cache_peak_bytes=cache_peak, projected_train_minutes=train_minutes,
        projected_reduce_minutes=reduce_minutes, resource_envelope_ok=bool(envelope),
        fits_debug_24h=bool(max(train_minutes, reduce_minutes) <= 1440),
        checkpoint_probability_roundtrip=True, technical_checks_passed=True,
        scientific_fit=False, scientific_metrics_for_selection=False, final_test_accessed=False,
        production_submission_authorized=False, automatic_followup=False)
    write_immutable_json(output / "preflight.json", result)
    return result
