"""Real installed-Weaver adapter parity and bounded CE/oracle/direct-KD check.

This is a technical acceptance probe, never a scored scientific campaign.
Full-population memory, Slurm resources, and submission gates remain separate.
"""
from __future__ import annotations

from itertools import islice
from pathlib import Path
import platform
import sys

import numpy as np
import torch

from hlt_classification.data.cache_contracts import write_immutable_json
from hlt_classification.models.particle_transformer import load_weaver_particle_transformer_class
from .banks import publish_bank, load_bank
from .cache import RamBlock, RamCache
from .campaign import build_campaign_plan, paired_seed
from .contracts import artifact
from .foundation import validate_foundation_spec
from .inputs import build_inputs
from .model import DelphesParticleTransformer, model_config, model_contract
from .provenance import source_record
from .reader import DatasetReader
from .splits import file_population
from .runner import train_kernel, predict


def _sample(spec, data_root, role, per_class):
    rows, counts = [], np.zeros(11, np.int64)
    for task in spec["assignment_tasks"]:
        if task["role"] != role:
            continue
        present = np.asarray(file_population(spec["splits"], spec["inventory"]["files"][task["file_index"]], role)[1]) > 0
        if not np.any((counts < per_class) & present):
            continue
        reader = DatasetReader(data_root, spec["inventory"], spec["splits"], role=role,
                               include_offline=True, file_paths=(task["path"],), step_size=512)
        for jet in islice(reader, 2048):
            if counts[jet.label] < per_class:
                rows.append(jet)
                counts[jet.label] += 1
            if not np.any((counts < per_class) & present):
                break
        if np.all(counts >= per_class):
            break
    if np.any(counts != per_class):
        raise ValueError(f"Acceptance sampling did not find requested class coverage: {counts.tolist()}")
    return rows


def _cache(jets, spec, role, coordinate_name):
    offsets, features, vectors, labels, identities = [0], [], [], [], []
    for jet in jets:
        raw = jet.hlt if coordinate_name == "D000" else jet.offline
        value = build_inputs(raw, capacity=spec["inputs"]["capacity"])
        offsets.append(offsets[-1] + len(raw))
        features.append(value.features[:, :len(raw)].T)
        vectors.append(value.vectors[:, :len(raw)].T)
        labels.append(jet.label)
        identities.append(np.frombuffer(bytes.fromhex(jet.identity), np.uint8))
    block = RamBlock(0, np.asarray(offsets, np.int64), np.concatenate(features), np.concatenate(vectors),
                     np.asarray(identities, np.uint8), np.asarray(labels, np.int64))
    return RamCache([block], role=role, foundation_sha256=spec["content_hash"], coordinate_name=coordinate_name)


def installed_parity(cache: RamCache, *, device) -> dict:
    torch.manual_seed(71)
    wrapper = DelphesParticleTransformer().to(device).eval()
    direct = load_weaver_particle_transformer_class()(**model_config()).to(device).eval()
    direct.load_state_dict(wrapper.mod.state_dict())
    raw = cache.batch(np.arange(min(4, len(cache))))
    x1 = torch.tensor(raw["features"], device=device, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    vectors, mask = (torch.tensor(raw[k], device=device) for k in ("vectors", "mask"))
    y1, y2 = wrapper(x1, vectors, mask), direct(x2, v=vectors, mask=mask)
    torch.testing.assert_close(y1, y2, rtol=1e-5, atol=1e-6)
    y1.square().mean().backward()
    y2.square().mean().backward()
    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-5, atol=1e-6)
    if not torch.isfinite(y1).all() or not torch.isfinite(x1.grad).all():
        raise ValueError("Nonfinite parity logits/input gradients")
    for (_, left), (_, right) in zip(wrapper.mod.named_parameters(), direct.named_parameters()):
        if left.grad is None or right.grad is None:
            if left.grad is not None or right.grad is not None:
                raise ValueError("Parity parameter gradient topology differs")
        else:
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)
            if not torch.isfinite(left.grad).all():
                raise ValueError("Nonfinite parity parameter gradient")
    return artifact("WEAVER_PARITY", model=model_contract(), device=str(device), passed=True,
                    forward_and_feature_and_parameter_gradients=True, final_test_accessed=False)


def run_acceptance(spec: dict, *, data_root: Path, output_root: Path, device="cuda", rows_per_class=16) -> dict:
    parent = validate_foundation_spec(spec)
    if rows_per_class < 2 or rows_per_class > 64:
        raise ValueError("Technical acceptance requires 2-64 rows per class")
    if Path(output_root).resolve().is_relative_to(Path(data_root).resolve()):
        raise ValueError("Acceptance output cannot be inside raw snapshot")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA acceptance has no CUDA device")
    load_weaver_particle_transformer_class()  # fail before scanning data if absent
    caches = {}
    for role in ("train", "validation"):
        rows = _sample(spec, data_root, role, rows_per_class)
        for coord in ("D000", "U000"):
            caches[(role, coord)] = _cache(rows, spec, role, coord)
    parity = installed_parity(caches[("train", "D000")], device=device)
    write_immutable_json(Path(output_root) / "parity.json", parity)
    plan = build_campaign_plan(spec)
    reports = []
    bank = None
    for node in [plan["nodes"][0], plan["nodes"][1], plan["nodes"][2]]:
        torch.manual_seed(node["initialization_seed"])
        model = DelphesParticleTransformer()
        train, val = (caches[(role, node["coordinate"])] for role in ("train", "validation"))
        kwargs = {} if node["teacher"] is None else dict(teacher_probabilities=bank, teacher_identities=train.identities)
        report, _ = train_kernel(model, train, val, node=node, device=device, acceptance_passes=1, **kwargs)
        reports.append(report)
        write_immutable_json(Path(output_root) / f"{node['node_id']}.json", report)
        if node["node_id"] == "U000":
            p = predict(model, train, device=device, temperature=2.)
            arguments = dict(foundation_sha256=parent, teacher_report_sha256=report["content_hash"], teacher_node="U000", role="train")
            publish_bank(Path(output_root) / "temporary_teacher_bank", identities=train.identities, probabilities=p, **arguments)
            bank = load_bank(Path(output_root) / "temporary_teacher_bank", expected_identities=train.identities, **arguments)
        del model
    result = artifact(
        "LOCAL_OR_REMOTE_ACCEPTANCE", foundation_sha256=parent, parity_sha256=parity["content_hash"],
        reports=[r["content_hash"] for r in reports], passed=True, scientific_results=False,
        device=str(device), platform=platform.platform(), python=sys.version, torch_version=str(torch.__version__),
        rows_per_class=rows_per_class, final_test_accessed=False, full_population_resources_measured=False,
        production_submission_authorized=False, stored_outputs="small_reports_and_tiny_acceptance_probability_bank_no_views_or_resume",
        producer=source_record(*[f"src/hlt_classification/jetclass2_delphes/{name}.py" for name in
                                ("acceptance", "banks", "cache", "campaign", "inputs", "model", "runner", "reporting")]),
    )
    write_immutable_json(Path(output_root) / "acceptance.json", result)
    return result
