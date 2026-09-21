"""Matched-seed JetClass2 U100 screen for salience-matcher selection."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, load_json, sha256_file, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag

from .cache import RamCache, prepare_cache as prepare_bottleneck_cache
from .campaign import paired_seed
from .contracts import artifact, relative_file, validate
from .execution import allocation, execution_site, gpu_identity, slurm_options
from .foundation import validate_foundation_spec as validate_bottleneck_foundation
from .model import DelphesParticleTransformer, installed_environment, model_contract
from .production import _source, validate_profile
from .reporting import evaluate_probabilities
from .runner import predict, train_kernel
from .salience_cache import prepare_cache as prepare_salience_cache
from .salience_foundation import authenticate_preparation, validate_foundation_spec
from .submission import _guarded_exact_submission

AUTHORIZE = "AUTHORIZE JETCLASS2 500K SALIENCE U100 SCREEN"
REGISTRY = ["SALIENCE_PT_LINEAR", "SALIENCE_PT_QUADRATIC", "SALIENCE_PT_QUADRATIC_CORE25"]
CONTEXT = "BOTTLENECK_CONTEXT"
DEBUG_SCREEN_POLICY = "sporc_debug_same_a100_screen_only_production_tier3_v1"


@dataclass
class SubsetCache:
    source: RamCache
    indices: np.ndarray

    def __post_init__(self):
        self.indices = np.asarray(self.indices, np.int64)
        if self.indices.ndim != 1 or not len(self.indices):
            raise ValueError("Screen cache subset must be nonempty")
        self.role = self.source.role
        self.foundation_sha256 = self.source.foundation_sha256
        self.coordinate_name = self.source.coordinate_name
        self.labels = self.source.labels[self.indices]
        self.identities = self.source.identities[self.indices]
        self.nbytes = self.labels.nbytes + self.identities.nbytes

    def __len__(self):
        return len(self.indices)

    def batch(self, indices):
        return self.source.batch(self.indices[np.asarray(indices)])


def _selection_indices(labels: np.ndarray, identities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    checkpoint, selection = [], []
    for label in sorted(set(map(int, labels))):
        rows = np.flatnonzero(labels == label)
        if len(rows) < 2:
            raise ValueError("Every screen class requires checkpoint and selection rows")
        ordered = sorted(rows, key=lambda i: hashlib.sha256(
            b"JC2_SALIENCE_SCREEN_SPLIT/v1" + identities[i].tobytes()
        ).digest())
        cut = max(1, min(len(ordered) - 1, (3 * len(ordered)) // 4))
        checkpoint.extend(ordered[:cut]); selection.extend(ordered[cut:])
    return np.asarray(sorted(checkpoint), np.int64), np.asarray(sorted(selection), np.int64)


def _load_foundations(spec: dict, *, deep: bool = False) -> dict[str, tuple[dict, Path]]:
    result = {}
    bottleneck = load_json(Path(spec["bottleneck_root"]) / "foundation_spec.json")
    validate_bottleneck_foundation(bottleneck)
    bottleneck_lock = load_json(Path(spec["bottleneck_root"]) / "foundation_lock.json")
    validate(bottleneck_lock, "FOUNDATION_LOCK")
    if bottleneck_lock["foundation_sha256"] != bottleneck["content_hash"]:
        raise ValueError("Context bottleneck foundation lock differs")
    result[CONTEXT] = (bottleneck, Path(spec["bottleneck_root"]))
    for row in spec["candidates"]:
        foundation = load_json(Path(row["foundation_root"]) / "foundation_spec.json")
        validate_foundation_spec(foundation)
        if foundation["candidate"] != row["candidate"]:
            raise ValueError("Screen candidate/foundation identity differs")
        lock = load_json(Path(row["foundation_root"]) / "foundation_lock.json")
        validate(lock, "SALIENCE_FOUNDATION_LOCK")
        if lock["foundation_sha256"] != foundation["content_hash"]:
            raise ValueError("Screen salience foundation lock differs")
        if deep:
            authenticate_preparation(foundation, Path(row["foundation_root"]))
        result[row["candidate"]] = (foundation, Path(row["foundation_root"]))
    return result


def _template(path: Path, bottleneck_foundation: dict | None = None) -> dict:
    value = load_json(path)
    validate(value, "RUNTIME_PROFILE", version=value["schema_version"])
    common_invalid = (value["execution_site"] != execution_site("sporc_a100")
            or value["ram_only_views"] is not True or value["rolling_resume"] is not False
            or value["final_test_accessed"] is not False
            or value.get("passed") is not True
            or value.get("measured_full_population") is not True
            or value.get("selected_state_bytes", 0) <= 0)
    if common_invalid:
        raise ValueError("Screen resource template is not the measured SPORC A100 profile")
    # Historical screens are immutable and retain their exact established
    # resource tuple.  A newer dataset may instead supply a complete,
    # foundation-bound runtime profile produced by the ordinary readiness
    # gate.  That route is validated deeply rather than accepted by looser
    # numeric bounds.
    legacy = (
        (value["cpus"], value["memory_mb"], value["workers"]) == (8, 73728, 8)
        and (value["train_minutes"], value["reduce_minutes"]) == (808, 43)
    )
    if not legacy:
        if bottleneck_foundation is None:
            raise ValueError("A nonlegacy screen profile requires its bottleneck foundation")
        validate_profile(value, bottleneck_foundation, value.get("source_commit"))
    return value


def create_screen(*, bottleneck_root: Path, candidate_roots: list[Path],
                  resource_template: Path, data_root: Path, output_root: Path,
                  project: Path, source_commit: str,
                  screen_execution_site_name: str = "sporc_a100",
                  debug_walltime_minutes: int = 480) -> dict:
    _source(project, source_commit)
    if len(candidate_roots) != 3:
        raise ValueError("Exactly three registered salience candidates are required")
    candidates = []
    for root in candidate_roots:
        foundation = load_json(Path(root) / "foundation_spec.json")
        validate_foundation_spec(foundation)
        authenticate_preparation(foundation, Path(root))
        candidates.append(dict(candidate=foundation["candidate"],
                               foundation_root=str(Path(root).resolve()),
                               foundation_sha256=foundation["content_hash"]))
    if [row["candidate"] for row in candidates] != REGISTRY:
        raise ValueError("Candidate roots must be supplied in registered order")
    bottleneck = load_json(Path(bottleneck_root) / "foundation_spec.json")
    validate_bottleneck_foundation(bottleneck)
    template = _template(resource_template, bottleneck)
    reference = candidates[0]
    first = load_json(Path(reference["foundation_root"]) / "foundation_spec.json")
    if any(load_json(Path(r["foundation_root"]) / "foundation_spec.json")["splits"] != first["splits"] for r in candidates[1:]):
        raise ValueError("Salience screen candidates do not share one frozen split")
    if bottleneck["splits"] != first["splits"] or bottleneck["inventory"] != first["inventory"]:
        raise ValueError("Context control and salience candidates differ in population")
    root = Path(output_root).resolve()
    if root.exists() or root.is_relative_to(Path(data_root).resolve()):
        raise FileExistsError("Screen requires a fresh isolated root")
    if screen_execution_site_name not in {"sporc_a100", "sporc_a100_debug"}:
        raise ValueError("Screen execution site is not an authorized SPORC A100 site")
    debug = screen_execution_site_name == "sporc_a100_debug"
    if type(debug_walltime_minutes) is not int or not 60 <= debug_walltime_minutes <= 480:
        raise ValueError("Debug screen walltime must be between 60 and 480 minutes")
    site_fields = {}
    if debug:
        site_fields = dict(
            screen_execution_site=execution_site("sporc_a100_debug"),
            production_execution_site=execution_site("sporc_a100"),
            execution_policy=DEBUG_SCREEN_POLICY,
            scientific_configuration_unchanged=True,
            execution_changes=["partition_tier3_to_debug", "bounded_debug_walltime"],
            debug_walltime_minutes=debug_walltime_minutes,
        )
    spec = artifact(
        "SALIENCE_SCREEN_SPEC", version=2 if debug else 1,
        source_commit=source_commit,
        project_dir=str(Path(project).resolve()), screen_root=str(root),
        data_root=str(Path(data_root).resolve()), bottleneck_root=str(Path(bottleneck_root).resolve()),
        bottleneck_sha256=bottleneck["content_hash"], candidates=candidates,
        resource_template_path=str(Path(resource_template).resolve()),
        resource_template_sha256=template["content_hash"],
        split_profile=first["splits"]["profile"],
        role_counts=first["splits"]["role_counts"], coordinate="U100",
        candidate_registry=REGISTRY, contextual_control=CONTEXT,
        validation_partition={"checkpoint_fraction": [3, 4], "selection_fraction": [1, 4],
                              "stratified": True, "identity_hash_bound": True},
        scientific_fit_count=4, final_test_accessed=False,
        existing_campaign_mutations=False,
        **site_fields,
    )
    write_immutable_json(root / "screen_spec.json", spec)
    submit_screen(spec, execute=False)
    return spec


def validate_screen(spec: dict, *, check_source=True, deep=False) -> str:
    version = spec.get("schema_version")
    if version not in {1, 2}:
        raise ValueError("Unsupported salience screen schema")
    digest = validate(spec, "SALIENCE_SCREEN_SPEC", version=version)
    if (spec["candidate_registry"] != REGISTRY or spec["contextual_control"] != CONTEXT
            or spec["scientific_fit_count"] != 4 or spec["coordinate"] != "U100"
            or spec["final_test_accessed"] is not False
            or spec["existing_campaign_mutations"] is not False):
        raise ValueError("Salience screen scientific scope differs")
    if version == 2 and (
        spec.get("screen_execution_site") != execution_site("sporc_a100_debug")
        or spec.get("production_execution_site") != execution_site("sporc_a100")
        or spec.get("execution_policy") != DEBUG_SCREEN_POLICY
        or spec.get("scientific_configuration_unchanged") is not True
        or spec.get("execution_changes") != [
            "partition_tier3_to_debug", "bounded_debug_walltime"
        ]
        or type(spec.get("debug_walltime_minutes")) is not int
        or not 60 <= spec["debug_walltime_minutes"] <= 480
    ):
        raise ValueError("Debug salience screen execution policy differs")
    foundations = _load_foundations(spec, deep=deep)
    template = _template(
        Path(spec["resource_template_path"]), foundations[CONTEXT][0],
    )
    if template["content_hash"] != spec["resource_template_sha256"]:
        raise ValueError("Screen resource template changed")
    if check_source:
        _source(Path(spec["project_dir"]), spec["source_commit"])
    return digest


def task_graph() -> list[dict]:
    fits = [CONTEXT] + REGISTRY
    rows = [dict(task_id="authenticate", kind="authenticate", dependencies=[]),
            dict(task_id="preflight", kind="preflight", dependencies=["authenticate"])]
    rows += [dict(task_id="fit_" + name, kind="fit", candidate=name,
                  dependencies=["preflight"]) for name in fits]
    rows += [dict(task_id="select", kind="select",
                  dependencies=["fit_" + name for name in fits]),
             dict(task_id="complete", kind="complete", dependencies=["select"])]
    return rows


def _task_report(spec: dict, task: str) -> dict | None:
    path = Path(spec["screen_root"]) / "tasks" / f"{task}.json"
    if not path.is_file():
        return None
    value = load_json(path); validate(value, "SALIENCE_SCREEN_TASK")
    if (value["screen_sha256"] != spec["content_hash"] or value["task_id"] != task
            or value["source_commit"] != spec["source_commit"]
            or value["final_test_accessed"] is not False):
        raise ValueError("Screen task lineage differs")
    paths = [row["path"] for row in value["outputs"]]
    if len(paths) != len(set(paths)) or not paths:
        raise ValueError("Screen task output inventory differs")
    for row in value["outputs"]:
        if sha256_file(relative_file(Path(spec["screen_root"]), row["path"])) != row["sha256"]:
            raise ValueError("Screen task output checksum differs")
    kind = next(row["kind"] for row in task_graph() if row["task_id"] == task)
    required = {
        "preflight": ["screen_split.json", "runtime_profile.json"],
        "select": ["selection_lock.json"], "complete": ["screen_complete.json"],
    }.get(kind, [])
    if kind == "fit" and value["result"]["checkpoint"] not in paths:
        raise ValueError("Screen fit checkpoint is not durably inventoried")
    if any(name not in paths for name in required):
        raise ValueError("Screen task lacks a required durable output")
    return value


def _cache(spec, name, role, coordinate_name="U100"):
    if name == CONTEXT:
        root = Path(spec["bottleneck_root"])
    else:
        row = next(row for row in spec["candidates"] if row["candidate"] == name)
        root = Path(row["foundation_root"])
    foundation = load_json(root / "foundation_spec.json")
    bottleneck = load_json(Path(spec["bottleneck_root"]) / "foundation_spec.json")
    profile = _template(Path(spec["resource_template_path"]), bottleneck)
    kwargs = dict(spec=foundation, data_root=Path(spec["data_root"]), foundation_root=root,
                  role=role, coordinate_name=coordinate_name, workers=profile["workers"],
                  max_ram_bytes=profile["cache_budgets"][role])
    prepare = prepare_bottleneck_cache if name == CONTEXT else prepare_salience_cache
    return prepare(**kwargs)


def _node(name: str) -> dict:
    return dict(node_id="SCREEN_" + name, coordinate="U100", teacher=None,
                branch="SALIENCE_SCREEN", u=[1, 1], f=[0, 1],
                initialization_seed=paired_seed("U100", "initialization"),
                sampler_seed=paired_seed("U100", "sampler"), deployable=False)


def _profile(spec: dict) -> dict:
    value = load_json(Path(spec["screen_root"]) / "runtime_profile.json")
    version = value.get("schema_version")
    if version not in {1, 2}:
        raise ValueError("Unsupported salience runtime profile schema")
    validate(value, "SALIENCE_RUNTIME_PROFILE", version=version)
    if value["screen_sha256"] != spec["content_hash"] or value["passed"] is not True:
        raise ValueError("Salience runtime profile differs")
    if spec.get("schema_version") == 2 and (
        version != 2
        or value.get("screen_execution_site") != spec["screen_execution_site"]
        or value.get("execution_site") != spec["production_execution_site"]
        or value.get("execution_policy") != DEBUG_SCREEN_POLICY
    ):
        raise ValueError("Debug salience runtime profile differs")
    return value


def _screen_execution_site(spec: dict, template: dict) -> dict:
    if spec.get("schema_version") == 2:
        if template["execution_site"] != spec["production_execution_site"]:
            raise ValueError("Debug screen production site differs from its measured template")
        return spec["screen_execution_site"]
    return template["execution_site"]


def run_task(spec: dict, task_id: str, *, attempt: str, device="cuda") -> dict:
    validate_screen(spec)
    tasks = {row["task_id"]: row for row in task_graph()}
    if task_id not in tasks or re.fullmatch(r"[A-Za-z0-9_-]+", attempt) is None:
        raise ValueError("Unknown task or unsafe attempt")
    existing = _task_report(spec, task_id)
    if existing is not None:
        return existing
    task = tasks[task_id]
    for parent in task["dependencies"]:
        if _task_report(spec, parent) is None:
            raise ValueError("Missing authenticated screen parent: " + parent)
    root = Path(spec["screen_root"])
    attempt_root = root / "attempts" / task_id / attempt
    attempt_root.mkdir(parents=True, exist_ok=False)
    outputs = []
    if task["kind"] == "authenticate":
        _load_foundations(spec, deep=True)
        result = artifact("SALIENCE_SCREEN_AUTH", screen_sha256=spec["content_hash"],
                          passed=True, final_test_accessed=False)
        path = attempt_root / "authentication.json"; write_immutable_json(path, result); outputs.append(path)
    elif task["kind"] == "preflight":
        bottleneck = load_json(Path(spec["bottleneck_root"]) / "foundation_spec.json")
        template = _template(Path(spec["resource_template_path"]), bottleneck)
        screen_site = _screen_execution_site(spec, template)
        job, cpus, memory = allocation(screen_site)
        if (cpus, memory) != (template["cpus"], template["memory_mb"]):
            raise ValueError("Screen allocation differs from measured template")
        identities = labels = None; cache_bytes = 0; timings = {}
        for name in [CONTEXT] + REGISTRY:
            started = time.monotonic()
            probe_coordinate = "U100" if name == CONTEXT else "U000"
            train = _cache(spec, name, "train", probe_coordinate)
            validation = _cache(spec, name, "validation", probe_coordinate)
            timings[name] = time.monotonic() - started
            cache_bytes = max(cache_bytes, train.nbytes + validation.nbytes)
            if identities is None:
                identities, labels = validation.identities, validation.labels
            elif not np.array_equal(identities, validation.identities) or not np.array_equal(labels, validation.labels):
                raise ValueError("Candidate validation populations differ")
            model = DelphesParticleTransformer().to(device).eval()
            with torch.inference_mode():
                raw = validation.batch(np.arange(min(256, len(validation))))
                logits = model(**{k: torch.from_numpy(raw[k]).to(device) for k in ("features", "vectors", "mask")})
                if logits.shape != (len(raw["labels"]), 11) or not torch.isfinite(logits).all():
                    raise ValueError("Candidate worst-support preflight is nonfinite")
            del train, validation, model, logits
        checkpoint, selection = _selection_indices(labels, identities)
        split = artifact("SALIENCE_SCREEN_SPLIT", screen_sha256=spec["content_hash"],
                         checkpoint_rows=len(checkpoint), selection_rows=len(selection),
                         checkpoint_indices=checkpoint.tolist(), selection_indices=selection.tolist(),
                         final_test_accessed=False)
        split_path = root / "screen_split.json"; write_immutable_json(split_path, split); outputs.append(split_path)
        debug = spec.get("schema_version") == 2
        execution_fields = ({
            "screen_execution_site": screen_site,
            "execution_policy": DEBUG_SCREEN_POLICY,
        } if debug else {})
        result = artifact("SALIENCE_RUNTIME_PROFILE", version=2 if debug else 1,
                          screen_sha256=spec["content_hash"],
                          source_commit=spec["source_commit"], execution_site=template["execution_site"],
                          slurm_job_id=job, cpus=cpus, memory_mb=memory, workers=template["workers"],
                          train_minutes=template["train_minutes"], reduce_minutes=template["reduce_minutes"],
                          max_train_minutes=template["max_train_minutes"],
                          cache_budgets=template["cache_budgets"], gpu=gpu_identity(),
                          selected_state_bytes=template["selected_state_bytes"],
                          installed_environment=installed_environment(), cache_seconds_by_candidate=timings,
                          probe_coordinates={name: ("U100" if name == CONTEXT else "U000")
                                             for name in [CONTEXT] + REGISTRY},
                          peak_train_plus_validation_cache_bytes=cache_bytes, model=model_contract(), passed=True,
                          ram_only_views=True, rolling_resume=False, final_test_accessed=False,
                          **execution_fields)
        path = root / "runtime_profile.json"; write_immutable_json(path, result); outputs.append(path)
    elif task["kind"] == "fit":
        profile = _profile(spec)
        _, cpus, memory = allocation(profile.get("screen_execution_site", profile["execution_site"]))
        if ((cpus, memory) != (profile["cpus"], profile["memory_mb"])
                or gpu_identity() != profile["gpu"]
                or installed_environment() != profile["installed_environment"]):
            raise ValueError("Screen fit environment differs from preflight")
        name = task["candidate"]
        train, validation = _cache(spec, name, "train"), _cache(spec, name, "validation")
        split = load_json(root / "screen_split.json"); validate(split, "SALIENCE_SCREEN_SPLIT")
        checkpoint = SubsetCache(validation, np.asarray(split["checkpoint_indices"], np.int64))
        selection = SubsetCache(validation, np.asarray(split["selection_indices"], np.int64))
        node = _node(name); torch.manual_seed(node["initialization_seed"])
        model = DelphesParticleTransformer()
        training, state = train_kernel(model, train, checkpoint, node=node, device=device)
        model.load_state_dict(state); probabilities = predict(model, selection, device=device)
        metrics = evaluate_probabilities(selection.labels, probabilities)
        buffer = BytesIO(); torch.save(state, buffer)
        checkpoint_path = attempt_root / "selected.pt"; atomic_publish_bytes(checkpoint_path, buffer.getvalue()); outputs.append(checkpoint_path)
        result = artifact("SALIENCE_SCREEN_FIT", screen_sha256=spec["content_hash"],
                          candidate=name, contextual=name == CONTEXT, node=node,
                          training=training, selection=metrics,
                          selection_rows=len(selection), checkpoint_rows=len(checkpoint),
                          checkpoint=checkpoint_path.relative_to(root).as_posix(),
                          final_test_accessed=False)
        path = attempt_root / "fit_report.json"; write_immutable_json(path, result); outputs.append(path)
    elif task["kind"] == "select":
        fits = {name: _task_report(spec, "fit_" + name)["result"] for name in [CONTEXT] + REGISTRY}
        candidate_fits = [fits[name] for name in REGISTRY]
        best_auc = max(row["selection"]["macro_ovr_auc"] for row in candidate_fits)
        band = [row for row in candidate_fits if best_auc - row["selection"]["macro_ovr_auc"] <= 5e-5]
        foundation_by_name = {row["candidate"]: row for row in spec["candidates"]}
        def key(row):
            metrics = row["selection"]
            foundation_root = Path(foundation_by_name[row["candidate"]]["foundation_root"])
            lock = load_json(foundation_root / "foundation_lock.json")
            log_r50 = metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]
            return (float("-inf") if log_r50 is None else log_r50, metrics["accuracy"],
                    -lock["pt_salience_weighted_selected_dr"], -REGISTRY.index(row["candidate"]))
        winner = max(band, key=key)["candidate"]
        chosen = foundation_by_name[winner]
        result = artifact("SALIENCE_SELECTION_LOCK", screen_sha256=spec["content_hash"],
                          winner=winner, winner_foundation_root=chosen["foundation_root"],
                          winner_foundation_sha256=chosen["foundation_sha256"],
                          candidates=REGISTRY, contextual_control=CONTEXT,
                          auc_tolerance=5e-5, fit_reports=fits,
                          final_test_accessed=False)
        path = root / "selection_lock.json"; write_immutable_json(path, result); outputs.append(path)
    else:
        result = artifact("SALIENCE_SCREEN_COMPLETE", screen_sha256=spec["content_hash"],
                          selection_lock_sha256=load_json(root / "selection_lock.json")["content_hash"],
                          scientific_fit_count=4, final_test_accessed=False)
        path = root / "screen_complete.json"; write_immutable_json(path, result); outputs.append(path)
    report = artifact("SALIENCE_SCREEN_TASK", screen_sha256=spec["content_hash"],
                      source_commit=spec["source_commit"], task_id=task_id, result=result,
                      outputs=[dict(path=p.relative_to(root).as_posix(), sha256=sha256_file(p)) for p in outputs],
                      final_test_accessed=False)
    write_immutable_json(root / "tasks" / f"{task_id}.json", report)
    return report


def command_plan(spec: dict) -> dict:
    validate_screen(spec)
    bottleneck = load_json(Path(spec["bottleneck_root"]) / "foundation_spec.json")
    profile = _template(Path(spec["resource_template_path"]), bottleneck)
    site = _screen_execution_site(spec, profile)
    rows = []
    for task in task_graph():
        gpu = task["kind"] in {"preflight", "fit"}
        walltime = (spec["debug_walltime_minutes"]
                    if gpu and spec.get("schema_version") == 2
                    else profile["train_minutes"] if gpu else 60)
        command = slurm_options(site) + [
            f"--cpus-per-task={profile['cpus'] if gpu else 1}",
            f"--mem={profile['memory_mb'] if gpu else 8192}M",
            f"--time={walltime}",
            "--job-name=jc2sals_" + task["task_id"], "--chdir=" + spec["project_dir"],
            "--output=" + str(Path(spec["screen_root"]) / "slurm-%j.out"),
        ]
        if gpu: command += ["--gres=" + site["gres"]]
        if task["dependencies"]:
            command += ["--dependency=afterok:" + ":".join("${JOB_" + p + "}" for p in task["dependencies"])]
        command += [str(Path(spec["project_dir"]) / "sbatch/run_jetclass2_delphes_salience_screen.sh"),
                    spec["project_dir"], str(Path(spec["screen_root"]) / "screen_spec.json"),
                    task["task_id"], site["name"]]
        rows.append(dict(task_id=task["task_id"], dependencies=task["dependencies"], command=command))
    return artifact("COMMAND_PLAN", screen_sha256=spec["content_hash"], commands=rows,
                    scientific_fits=4, final_test_accessed=False)


def submit_screen(spec: dict, *, execute: bool, authorization_phrase=None) -> dict:
    validate_screen(spec); root = Path(spec["screen_root"]); plan = command_plan(spec)
    write_immutable_json(root / "command_plan.json", plan)
    if execute and authorization_phrase != AUTHORIZE:
        raise PermissionError("Exact U100-screen authorization phrase required")
    claim = root / "submission_in_progress.claim"
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600); os.close(descriptor)
    try:
        if execute: return _guarded_exact_submission(spec, plan, root)
        return submit_exact_dag(identity=spec["content_hash"], plan=plan,
                                output=root / "dry_run_submission_ledger.json",
                                canonical_dry_run=root / "dry_run_submission_ledger.json", execute=False)
    finally:
        claim.unlink()
