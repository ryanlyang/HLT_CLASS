"""Recovery I/O adapter around the unchanged NOISE-K2 scientific kernels."""
from pathlib import Path
import platform

import torch

from hlt_classification.data.cache_contracts import load_json, sha256_file
from . import campaign, data, runtime as science
from .contracts import publish, reference, require, safe
from .recovery import Context, artifact, authenticate_job, validate_recovery


def gpu_smoke(device):
    """Tiny allocation/forward/backward check, not scientific acceptance evidence."""
    require(str(device) in ("cuda", "cuda:0"), "Replacement science requires a real CUDA allocation")
    # Do not consume the fit's random stream or alter global numerical settings.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        x = torch.ones((16, 16), device=device, requires_grad=True)
        weight = torch.ones((16, 16), device=device, requires_grad=True)
        y = (x @ weight).square().mean()
        y.backward()
        torch.cuda.synchronize()
        require(all(bool(torch.isfinite(t).all()) for t in (y, x.grad, weight.grad)),
                "Nonfinite CUDA startup smoke")
    return dict(passed=True, device=str(device), node=platform.node(),
                forward_backward=True, scientific_acceptance=False)


def fields(context):
    return dict(campaign_sha256=context.source["content_hash"],
                recovery_sha256=context.spec["content_hash"])


def fit(context, node, directory, device):
    spec = context.source
    train, validation, population = science.caches(spec, node)
    q, teacher_lineage = None, None
    teacher = node["teacher_distribution"]
    if teacher is not None:
        parent = context.completed("reduce_"+teacher)
        require(parent is not None, "Teacher is incomplete")
        report = context.training_report(teacher)
        result = parent[0]["result"]
        bank_path = context.result_path(parent, "bank")
        require(report and report["content_hash"] == result["training_report_sha256"]
            and load_json(bank_path/"manifest.json")["content_hash"] == result["bank_sha256"],
            "Teacher bank/report receipt differs")
        q = science.load_bank(bank_path, foundation_sha256=train.foundation_sha256,
            teacher_report_sha256=report["content_hash"], teacher_node=teacher,
            role="train", expected_identities=train.identities)
        teacher_lineage = dict(task_sha256=parent[0]["content_hash"],
                               bank_sha256=result["bank_sha256"], teacher=teacher)
    # Keep the same constructor, cold seed, loss, optimizer, batching, selection
    # and reporting calls as the original runtime.fit; only artifact I/O differs.
    model = science.new_model(node)
    report, state = science.train_kernel(model, lambda _: train, lambda _: validation["checkpoint"],
        node=node, device=device, teacher_probabilities=q,
        teacher_identities=None if q is None else train.identities,
        batch_size=spec["training"]["batch_size"], inference_batch_size=spec["inference_batch_size"])
    science.batching(spec, report)
    require(report["scientific_fit"] is True and report["acceptance_only"] is False, "Not a scientific fit")
    p = science.predict(model, validation["report"], node=node, device=device,
                        batch_size=spec["inference_batch_size"])
    checkpoint = directory/"selected.pt"
    science.save_state(checkpoint, state)
    outer = artifact("TRAINING_REPORT", **fields(context), node=node,
        foundation_sha256=train.foundation_sha256, population_sha256=population["content_hash"],
        training=spec["training"], kernel_report=report, teacher_lineage=teacher_lineage,
        selected_checkpoint_sha256=sha256_file(checkpoint), checkpoint_validation=report["validation"],
        report_validation=science.evaluate_probabilities(validation["report"].labels, p),
        report_is_validation_not_final_test=True, deployable=node["deployable"])
    path = directory/"training_report.json"
    publish(path, outer)
    if node["deployable"]:
        from hlt_classification.cms_proxy_ladder.inputs import input_contract
        publish(directory/"deployment.json", artifact("DEPLOYMENT", **fields(context),
            report_sha256=outer["content_hash"], checkpoint="selected.pt", checkpoint_sha256=sha256_file(checkpoint),
            model=spec["model"], inputs=input_contract(capacity=data.get_foundation(spec)["capacity"]),
            physical_proxy_copies=1 if node["primary_coordinate"] == "HLT_X1" else 3,
            preprocessing="hlt_classification.noise_k2.views.deployment_inputs",
            offline_inputs=False, assignment_inputs=False, identity_inputs=False, source_embeddings=False,
            detector_qualification=False, dataset_kind=spec["dataset_kind"]))
    return dict(training_report=path, checkpoint=checkpoint, training_report_sha256=outer["content_hash"])


def reduce(context, node, directory, device):
    spec = context.source
    parent = context.completed("train_"+node["node_id"])
    report = context.training_report(node["node_id"])
    require(parent is not None and report is not None, "Missing teacher fit")
    model = science.new_model(node)
    model.load_state_dict(torch.load(context.result_path(parent, "checkpoint"),
                                    map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    train = data.cache(spec, "train", node["primary_coordinate"])
    p = science.predict(model, train, node=node, device=device, temperature=2.,
                        batch_size=spec["inference_batch_size"])
    path = directory/"train_bank"
    bank = science.publish_bank(path, foundation_sha256=train.foundation_sha256,
        teacher_report_sha256=report["content_hash"], teacher_node=node["node_id"],
        role="train", identities=train.identities, probabilities=p)
    return dict(bank=path, bank_sha256=bank["content_hash"], training_report_sha256=report["content_hash"])


def result_rows(context):
    reports = {n["node_id"]: context.training_report(n["node_id"]) for n in context.source["nodes"]}
    baseline, oracle = reports.get("HLT_X1_CE"), reports.get("OFFLINE_CE")
    rows = []
    for node in context.source["nodes"]:
        report = reports[node["node_id"]]
        metrics = None if report is None else report["report_validation"]
        recovered = None
        if metrics is not None and baseline is not None and oracle is not None:
            recovered = science.recovery(metrics, baseline["report_validation"], oracle["report_validation"])
        rows.append(dict(node_id=node["node_id"], deployable=node["deployable"], validation=metrics,
            recovery=recovered, provenance="original" if "train_"+node["node_id"] in context.spec["reused_tasks"] else "replacement",
            selected_pass=None if report is None else report["kernel_report"]["selected_pass"],
            passes=None if report is None else report["kernel_report"]["passes"]))
    return rows


def run_task(spec, name, *, device="cuda"):
    source = validate_recovery(spec)
    require(name in spec["retry_tasks"], "Original completed tasks are read-only, not restart targets")
    context = Context(spec, source)
    job = authenticate_job(spec, source, name)
    old = context.completed(name)
    if old is not None:
        return old[0]
    row = campaign.task(source, name)
    parents = {p: context.completed(p) for p in row["dependencies"]}
    require(all(parents.values()), "Missing/corrupt recovery parent outputs")
    root = context.root
    directory = safe(root, "outputs/"+name)
    # Never overwrite a partial replacement either. Another failed attempt
    # requires explicit diagnosis/registration, not an implicit restart.
    directory.mkdir(parents=True, exist_ok=False)
    try:
        if row["kind"] in ("train", "reduce"):
            acceptance = science.science_gate(source)
            require(acceptance["gpu"] == science.gpu_identity()
                and acceptance["environment"] == science.installed_environment(),
                "GPU/software differs from accepted preflight")
            health = gpu_smoke(device)  # Deliberately before caches()/data.cache().
            publish(directory/"gpu_startup.json", artifact("GPU_STARTUP", **fields(context),
                task_id=name, job_id=job, evidence=health, gpu=acceptance["gpu"]))
            node = science.node_for(source, row["node_id"])
            result = (fit if row["kind"] == "train" else reduce)(context, node, directory, device)
        elif row["kind"] == "aggregate":
            rows = result_rows(context)
            require(all(r["validation"] is not None for r in rows), "Missing registered result")
            result = dict(rows=rows, recovery_reference="HLT_X1_CE=0%, OFFLINE_CE=100%",
                          subset="validation_report")
        elif row["kind"] == "complete":
            result = dict(total_scientific_fits=10, total_reducers=5,
                reused_science_tasks=sum(campaign.task(source, p)["kind"] in campaign.SCIENCE
                                        for p in spec["reused_tasks"]),
                replacement_tasks=len(spec["retry_tasks"]), final_test_sealed=True)
        else:
            raise ValueError("Recovery cannot rerun preparation or gates")
        result = {k: v.relative_to(root).as_posix() if isinstance(v, Path) else v for k, v in result.items()}
        publish(directory/"result.json", artifact("RESULT", **fields(context), task_id=name, result=result))
        receipt = artifact("TASK", **fields(context), source_commit=spec["source_commit"],
            task_id=name, job_id=job, parents={p: r[0]["content_hash"] for p, r in parents.items()},
            result=result, outputs=[reference(root, p) for p in sorted(directory.rglob("*")) if p.is_file()])
        return publish(safe(root, "tasks/"+name+".json"), receipt)
    except Exception as error:
        publish(directory/"failure.json", artifact("FAILURE", **fields(context), task_id=name, job_id=job,
            error_type=type(error).__name__, message=str(error)))
        raise
    finally:
        science.clear()
