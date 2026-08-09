from __future__ import annotations

from pathlib import Path

import pytest

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    sha256_file,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.scouting.hcwdl_representation_runtime_adapters import (
    ProductionConfigurationError,
    resolve_registered_arguments,
)
from hlt_classification.scouting.hcwdl_representation_runtime_binding import (
    CAMPAIGN_ARTIFACT_BINDING, IMMUTABLE_OUTPUT_ROOT_BINDING,
    build_runtime_binding, runtime_campaign_identity,
    resolve_runtime_row,
    validate_runtime_binding,
)
from hlt_classification.scouting.hcwdl_representation_campaign import (
    build_command_plan, create_campaign_spec, materialize_command,
)
from hlt_classification.scouting.hcwdl_representation_contracts import (
    SHUFFLE_MAP_CONTRACT, SUBMISSION_LEDGER_CONTRACT,
)
from hlt_classification.scouting import hcwdl_representation_task_runtime as task_runtime
from hlt_classification.scouting.hcwdl_representation_artifacts import (
    publish_binary_envelope,
)


def _artifact(name: str):
    return with_content_hash({
        "contract": f"TEST_{name.upper()}/v1",
        "schema_version": 1,
        "name": name,
    })


def _directory_sha256(path: Path) -> str:
    rows = [
        {
            "path": member.relative_to(path).as_posix(),
            "bytes": member.stat().st_size,
            "sha256": sha256_file(member),
        }
        for member in sorted(path.rglob("*"))
        if member.is_file()
    ]
    return canonical_sha256(rows)


def _binding_fixture(*, producer: str = "producer", frozen_path: str | None = None):
    tasks = [
        {
            "task_key": "producer", "kind": "tap_schema", "dependencies": [],
            "resource_class": "cpu_small", "registered_inputs": [],
            "registered_outputs": ["artifacts/producer.json"],
        },
        {
            "task_key": "unrelated", "kind": "tap_schema", "dependencies": [],
            "resource_class": "cpu_small", "registered_inputs": [],
            "registered_outputs": ["artifacts/unrelated.json"],
        },
        {
            "task_key": "consumer", "kind": "parent_import",
            "dependencies": ["producer"], "resource_class": "cpu_small",
            "registered_inputs": ["${producer}"],
            "registered_outputs": ["artifacts/consumer.json"],
        },
    ]
    spec = {
        "project_dir": "/project", "source_commit": "1" * 40, "tasks": tasks,
    }
    runtime_facts = {
        "conda_environment": "atlas_kd_tigris", "data_root": "/data",
        "device": "cpu", "project_dir": "/project",
        "python_no_user_site": True, "source_snapshot_sha256": "2" * 64,
        "weaver_runtime_sha256": "3" * 64,
    }

    def row(inputs, output):
        return {
            "array_index": None, "device": "cpu", "inputs": inputs,
            "outputs": output, "parameters": {},
            "runtime_signature_sha256": "4" * 64,
        }

    upstream = {
        "task_key": producer, "array_index": None,
        "registered_output": f"artifacts/{producer}.json",
        "artifact_kind": "json", "expected_contract": "TEST_FUTURE/v1",
        "expected_schema_version": 1,
    }
    reference = {"upstream_output": upstream}
    if frozen_path is not None:
        reference["path"] = frozen_path
    task_rows = {
        "producer": {"single": row({}, {
            "artifacts/producer.json": "/campaign/artifacts/producer.json",
        })},
        "unrelated": {"single": row({}, {
            "artifacts/unrelated.json": "/campaign/artifacts/unrelated.json",
        })},
        "consumer": {"single": row({"${producer}": reference}, {
            "artifacts/consumer.json": "/campaign/artifacts/consumer.json",
        })},
    }
    return spec, runtime_facts, task_rows


def test_registered_argument_tags_resolve_only_authenticated_input_bytes(
    tmp_path: Path,
) -> None:
    artifact = _artifact("direct")
    path = tmp_path / "direct.json"
    write_immutable_json(path, artifact)

    bundle = tmp_path / "bundle"
    member_artifact = _artifact("member")
    member_path = bundle / "nested" / "manifest.json"
    write_immutable_json(member_path, member_artifact)
    unrelated = bundle / "payload.bin"
    unrelated.write_bytes(b"authenticated-payload")
    runtime_row = {
        "inputs": {
            "${direct}": {"path": str(path), "sha256": sha256_file(path)},
            "${bundle}": {
                "path": str(bundle),
                "sha256": _directory_sha256(bundle),
            },
        }
    }

    assert resolve_registered_arguments(
        {"registered_json": "${direct}"}, runtime_row,
    ) == artifact
    assert resolve_registered_arguments(
        {"registered_reference": "${direct}"}, runtime_row,
    ) == {"path": str(path), "sha256": sha256_file(path)}
    assert resolve_registered_arguments(
        {"registered_path": "${direct}"}, runtime_row,
    ) == str(path)
    assert resolve_registered_arguments({
        "registered_member": {
            "input": "${bundle}", "relative": "nested/manifest.json",
        },
    }, runtime_row) == str(member_path)
    assert resolve_registered_arguments({
        "registered_member": {
            "input": "${bundle}", "relative": "nested/manifest.json",
            "mode": "reference",
        },
    }, runtime_row) == {"path": str(member_path), "sha256": sha256_file(member_path)}
    assert resolve_registered_arguments({
        "registered_member": {
            "input": "${bundle}", "relative": "nested/manifest.json",
            "mode": "json",
        },
    }, runtime_row) == member_artifact
    assert resolve_registered_arguments({
        "registered_field": {
            "input": "${direct}", "field": ["content_hash"],
        },
    }, runtime_row) == artifact["content_hash"]
    assert resolve_registered_arguments({
        "registered_field": {
            "input": "${bundle}", "relative": "nested/manifest.json",
            "field": ["name"],
        },
    }, runtime_row) == "member"

    unrelated.write_bytes(b"mutated-unrelated-payload")
    with pytest.raises(ValueError, match="byte/hash identity differs"):
        resolve_registered_arguments({
            "registered_member": {
                "input": "${bundle}", "relative": "nested/manifest.json",
                "mode": "json",
            },
        }, runtime_row)


def test_runtime_resolver_rejects_inline_artifacts_and_unsafe_members(
    tmp_path: Path,
) -> None:
    artifact = _artifact("registered")
    path = tmp_path / "registered.json"
    write_immutable_json(path, artifact)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    member = bundle / "manifest.json"
    write_immutable_json(member, artifact)
    runtime_row = {
        "inputs": {
            "${artifact}": {"path": str(path), "sha256": sha256_file(path)},
            "${bundle}": {
                "path": str(bundle), "sha256": _directory_sha256(bundle),
            },
        }
    }

    with pytest.raises(PermissionError, match="raw artifact reference"):
        resolve_registered_arguments(
            {"path": str(path), "sha256": sha256_file(path)}, runtime_row,
        )
    with pytest.raises(PermissionError, match="inline self-hashed artifact"):
        resolve_registered_arguments(dict(artifact), runtime_row)
    with pytest.raises(ProductionConfigurationError, match="unregistered input"):
        resolve_registered_arguments(
            {"registered_json": "${missing}"}, runtime_row,
        )
    with pytest.raises(PermissionError, match="path is unsafe"):
        resolve_registered_arguments({
            "registered_member": {
                "input": "${bundle}", "relative": "../registered.json",
            },
        }, runtime_row)
    with pytest.raises(ProductionConfigurationError, match="tag fields differ"):
        resolve_registered_arguments({
            "registered_json": "${artifact}", "fallback": dict(artifact),
        }, runtime_row)

    path.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="byte/hash identity differs"):
        resolve_registered_arguments(
            {"registered_json": "${artifact}"}, runtime_row,
        )


def test_registered_json_rejects_corrupt_declared_content_hash(tmp_path: Path) -> None:
    artifact = _artifact("corrupt")
    path = tmp_path / "corrupt.json"
    write_immutable_json(path, {**artifact, "name": "changed"})
    runtime_row = {
        "inputs": {
            "${corrupt}": {"path": str(path), "sha256": sha256_file(path)},
        }
    }
    with pytest.raises(ValueError, match="content hash"):
        resolve_registered_arguments(
            {"registered_json": "${corrupt}"}, runtime_row,
        )


def test_runtime_binding_late_binds_exact_transitive_producer_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, runtime_facts, task_rows = _binding_fixture()
    # The producer path deliberately does not exist when the complete binding
    # is frozen.  No fake future SHA is required.
    binding = build_runtime_binding(
        spec=spec, runtime_facts=runtime_facts, task_rows=task_rows,
    )
    validate_runtime_binding(binding, spec=spec)
    consumer = resolve_runtime_row(
        binding, spec=spec, task_key="consumer", array_index=None,
    )
    reference = consumer["inputs"]["${producer}"]
    assert reference["path"] == "/campaign/artifacts/producer.json"
    assert "sha256" not in reference

    published = tmp_path / "producer.json"
    artifact = with_content_hash({
        "contract": "TEST_FUTURE/v1", "schema_version": 1, "published": True,
    })
    write_immutable_json(published, artifact)
    real_path = Path
    monkeypatch.setattr(
        task_runtime, "Path",
        lambda value: published
        if str(value) == "/campaign/artifacts/producer.json"
        else real_path(value),
    )
    materialized = task_runtime._validate_input_bytes(
        {"inputs": {"${producer}": reference}}, spec={},
    )
    assert materialized == {
        "${producer}": {"path": str(published), "sha256": sha256_file(published)},
    }

    published.write_text(
        published.read_text(encoding="utf-8").replace("true", "false"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="content hash"):
        task_runtime._validate_input_bytes(
            {"inputs": {"${producer}": reference}}, spec={},
        )


def test_late_bound_route_rejects_wrong_path_or_nonancestor_producer() -> None:
    spec, runtime_facts, task_rows = _binding_fixture(
        frozen_path="/campaign/artifacts/wrong.json",
    )
    with pytest.raises(PermissionError, match="exact producer output"):
        build_runtime_binding(
            spec=spec, runtime_facts=runtime_facts, task_rows=task_rows,
        )
    spec, runtime_facts, task_rows = _binding_fixture(producer="unrelated")
    with pytest.raises(PermissionError, match="not a transitive dependency"):
        build_runtime_binding(
            spec=spec, runtime_facts=runtime_facts, task_rows=task_rows,
        )


def _envelope_binding_fixture():
    logical = "artifacts/bundle/committed/${envelope_id}"
    tasks = [
        {
            "task_key": "producer", "kind": "kernel_resources", "dependencies": [],
            "resource_class": "cpu_small", "registered_inputs": [],
            "registered_outputs": [logical],
        },
        {
            "task_key": "consumer", "kind": "numerical_acceptance",
            "dependencies": ["producer"], "resource_class": "cpu_small",
            "registered_inputs": ["${bundle}"],
            "registered_outputs": ["artifacts/result.json"],
        },
    ]
    spec = with_content_hash({
        "project_dir": "/project", "source_commit": "1" * 40, "tasks": tasks,
    })
    facts = {
        "conda_environment": "atlas_kd_tigris", "data_root": "/data",
        "device": "cpu", "project_dir": "/project", "python_no_user_site": True,
        "source_snapshot_sha256": "2" * 64, "weaver_runtime_sha256": "3" * 64,
    }
    descriptor = {
        "root": "/campaign/artifacts/bundle",
        "artifact_contract": SHUFFLE_MAP_CONTRACT,
        "producer_task_id": "producer-task",
        "registered_output_row": {"route": "bundle"},
        "campaign_identity_sha256": canonical_sha256(
            runtime_campaign_identity(spec),
        ),
        "expected_publication_owner": {
            "campaign_task": "producer", "array_index": None,
            "owner_kind": "initial_campaign",
            "campaign_identity_sha256": canonical_sha256(
                runtime_campaign_identity(spec),
            ),
        },
        "expected_parent_sources": {
            "parent": {"source_kind": "literal_sha256", "sha256": "a" * 64},
        },
    }
    upstream = {
        "task_key": "producer", "array_index": None, "registered_output": logical,
        "artifact_kind": "directory", "expected_contract": None,
        "expected_schema_version": None,
    }

    def row(inputs, outputs):
        return {
            "array_index": None, "device": "cpu", "inputs": inputs,
            "outputs": outputs, "parameters": {},
            "runtime_signature_sha256": "4" * 64,
        }

    rows = {
        "producer": {"single": row({}, {
            logical: {IMMUTABLE_OUTPUT_ROOT_BINDING: descriptor},
        })},
        "consumer": {"single": row({"${bundle}": {"upstream_output": upstream}}, {
            "artifacts/result.json": "/campaign/artifacts/result.json",
        })},
    }
    return spec, facts, rows, logical, descriptor


def test_deferred_envelope_root_builds_before_publication_and_resolves_exact_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, facts, rows, logical, descriptor = _envelope_binding_fixture()
    binding = build_runtime_binding(spec=spec, runtime_facts=facts, task_rows=rows)
    consumer = resolve_runtime_row(
        binding, spec=spec, task_key="consumer", array_index=None,
    )
    reference = consumer["inputs"]["${bundle}"]
    assert "path" not in reference
    assert reference[IMMUTABLE_OUTPUT_ROOT_BINDING]["registered_route"] == {
        "task_key": "producer", "array_index": None, "registered_output": logical,
    }

    root = tmp_path / "bundle"
    envelope = publish_binary_envelope(
        root, artifact_contract=descriptor["artifact_contract"],
        producer_task_id=descriptor["producer_task_id"],
        schema={"kind": "test"}, immutable_parent_hashes={"parent": "a" * 64},
        registered_output_row=descriptor["registered_output_row"],
        campaign_or_recovery_owner=descriptor["expected_publication_owner"],
        payloads={"payload.bin": b"payload"},
        member_metadata={"payload.bin": {"logical_sha256": "c" * 64}},
        sidecar_payload={"test": True},
    )
    real_path = Path
    monkeypatch.setattr(
        task_runtime, "Path",
        lambda value: root
        if str(value) == descriptor["root"] else real_path(value),
    )
    materialized = task_runtime._validate_input_bytes(
        {"inputs": {"${bundle}": reference}}, spec=spec,
    )
    assert materialized["${bundle}"]["path"] == str(envelope.directory)
    assert materialized["${bundle}"]["sha256"] == _directory_sha256(envelope.directory)

    payload_path = envelope.directory / "payload.bin"
    original_payload = payload_path.read_bytes()
    payload_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="member bytes differ"):
        task_runtime._validate_input_bytes(
            {"inputs": {"${bundle}": reference}}, spec=spec,
        )
    payload_path.write_bytes(original_payload)

    publish_binary_envelope(
        root, artifact_contract=descriptor["artifact_contract"],
        producer_task_id="second-task", schema={"kind": "test"},
        immutable_parent_hashes={"parent": "d" * 64},
        registered_output_row={"route": "second"},
        campaign_or_recovery_owner=descriptor["expected_publication_owner"],
        payloads={"payload.bin": b"second"},
        member_metadata={"payload.bin": {"logical_sha256": "e" * 64}},
        sidecar_payload={"test": True},
    )
    with pytest.raises(ValueError, match="exactly one committed envelope"):
        task_runtime._validate_input_bytes(
            {"inputs": {"${bundle}": reference}}, spec=spec,
        )


def test_deferred_envelope_rejects_consumer_declared_wrong_root() -> None:
    spec, facts, rows, logical, descriptor = _envelope_binding_fixture()
    wrong = {
        **descriptor, "root": "/campaign/artifacts/wrong",
        "template": "committed/${envelope_id}",
        "registered_route": {
            "task_key": "producer", "array_index": None,
            "registered_output": logical,
        },
    }
    rows["consumer"]["single"]["inputs"]["${bundle}"][
        IMMUTABLE_OUTPUT_ROOT_BINDING
    ] = wrong
    with pytest.raises(PermissionError, match="immutable root differs"):
        build_runtime_binding(spec=spec, runtime_facts=facts, task_rows=rows)


def test_deferred_envelope_rejects_self_consistent_wrong_parents_campaign_and_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, facts, rows, _logical, descriptor = _envelope_binding_fixture()
    binding = build_runtime_binding(spec=spec, runtime_facts=facts, task_rows=rows)
    reference = resolve_runtime_row(
        binding, spec=spec, task_key="consumer", array_index=None,
    )["inputs"]["${bundle}"]
    root = tmp_path / "bundle"
    publish_binary_envelope(
        root, artifact_contract=descriptor["artifact_contract"],
        producer_task_id=descriptor["producer_task_id"], schema={"kind": "test"},
        immutable_parent_hashes={"parent": "f" * 64},
        registered_output_row=descriptor["registered_output_row"],
        campaign_or_recovery_owner=descriptor["expected_publication_owner"],
        payloads={"payload.bin": b"payload"},
        member_metadata={"payload.bin": {"logical_sha256": "c" * 64}},
        sidecar_payload={"test": True},
    )
    real_path = Path
    monkeypatch.setattr(
        task_runtime, "Path",
        lambda value: root if str(value) == descriptor["root"] else real_path(value),
    )
    with pytest.raises(ValueError, match="parent lineage"):
        task_runtime._validate_input_bytes(
            {"inputs": {"${bundle}": reference}}, spec=spec,
        )

    changed = {
        **descriptor,
        "campaign_identity_sha256": "e" * 64,
        "expected_publication_owner": {
            **descriptor["expected_publication_owner"],
            "campaign_identity_sha256": "e" * 64,
        },
    }
    rows["producer"]["single"]["outputs"][
        "artifacts/bundle/committed/${envelope_id}"
    ][IMMUTABLE_OUTPUT_ROOT_BINDING] = changed
    with pytest.raises(PermissionError, match="campaign lineage"):
        build_runtime_binding(spec=spec, runtime_facts=facts, task_rows=rows)


def test_submission_ledger_late_binding_validates_campaign_and_plan_lineage(
    tmp_path: Path,
) -> None:
    spec = create_campaign_spec(
        mode="pilot", campaign_root=tmp_path,
        checkpoint_namespace=tmp_path / "checkpoints", project_dir="/project",
        source_commit="1" * 40, source_manifest_sha256="2" * 64,
        split_manifest_sha256="3" * 64, parent_import_sha256="4" * 64,
        representation_recipe_sha256="5" * 64, graph_sha256="6" * 64,
        disposition_sha256="7" * 64, disposition="combined_confirmatory",
        role_counts={"train": 300_000, "validation": 100_000, "final_test": 100_000},
        final_source_partitions=2, combined_finalist_count=3,
    )
    plan = build_command_plan(spec)
    reference = {
        CAMPAIGN_ARTIFACT_BINDING: {
            "artifact_role": "submission_ledger",
            "expected_contract": SUBMISSION_LEDGER_CONTRACT,
            "expected_schema_version": 1,
            "campaign_identity_sha256": canonical_sha256(runtime_campaign_identity(spec)),
            "command_plan_sha256": plan["content_hash"],
        },
        "path": str(Path(str(spec["campaign_root"])) / "submission_ledger.json"),
    }
    with pytest.raises(ValueError, match="late-bound path|absent"):
        task_runtime._validate_input_bytes(
            {"inputs": {"${submission_ledger}": reference}}, spec=spec,
        )
    jobs = {}
    rows = []
    for index, template in enumerate(plan["commands"]):
        job_id = str(10_000 + index)
        command = materialize_command(template, job_ids=jobs)
        jobs[template["task_key"]] = job_id
        rows.append({
            "task_key": template["task_key"], "job_id": job_id,
            "command": command,
        })
    ledger = with_content_hash({
        "contract": SUBMISSION_LEDGER_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"],
        "command_plan_sha256": plan["content_hash"],
        "jobs": jobs, "materialized_commands": rows,
    })
    ledger_path = Path(reference["path"])
    write_immutable_json(ledger_path, ledger)
    materialized = task_runtime._validate_input_bytes(
        {"inputs": {"${submission_ledger}": reference}}, spec=spec,
    )
    assert materialized["${submission_ledger}"]["sha256"] == sha256_file(ledger_path)

    wrong = with_content_hash({
        **{key: value for key, value in ledger.items() if key != "content_hash"},
        "campaign_spec_sha256": "f" * 64,
    })
    ledger_path.unlink()
    write_immutable_json(ledger_path, wrong)
    with pytest.raises(ValueError, match="parent lineage"):
        task_runtime._validate_input_bytes(
            {"inputs": {"${submission_ledger}": reference}}, spec=spec,
        )
