import numpy as np

from hlt_classification.scouting.hcwdl_representation_controls import (
    apply_representation_shuffle,
    build_within_class_shuffle_map,
    publish_within_class_shuffle_map,
    validate_within_class_shuffle_map,
)


def test_within_class_shuffle_is_deterministic_derangement_and_rep_only(tmp_path) -> None:
    labels = np.repeat(np.arange(15), 2)
    identities = [f"{index:064x}" for index in range(len(labels))]
    artifact, mapping = build_within_class_shuffle_map(
        identity_sha256=identities, labels=labels,
        split_manifest_sha256="a" * 64, row_selection_sha256="b" * 64,
        parent_hashes={"recipe": "c" * 64},
    )
    validate_within_class_shuffle_map(
        artifact, mapping, identity_sha256=identities, labels=labels,
    )
    assert not np.any(mapping == np.arange(len(mapping)))
    assert mapping.dtype == np.uint32
    assert np.array_equal(labels, labels[mapping])
    artifact_two, mapping_two = build_within_class_shuffle_map(
        identity_sha256=identities, labels=labels,
        split_manifest_sha256="a" * 64, row_selection_sha256="b" * 64,
        parent_hashes={"recipe": "c" * 64},
    )
    assert artifact == artifact_two
    assert np.array_equal(mapping, mapping_two)

    arrays = {
        "logits": np.arange(len(labels) * 15).reshape(len(labels), 15),
        "identity_digest": np.arange(len(labels))[:, None],
        "jet_penultimate": np.arange(len(labels))[:, None] + 100,
    }
    shuffled = apply_representation_shuffle(arrays, mapping)
    assert np.array_equal(shuffled["logits"], arrays["logits"])
    assert np.array_equal(shuffled["identity_digest"], arrays["identity_digest"])
    assert np.array_equal(shuffled["jet_penultimate"], arrays["jet_penultimate"][mapping])

    envelope = publish_within_class_shuffle_map(
        tmp_path,
        artifact=artifact,
        mapping=mapping,
        producer_task_id="build_shuffle_map",
        registered_output_row={"control": "shared_m5_shuffle"},
        campaign_or_recovery_owner={"campaign": "pilot"},
    )
    assert (envelope.directory / "shuffle_map.npz").is_file()
    assert envelope.sidecar["payload"]["source_shuffle_artifact_sha256"] == artifact["content_hash"]
