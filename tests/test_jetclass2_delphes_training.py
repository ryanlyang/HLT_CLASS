import numpy as np
import pytest
import torch

from hlt_classification.jetclass2_delphes.cache import RamBlock, RamCache
from hlt_classification.jetclass2_delphes.campaign import learning_rate, paired_seed, coordinate
from hlt_classification.jetclass2_delphes.model import model_config, distillation_loss
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.jetclass2_delphes.runner import train_kernel, predict


class TinyModel(torch.nn.Module):
    """Tests the kernel only. Never represents installed-Weaver parity."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(17, 11)

    def forward(self, features, vectors, mask):
        return self.linear((features * mask).sum(-1) / mask.sum(-1))


def toy_cache(role):
    n, tokens = 22, 2
    labels = np.arange(n, dtype=np.int64) % 11
    values = np.random.default_rng(13).normal(size=(n * tokens, 17)).astype(np.float32)
    identities = np.zeros((n, 32), np.uint8)
    identities[:, 0] = np.arange(n)
    identities[:, 1] = 0 if role == "train" else 1
    block = RamBlock(0, np.arange(n + 1, dtype=np.int64) * tokens, values,
                     np.ones((n * tokens, 4), np.float32), identities, labels)
    return RamCache([block], role=role, foundation_sha256="a" * 64, coordinate_name="D000")


def test_schedule_graph_coordinate_and_new_model_configuration():
    assert learning_rate(3) == 3e-4
    assert learning_rate(45) == 3e-4
    assert learning_rate(60) == pytest.approx(1.5e-5)
    assert learning_rate(100) == 1.5e-5
    assert learning_rate(46) > learning_rate(50) > learning_rate(59)
    assert float(coordinate("D066")[1]) == pytest.approx(1 / 3)
    assert paired_seed("D000", "sampler") != paired_seed("D000", "initialization")
    assert model_config()["num_classes"] == 11
    assert model_config()["input_dim"] == 17
    assert model_config()["trim"] is False


def test_kd_equation_and_validation():
    torch.manual_seed(1)
    x = torch.randn(8, 11, requires_grad=True)
    y = torch.arange(8)
    q = torch.softmax(torch.randn(8, 11), -1)
    expected = .25 * torch.nn.functional.cross_entropy(x, y) + .75 * 4 * torch.nn.functional.kl_div(
        torch.log_softmax(x / 2, -1), q, reduction="batchmean")
    loss = distillation_loss(x, y, teacher_probabilities=q)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(x.grad).all()
    with pytest.raises(ValueError):
        distillation_loss(x, y, teacher_probabilities=q * 2)


def test_metrics_ties_zero_fpr_and_fresh_recovery():
    labels = np.tile(np.arange(11), 4)
    tied = np.full((len(labels), 11), 1 / 11)
    result = evaluate_probabilities(labels, tied)
    assert result["macro_ovr_auc"] == .5
    assert result["macro_r50"] == 1.
    assert result["per_class"]["X_bb"]["achieved_signal_efficiency"] == 1.
    perfect = np.eye(11)[labels]
    oracle = evaluate_probabilities(labels, perfect)
    assert oracle["macro_ovr_auc"] == 1.
    assert oracle["macro_r50"] is None
    assert recovery(oracle, result, oracle)["macro_ovr_auc"] == 100
    assert recovery(result, result, oracle)["macro_r50"] is None
    with pytest.raises(ValueError, match="every registered class"):
        evaluate_probabilities(labels[:2], tied[:2])


def test_cache_batch_and_acceptance_train_restore_and_join(tmp_path):
    torch.set_num_threads(1)
    train, val = toy_cache("train"), toy_cache("validation")
    batch = train.batch(np.array([2, 0, 21]))
    np.testing.assert_array_equal(batch["identities"], train.identities[[2, 0, 21]])
    assert batch["features"].shape == (3, 17, 16)
    assert not batch["features"][:, :, 2:].any()
    model = TinyModel()
    node = dict(node_id="M0HLT", coordinate="D000", teacher=None,
                sampler_seed=paired_seed("D000", "sampler"))
    report, state = train_kernel(model, train, val, node=node, device="cpu", acceptance_passes=2)
    assert report["passes"] == 2 and report["acceptance_only"] and not report["scientific_fit"]
    assert report["selected_weights_restored"] and not report["rolling_resume_written"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name])
    q = predict(model, train, device="cpu", temperature=2.)
    node = {**node, "node_id": "DIRECT", "teacher": "U000"}
    with pytest.raises(ValueError, match="identity join"):
        train_kernel(model, train, val, node=node, device="cpu", acceptance_passes=1,
                     teacher_probabilities=q, teacher_identities=train.identities[::-1])
    kd_report, _ = train_kernel(TinyModel(), train, val, node=node, device="cpu", acceptance_passes=1,
                                teacher_probabilities=q, teacher_identities=train.identities)
    assert kd_report["selected_pass"] == 1
    assert not list(tmp_path.iterdir())  # kernels have no implicit filesystem outputs


def test_probability_bank_lineage_shards_and_corruption(tmp_path):
    from hlt_classification.jetclass2_delphes.banks import publish_bank, load_bank
    cache = toy_cache("train")
    p = np.full((len(cache), 11), 1/11, np.float32)
    parents = dict(foundation_sha256="a" * 64, teacher_report_sha256="b" * 64, teacher_node="U000", role="train")
    report = publish_bank(tmp_path, identities=cache.identities, probabilities=p, shard_rows=5, **parents)
    assert len(report["shards"]) == 5 and report["temperature"] == 2.
    np.testing.assert_array_equal(p, load_bank(tmp_path, expected_identities=cache.identities, **parents))
    with pytest.raises(ValueError, match="lineage"):
        load_bank(tmp_path, expected_identities=cache.identities[::-1], **parents)
    with pytest.raises(ValueError, match="lineage"):
        load_bank(tmp_path, expected_identities=cache.identities, **{**parents, "teacher_node": "wrong_teacher"})
    with pytest.raises(PermissionError):
        load_bank(tmp_path, expected_identities=cache.identities, **{**parents, "role": "final_test"})
    with (tmp_path / report["shards"][0]["path"]).open("ab") as f:
        f.write(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        load_bank(tmp_path, expected_identities=cache.identities, **parents)
