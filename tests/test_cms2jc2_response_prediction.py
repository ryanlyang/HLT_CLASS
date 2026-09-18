import numpy as np
import pytest

from hlt_classification.cms2jc2_response.families import fit_conditional, predict, Predictor
from hlt_classification.cms2jc2_response.features import features
from hlt_classification.cms2jc2_response.contracts import validate
from test_cms2jc2_response_science import particles


@pytest.mark.parametrize("candidate", ["A_H", "B_H", "C_H"])
@pytest.mark.parametrize("task", ["continuous", "categorical"])
def test_prepared_prediction_preserves_registered_numerics_and_artifact(candidate, task):
    x = np.repeat(features(particles())[:1], 3200, axis=0)
    x[:, 2] = np.linspace(1., 4., len(x))
    x[:, 3] = np.abs(np.sin(x[:, 2]))
    x[:, 17] = np.arange(len(x)) % 8
    x[:, 18] = np.log1p(x[:, 17])
    y = (np.column_stack((.1*x[:, 2], -.2*x[:, 3])) if task == "continuous" else
         (np.arange(len(x)) > len(x)//2).astype(np.int64))
    fitted = fit_conditional(x, y, np.ones(len(x)), candidate_id=candidate, task=task,
                            classes=2 if task == "categorical" else None, membership_hash="a"*64)
    before = fitted["content_hash"]
    prepared = Predictor(fitted)
    held = x[::61]
    for probabilities in (False, True):
        np.testing.assert_array_equal(prepared(held, probabilities=probabilities),
                                      predict(held, fitted, probabilities=probabilities))
        np.testing.assert_array_equal(prepared(held[:1], probabilities=probabilities),
                                      predict(held[:1], fitted, probabilities=probabilities))
    assert prepared(held[:0]).shape == (0, fitted["output_dim"])
    assert validate(fitted, "CONDITIONAL_MODEL") == before
