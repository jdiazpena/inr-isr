"""Protect xyzt physical-gradient behavior while sharing only its chunk loop."""
import numpy as np
import torch

from inr_isr_4d.data import AffineScaler
from inr_isr_4d.synthetic_evaluation import predict_log_and_physical_derivatives


def test_xyzt_analytic_first_derivatives_keep_physical_units_mode_and_shape():
    model = torch.nn.Linear(4, 1, bias=False, dtype=torch.float64)
    weights = np.array([0.125, -0.25, 0.375, -0.5])
    with torch.no_grad():
        model.weight.copy_(torch.tensor(weights[None, :], dtype=torch.float64))
    model.train()
    spans = np.array([10.0, 20.0, 30.0, 40.0])
    coordinates = np.arange(20, dtype=np.float64).reshape(5, 4)
    coordinate_scaler = AffineScaler(np.zeros(4), spans)
    target_scaler = AffineScaler(np.array([11.0]), np.array([13.0]))
    prediction, derivatives = predict_log_and_physical_derivatives(
        model, coordinates, coordinate_scaler, target_scaler,
        device=torch.device("cpu"), dtype=torch.float64, chunk_size=3)
    assert not model.training  # xyzt native evaluator, unlike xyt, calls eval().
    assert prediction.shape == (5, 1)
    assert derivatives.shape == (5, 4)
    # Both coordinate and target affine maps use [-1,1], not [0,1].
    normalized = 2.0*coordinates/spans - 1.0
    np.testing.assert_allclose(prediction[:, 0], 12.0 + normalized@weights,
                               rtol=0, atol=1e-14)
    np.testing.assert_allclose(derivatives, np.broadcast_to(2.0*weights/spans, (5, 4)),
                               rtol=0, atol=1e-14)
