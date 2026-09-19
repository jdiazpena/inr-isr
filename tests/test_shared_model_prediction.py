"""Protect R3 sharing, named-mode selection and native compatibility policies.

Portable unit checks supplement, not replace, SHA-bound real-radar/old-checkpoint
characterization and unchanged four-workflow regression runs. No science changes.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from models import MLPINR
from inr_isr.model import build_model, restore_parameters
from inr_isr.transforms import affine_forward, affine_inverse
from inr_isr.prediction import predict_normalized
from inr_isr.checkpoints import torch_load_checkpoint, build_model_from_checkpoint
from inr_isr_4d.data import AffineScaler
from inr_isr_4d.evaluation import predict_chunked


@pytest.mark.parametrize('mode,dim', [('xyt', 3), ('xyzt', 4)])
@pytest.mark.parametrize('activation', ['sine', 'tanh', 'relu', 'softplus'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_named_construction_preserves_initialization_keys_rng(mode, dim, activation, dtype):
    """Explicit mode must reuse the original constructor, keys and RNG draws."""
    kwargs = dict(hidden_features=8, hidden_layers=1, activation=activation,
                  first_omega_0=3., hidden_omega_0=7., outermost_linear=False)
    torch.manual_seed(17)
    reference = MLPINR(in_features=dim, **kwargs).to(dtype=dtype)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(17)
    candidate = build_model(mode, **kwargs).to(dtype=dtype)
    assert list(reference.state_dict()) == list(candidate.state_dict())
    for key, value in reference.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key])
    assert torch.equal(expected_rng, torch.get_rng_state())
    query = torch.linspace(-1, 1, 7 * dim, dtype=dtype).reshape(7, dim)
    assert torch.equal(reference(query), candidate(query))
    assert isinstance(candidate.net[-1], torch.nn.Linear)


def test_invalid_mode_fails_before_rng():
    """Unsupported named axes are rejected, not inferred from source or a recipe."""
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='coordinate_mode must be xyt or xyzt'):
        build_model('xyz')
    assert torch.equal(before, torch.get_rng_state())


def test_affine_kernels_and_native_fit_state():
    """Sharing raw arithmetic must keep broadcast order, held-out range and schema."""
    values = np.array([[1., 3., 7., 11.], [3., 8., 12., 20.]])
    scaler = AffineScaler.fit(values)
    probe = np.array([[-1., 0., 3., 8.], [5., 10., 16., 25.]])
    expected = 2.0 * (probe - scaler.minimum) / (scaler.maximum - scaler.minimum) - 1.0
    np.testing.assert_array_equal(affine_forward(probe, scaler.minimum, scaler.maximum), expected)
    np.testing.assert_array_equal(scaler.transform(probe), expected)
    inverse = 0.5 * (expected + 1.0) * (scaler.maximum - scaler.minimum) + scaler.minimum
    np.testing.assert_array_equal(affine_inverse(expected, scaler.minimum, scaler.maximum), inverse)
    np.testing.assert_array_equal(scaler.inverse_transform(expected), inverse)
    assert np.any(np.abs(expected) > 1)
    assert set(scaler.state_dict()) == {'minimum', 'maximum'}
    assert not scaler.minimum.flags.writeable
    assert AffineScaler.__module__ == 'inr_isr_4d.data'


class RecordingField(torch.nn.Module):
    """Deterministic analytic query field recording chunk shape/dtype/mode/autograd."""

    def __init__(self):
        """No parameters or RNG; record only actual forwarded batches."""
        super().__init__()
        self.batches = []

    def forward(self, coordinates):
        """Return normalized[N,1] row sum and preserve caller-owned query inputs."""
        self.batches.append((len(coordinates), coordinates.dtype, self.training, torch.is_grad_enabled()))
        return coordinates.sum(dim=1, keepdim=True)


@pytest.mark.parametrize('dtype', [None, torch.float32, torch.float64])
def test_ordered_prediction_chunks_dtype_and_mode(dtype):
    """4355rows preserve16full257chunks+243 without implicit eval or dtype policy."""
    values = np.arange(4355 * 4, dtype=np.float64).reshape(4355, 4)
    if dtype is None:
        values = values.astype(np.float32)
    field = RecordingField()
    before = values.copy()
    output = predict_normalized(field, values, device=torch.device('cpu'), dtype=dtype, chunk_size=257)
    expected_dtype = torch.float32 if dtype is None else dtype
    assert field.batches == [(257, expected_dtype, True, False)] * 16 + [(243, expected_dtype, True, False)]
    np.testing.assert_array_equal(output[:, 0], values.astype(np.float64 if dtype == torch.float64 else np.float32).sum(axis=1))
    np.testing.assert_array_equal(values, before)
    assert field.training


def test_four_d_prediction_validation_and_eval_policy():
    """4D rejects bad chunk before transforms and retains its explicit eval side effect."""
    scaler = AffineScaler(np.zeros(4), np.ones(4))
    target = AffineScaler(np.array([10.]), np.array([12.]))
    field = RecordingField()
    with pytest.raises(ValueError, match='Prediction chunk size must be positive'):
        predict_chunked(field, np.ones((2, 3)), scaler, target, device=torch.device('cpu'), dtype=torch.float32, chunk_size=0)
    assert field.training and not field.batches
    values = np.array([[0., .25, .5, .75], [1., 1.25, 1.5, 1.75]])
    output = predict_chunked(field, values, scaler, target, device=torch.device('cpu'), dtype=torch.float32, chunk_size=1)
    assert not field.training and all(not b[2] and not b[3] for b in field.batches)
    normalized = scaler.transform(values).astype(np.float32).sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(output, target.inverse_transform(normalized))


def test_trusted_checkpoint_keyword_fallback_is_explicit(monkeypatch):
    """Legacy3D retries TypeError;4D direct-read policy must not silently retry."""
    calls = []
    def loader(path, **kwargs):
        """Simulate only a historical keyword rejection, without reading pickle data."""
        calls.append(kwargs)
        if 'weights_only' in kwargs:
            raise TypeError('old keyword')
        return {'trusted': True}
    monkeypatch.setattr(torch, 'load', loader)
    assert torch_load_checkpoint(Path('trusted.pt'), torch.device('cpu')) == {'trusted': True}
    assert len(calls) == 2 and calls[0]['weights_only'] is False and 'weights_only' not in calls[1]
    calls.clear()
    with pytest.raises(TypeError, match='old keyword'):
        torch_load_checkpoint(Path('trusted.pt'), 'cpu', legacy_keyword_fallback=False)
    assert len(calls) == 1


def test_legacy_builder_scaler_errors_and_strict_parameter_keys():
    """Missing legacy scalers fail before RNG; wrong model keys are not adapted away."""
    before = torch.get_rng_state().clone()
    with pytest.raises(KeyError, match='missing coord_scalers'):
        build_model_from_checkpoint({}, torch.device('cpu'))
    with pytest.raises(KeyError, match='missing target_scaler'):
        build_model_from_checkpoint({'coord_scalers': {}}, torch.device('cpu'))
    assert torch.equal(before, torch.get_rng_state())
    model = build_model('xyt', hidden_features=8, hidden_layers=0)
    with pytest.raises(RuntimeError, match='Missing key'):
        restore_parameters(model, {}, device=torch.device('cpu'))
    assert MLPINR.__module__ == 'models'
