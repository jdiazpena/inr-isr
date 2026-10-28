"""Loss arithmetic and interfaces only: no neural models, optimizers or training."""
import ast
import inspect
import math
from pathlib import Path
import textwrap

import pytest
import torch

from inr_isr.training.steps import (
    data_objective, data_objective_lineardata, density_mse_lineardata,
    validate_data_loss_definition, validate_lineardata_target_scaler,
)


SCALER = {'column': 'log10_Ne', 'min': 9.0, 'max': 13.0}


@pytest.mark.parametrize('recipe,dimension', [('xyt', 3), ('xyzt', 4)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_old_log_arithmetic_and_gradients_remain_exact(recipe, dimension, dtype):
    points = torch.linspace(-.7, .8, 7 * dimension, dtype=dtype).reshape(7, dimension)
    points.requires_grad_()
    targets = torch.linspace(-.2, .3, 7, dtype=dtype).reshape(7, 1)
    calls = []

    def analytic_sum(values):
        calls.append(values)
        return values.sum(dim=1, keepdim=True)

    prediction, actual = data_objective(analytic_sum, points, targets, recipe=recipe)
    expected = (torch.nn.functional.mse_loss(prediction, targets) if recipe == 'xyt'
                else torch.mean((prediction - targets) ** 2))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(actual, points, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, points)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    assert len(calls) == 1 and calls[0] is points


def test_linear_loss_undoes_normalization_before_exponentiation():
    prediction = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    targets = torch.zeros_like(prediction)
    actual = density_mse_lineardata(prediction, targets, target_scaler=SCALER,
                                   linear_density_reference=1e12)
    # Physical predicted densities are1e9,1e11,1e13; all targets are1e11.
    expected = ((.001 - .1) ** 2 + (.1 - .1) ** 2 + (10 - .1) ** 2) / 3
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=actual.dtype), rtol=1e-14, atol=0)


def test_linear_autograd_matches_analytic_chain_rule():
    prediction = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64, requires_grad=True)
    targets = torch.zeros_like(prediction)
    loss = density_mse_lineardata(prediction, targets, target_scaler=SCALER,
                                 linear_density_reference=1e12)
    actual = torch.autograd.grad(loss, prediction)[0]
    q = [.001, .1, 10.0]
    expected = torch.tensor([[2 / 3 * (value - .1) * math.log(10) * 2 * value]
                             for value in q], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=0)


def test_common_reference_rescales_loss_and_gradient_not_observation_weights():
    prediction = torch.tensor([[-.5], [.5]], dtype=torch.float64, requires_grad=True)
    targets = torch.zeros_like(prediction)
    first = density_mse_lineardata(prediction, targets, target_scaler=SCALER,
                                  linear_density_reference=1e12)
    second = density_mse_lineardata(prediction, targets, target_scaler=SCALER,
                                   linear_density_reference=2e12)
    first_gradient = torch.autograd.grad(first, prediction, retain_graph=True)[0]
    second_gradient = torch.autograd.grad(second, prediction)[0]
    torch.testing.assert_close(second, first / 4, rtol=1e-14, atol=0)
    torch.testing.assert_close(second_gradient, first_gradient / 4, rtol=1e-14, atol=0)
    individual = [density_mse_lineardata(prediction[index:index + 1], targets[index:index + 1],
                  target_scaler=SCALER, linear_density_reference=1e12) for index in range(2)]
    torch.testing.assert_close(first, (individual[0] + individual[1]) / 2, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_linear_wrapper_calls_once_preserves_output_dtype_and_rng(dtype):
    points = torch.tensor([[-1.0], [.5]], dtype=dtype)
    targets = torch.zeros_like(points)
    calls = []

    def analytic_identity(values):
        calls.append(values)
        return values

    before = torch.random.get_rng_state().clone()
    prediction, loss = data_objective_lineardata(analytic_identity, points, targets,
                        target_scaler=SCALER, linear_density_reference=1e12)
    assert len(calls) == 1 and prediction is points
    assert loss.dtype == dtype and loss.device == points.device
    assert torch.equal(before, torch.random.get_rng_state())
    exact = density_mse_lineardata(points, points, target_scaler=SCALER,
                                   linear_density_reference=1e12)
    assert exact.item() == 0


def test_historical_defaults_and_explicit_linear_definition():
    assert validate_data_loss_definition() == {
        'data_loss_domain': 'log', 'linear_density_reference': None}
    assert validate_data_loss_definition('linear', 1e12) == {
        'data_loss_domain': 'linear', 'linear_density_reference': 1e12}


@pytest.mark.parametrize('reference', [None, 0, -1, math.inf, -math.inf, math.nan, True, '1e12'])
def test_invalid_linear_reference_rejected_before_forward(reference):
    def forbidden_forward(values):
        raise AssertionError('Invalid settings must fail before a forward')

    with pytest.raises(ValueError, match='finite positive'):
        data_objective_lineardata(forbidden_forward, torch.zeros(1, 1), torch.zeros(1, 1),
                                 target_scaler=SCALER, linear_density_reference=reference)


@pytest.mark.parametrize('domain,reference', [('other', None), ('log', 1e12)])
def test_ambiguous_or_unknown_loss_definition_rejected(domain, reference):
    with pytest.raises(ValueError):
        validate_data_loss_definition(domain, reference)


@pytest.mark.parametrize('scaler,normalized', [
    (SCALER, False), ({'column': 'Ne', 'min': 9, 'max': 13}, True),
    ({'min': 9, 'max': 13}, True),
    ({'column': 'log10_Ne', 'min': 9, 'max': 9}, True),
    ({'column': 'log10_Ne', 'min': math.nan, 'max': 13}, True),
    ({'column': 'log10_Ne', 'minimum': [9], 'maximum': [13]}, True),
])
def test_invalid_target_scaling_rejected(scaler, normalized):
    with pytest.raises(ValueError):
        validate_lineardata_target_scaler(scaler, normalize_values=normalized)


def test_invalid_shape_dtype_and_nonfinite_loss_fail_without_clipping():
    for prediction, target in [(torch.zeros(2, 1), torch.zeros(2)),
                               (torch.empty(0, 1), torch.empty(0, 1)),
                               (torch.zeros(1, 1), torch.zeros(1, 1, dtype=torch.float64))]:
        with pytest.raises(ValueError):
            density_mse_lineardata(prediction, target, target_scaler=SCALER,
                                  linear_density_reference=1e12)
    for value in [math.inf, -math.inf, math.nan, 100.0]:
        with pytest.raises(FloatingPointError, match='Nonfinite'):
            density_mse_lineardata(torch.tensor([[value]]), torch.zeros(1, 1),
                                  target_scaler=SCALER, linear_density_reference=1e12)


def test_parser_keeps_log_default_and_requires_explicit_linear_units_in_validation():
    from synthetic_train_3d import build_parser
    original = build_parser().parse_args([])
    assert original.data_loss_domain == 'log' and original.linear_density_reference is None
    candidate = build_parser().parse_args(['--data_loss_domain', 'linear',
                                          '--linear_density_reference', '1e12'])
    original_values, candidate_values = vars(original).copy(), vars(candidate).copy()
    for key in ['data_loss_domain', 'linear_density_reference']:
        original_values.pop(key)
        candidate_values.pop(key)
    assert original_values == candidate_values


def test_original_data_function_source_and_four_dimensional_call_are_preserved():
    expected = '''
def original(model, points, targets, *, recipe):
    if recipe not in ('xyt','xyzt'):
        raise ValueError(f'Unknown native training recipe: {recipe}')
    prediction=model(points)
    loss=functional.mse_loss(prediction,targets) if recipe=='xyt' else torch.mean((prediction-targets)**2)
    return prediction,loss
'''
    actual = ast.parse(textwrap.dedent(inspect.getsource(data_objective))).body[0]
    old = ast.parse(textwrap.dedent(expected)).body[0]
    assert [ast.dump(node) for node in actual.body[1:]] == [ast.dump(node) for node in old.body]
    root = Path(__file__).resolve().parents[1] / 'src'
    source = ast.parse((root / 'inr_isr_4d/training.py').read_text())
    calls = [node for node in ast.walk(source) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'data_objective']
    assert len(calls) == 1
    assert len(calls[0].args) == 3
    assert [(key.arg, ast.literal_eval(key.value)) for key in calls[0].keywords] == [('recipe', 'xyzt')]
