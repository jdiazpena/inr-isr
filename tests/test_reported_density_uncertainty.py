"""Analytic tensors and metadata only: no network, optimizer or training."""
from copy import deepcopy
import math

import pytest
import torch

from inr_isr.experiments.contract import Contract
from inr_isr.training.uncertainties import (
    density_uncertainty_sigma, standardized_density_mse,
    uncertainty_data_objective, weighting_definition,
)
from training_common import sample_batch


SCALER = {'column': 'log10_Ne', 'min': 9., 'max': 13.}


@pytest.mark.parametrize('domain', ['log', 'linear'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_standardized_loss_and_gradient_match_physical_formula(domain, dtype):
    ne = torch.tensor([[1e10], [1e12]], dtype=dtype)
    dne = ne * torch.tensor([[.1], [.2]], dtype=dtype)
    targets = (torch.log10(ne) - 9.) * .5 - 1.
    prediction = (targets + torch.tensor([[.08], [-.04]], dtype=dtype)).requires_grad_()
    reference = 1e12 if domain == 'linear' else None
    sigma = density_uncertainty_sigma(ne, dne, data_loss_domain=domain,
                                     target_scaler=SCALER, linear_density_reference=reference)
    actual = standardized_density_mse(prediction, targets, sigma, data_loss_domain=domain,
                                     target_scaler=SCALER, linear_density_reference=reference)
    pred_log = (prediction + 1.) * 2. + 9.
    expected = torch.mean(((pred_log - torch.log10(ne)) / (dne / (ne * math.log(10.)))) ** 2) if domain == 'log' else torch.mean(((10. ** pred_log - ne) / dne) ** 2)
    torch.testing.assert_close(actual, expected, rtol=3e-5 if dtype == torch.float32 else 1e-12, atol=2e-6 if dtype == torch.float32 else 1e-12)
    actual_grad, = torch.autograd.grad(actual, prediction, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, prediction)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-5 if dtype == torch.float32 else 1e-12, atol=2e-5 if dtype == torch.float32 else 1e-12)


@pytest.mark.parametrize('bounds', [(8., 12.), (9., 13.), (9., 15.)])
def test_log_uncertainty_and_residual_share_normalization(bounds):
    minimum, maximum = bounds
    scale = 2. / (maximum - minimum)
    ne = torch.tensor([[1e11]], dtype=torch.float64)
    dne = ne * .1
    scaler = SCALER | {'min': minimum, 'max': maximum}
    sigma = density_uncertainty_sigma(ne, dne, data_loss_domain='log', target_scaler=scaler)
    torch.testing.assert_close(sigma, torch.tensor([[.1 / math.log(10.) * scale]], dtype=torch.float64))
    target = (torch.log10(ne) - minimum) * scale - 1.
    actual = standardized_density_mse(target + .2 * scale, target, sigma,
                                     data_loss_domain='log', target_scaler=scaler)
    assert actual.item() == pytest.approx((.2 / (.1 / math.log(10.))) ** 2)


@pytest.mark.parametrize('bad', [0., -.1, math.nan, math.inf])
def test_invalid_uncertainties_fail_without_replacement(bad):
    with pytest.raises(ValueError):
        density_uncertainty_sigma(torch.ones(2, 1), torch.tensor([[.1], [bad]]),
                                  data_loss_domain='log', target_scaler=SCALER)


def test_uncertainty_is_not_logged_or_clipped_and_large_errors_remain():
    ne = torch.tensor([[1e11], [1e11]], dtype=torch.float64)
    dne = ne * torch.tensor([[.1], [2.]], dtype=torch.float64)
    actual = density_uncertainty_sigma(ne, dne, data_loss_domain='log', target_scaler=SCALER)
    torch.testing.assert_close(actual, (dne / ne) / math.log(10.) * .5)
    assert weighting_definition()['data_weighting'] == 'equal'
    assert weighting_definition('reported_dNe')['uncertainty_floor'] is None
    with pytest.raises(ValueError):
        weighting_definition('auto')


def test_linear_reference_cancels_between_residual_and_sigma():
    target = torch.tensor([[0.], [.1]], dtype=torch.float64)
    ne = 10. ** ((target + 1) * 2 + 9)
    results = []
    for reference in (1e10, 1e12):
        sigma = density_uncertainty_sigma(ne, ne * .1, data_loss_domain='linear',
                                         target_scaler=SCALER, linear_density_reference=reference)
        results.append(standardized_density_mse(target + .05, target, sigma,
            data_loss_domain='linear', target_scaler=SCALER, linear_density_reference=reference))
    torch.testing.assert_close(*results, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('batch_size', [0, 3, 8])
def test_values_and_uncertainties_use_identical_sample_indices(batch_size):
    points = torch.arange(8.).reshape(-1, 1)
    packed = torch.cat((points * 3, points + 100), dim=1)
    before = torch.random.get_rng_state().clone()
    coords, batch = sample_batch(points, packed, batch_size)
    after = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(before)
    old_coords, _ = sample_batch(points, points * 3, batch_size)
    assert torch.equal(after, torch.random.get_rng_state())
    assert torch.equal(coords, old_coords)
    assert torch.equal(batch[:, :1], coords * 3)
    assert torch.equal(batch[:, 1:], coords + 100)


def test_checkpoint_cannot_silently_change_weighting():
    data = {'schema': 'shared-xyt-radar-reconstruction-1',
            'training': {'num_steps': 5000, 'data_weighting': 'reported_dNe'}}
    contract = Contract(None, data)
    definition = weighting_definition('reported_dNe')
    checkpoint = {'step': 5000, 'config': {'num_steps': 5000,
                  'data_weighting': 'reported_dNe', 'data_weighting_definition': definition}}
    assert contract.validate_checkpoint_loss(checkpoint)['data_loss_domain'] == 'log'
    for change in ({'data_weighting': 'equal'}, {'data_weighting_definition': None},
                   {'data_weighting_definition': definition | {'weight_cap': 100}}):
        changed = deepcopy(checkpoint)
        changed['config'].update(change)
        with pytest.raises(ValueError):
            contract.validate_checkpoint_loss(changed)


def test_invalid_sigma_stops_before_any_forward_call():
    def forbidden(_):
        raise AssertionError('Must reject invalid uncertainty first')
    with pytest.raises(ValueError):
        uncertainty_data_objective(forbidden, torch.zeros(1, 3), torch.zeros(1, 1),
            torch.zeros(1, 1), data_loss_domain='log', target_scaler=SCALER)
