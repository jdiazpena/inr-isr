"""Scalar-autograd/controller/metadata checks; no network or optimizer execution."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from inr_isr.training.data_tolerance import apply_data_loss_floor, validate_data_loss_floor
from inr_isr.training.controllers import update_xyt_weights
from inr_isr.experiments.contract import Contract


def test_disabled_returns_original_tensor_and_gradient():
    raw=torch.tensor(0.00002,dtype=torch.float64,requires_grad=True)
    used=apply_data_loss_floor(raw,0)
    assert used is raw
    assert torch.autograd.grad(used,raw)[0].item()==1


@pytest.mark.parametrize('value',[-1,float('nan'),float('inf'),True])
def test_invalid_floor_rejected(value):
    with pytest.raises(ValueError,match='finite nonnegative'):
        validate_data_loss_floor(value)


@pytest.mark.parametrize('value,expected,gradient',[
    (0.00005,0.0001,0), (0.0001,0.0001,0), (0.0002,0.0002,1)])
def test_below_equal_and_above_have_required_value_and_gradient(value,expected,gradient):
    raw=torch.tensor(value,dtype=torch.float64,requires_grad=True)
    used=apply_data_loss_floor(raw,0.0001)
    assert used.item()==expected
    assert torch.autograd.grad(used,raw)[0].item()==gradient


def test_data_reactivates_and_curvature_keeps_its_gradient():
    # Independent evaluations represent a below/above/below sequence; no fit.
    for value,active in [(0.00005,False),(0.0002,True),(0.00003,False)]:
        raw=torch.tensor(value,dtype=torch.float64,requires_grad=True)
        curvature=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
        objective=apply_data_loss_floor(raw,0.0001)+0.5*curvature.square()
        data_gradient,curvature_gradient=torch.autograd.grad(objective,(raw,curvature))
        assert data_gradient.item()==int(active)
        assert curvature_gradient.item()==2.


@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_nonfinite_raw_loss_is_not_hidden(value):
    raw=torch.tensor(value)
    assert not torch.isfinite(apply_data_loss_floor(raw,.0001))


def test_existing_controller_receives_floored_contribution():
    args=SimpleNamespace(loss_ema_beta=.99,epsilon_data=1e-6,reference_loss_weights=True,
        curvature_ema_floor=1e-30,target_xy_ratio=.5,target_t_ratio=.5,
        lambda_curv_xy_min=0.,lambda_curv_xy_max=1.,lambda_curv_t_min=0.,lambda_curv_t_max=1.,
        freeze_lambdas_after_step=0,lambda_warmup_steps=0,lambda_update_every=1,
        lambda_smoothing=1.,num_steps=30000,reg_ramp_frac=0.)
    contribution=apply_data_loss_floor(torch.tensor(1e-8,dtype=torch.float64),.0001).item()
    state=update_xyt_weights(args,step=1,use_xy_curv=True,use_t_curv=True,
        data_scalar_current=contribution,curv_xy_scalar_current=2.,curv_t_scalar_current=4.,
        data_loss_ema=None,curv_xy_raw_ema=None,curv_t_raw_ema=None,
        lambda_curv_xy_base=0.,lambda_curv_t_base=0.,lambda_curv_xy_target=0.,
        lambda_curv_t_target=0.,lambda_frozen=False)
    assert state['data_reference']==.0001
    assert state['lambda_curv_xy_eff']*2==pytest.approx(.5*.0001)
    assert state['lambda_curv_t_eff']*4==pytest.approx(.5*.0001)


def test_contract_rejects_checkpoint_with_different_tolerance():
    contract=Contract(Path('unused.json'),{'schema':'shared-xyt-reconstruction-1',
        'training':{'num_steps':30000,'data_loss_floor':.0001}})
    matching={'step':30000,'config':{'num_steps':30000,'data_loss_floor':.0001}}
    assert contract.validate_checkpoint_loss(matching)==contract.loss_definition()
    with pytest.raises(ValueError,match='tolerance'):
        contract.validate_checkpoint_loss({'step':30000,'config':{'num_steps':30000}})


def test_legacy_omitted_tolerance_remains_disabled():
    contract=Contract(Path('unused.json'),{'schema':'shared-xyt-reconstruction-1','training':{'num_steps':3}})
    assert contract.validate_checkpoint_loss({'step':3,'config':{'num_steps':3}})==contract.loss_definition()
