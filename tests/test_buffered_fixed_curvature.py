"""Geometry, scalar schedules and config identity only; no network or optimizer."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from inr_isr.experiments.contract import Contract
from inr_isr.training.collocation import (
    buffered_beam_grid, collocation_definition, distance_to_beam_hull,
    make_buffered_collocation_pool,
)
from inr_isr.training.controllers import update_xyt_weights
from training_common import make_query_grid_from_points


SQUARE = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]])


def state(step, **overrides):
    settings = dict(loss_ema_beta=.99, epsilon_data=1e-6, reference_loss_weights=False,
                    lambda_curv_xy=1e-5, lambda_curv_t=7.74e-7, num_steps=30000,
                    reg_ramp_frac=1/30, lambda_warmup_steps=500, fixed_lambda_warmup=True)
    settings.update(overrides)
    return update_xyt_weights(SimpleNamespace(**settings), step=step,
        use_xy_curv=True, use_t_curv=True, data_scalar_current=2.,
        curv_xy_scalar_current=3., curv_t_scalar_current=4., data_loss_ema=None,
        curv_xy_raw_ema=None, curv_t_raw_ema=None, lambda_curv_xy_base=0.,
        lambda_curv_t_base=0., lambda_curv_xy_target=0., lambda_curv_t_target=0.,
        lambda_frozen=False)


@pytest.mark.parametrize('step,factor', [(1,0), (3,0), (500,0), (501,.002),
                                        (750,.5), (1000,1), (30000,1)])
def test_selected_schedule_and_constants(step, factor):
    result = state(step)
    assert result['lambda_curv_xy_eff'] == pytest.approx(1e-5*factor)
    assert result['lambda_curv_t_eff'] == pytest.approx(7.74e-7*factor)
    assert not result['lambda_update_active']


def test_short_execution_check_stays_in_warmup():
    assert state(3, num_steps=3)['lambda_curv_xy_eff'] == 0


def test_legacy_fixed_mode_keeps_original_ramp():
    assert state(500, fixed_lambda_warmup=False)['lambda_curv_xy_eff'] == 5e-6


def test_empty_ramp_rejected_when_training_leaves_warmup():
    with pytest.raises(ValueError, match='after warmup'):
        state(501, reg_ramp_frac=.01)


def test_rounded_corner_distance_not_square_or_nearest_beam_mask():
    points = np.array([[5,5], [5,10], [5,11], [11,11], [10.6,10.6]])
    assert np.allclose(distance_to_beam_hull(points, SQUARE), [0,0,1,2**.5,.6*2**.5])


def test_full_margin_covered_and_interior_grid_retained():
    X0,Y0 = make_query_grid_from_points(SQUARE[:,0], SQUARE[:,1], 11,11,.05)
    X,Y,mask = buffered_beam_grid(SQUARE,11,11,.05,4.)
    assert X.min() <= -4 and X.max() >= 14 and Y.min() <= -4 and Y.max() >= 14
    assert set(X0[0]) <= set(X[0]) and set(Y0[:,0]) <= set(Y[:,0])
    points = np.column_stack([X.ravel(),Y.ravel()])
    d = distance_to_beam_hull(points, SQUARE)
    assert (d[mask.ravel()] <= 4).all()
    assert (mask.ravel()[d == 0]).all()
    assert (~mask.ravel()[d > 4]).all()


def test_pool_preserves_times_scalers_and_rng():
    import torch
    frame = pd.DataFrame([(x,y,t) for t in [0.,120.] for x,y in SQUARE],
                         columns=['x_km','y_km','t_sec'])
    dataset = SimpleNamespace(coord_scalers={'x_km':{'min':0.,'max':10.},
                           'y_km':{'min':0.,'max':10.},'t_sec':{'min':0.,'max':120.}})
    rng = torch.random.get_rng_state().clone()
    pool, buffer, fraction, n_times = make_buffered_collocation_pool(dataset,frame,11,11,.05,4.)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert pool.dtype == torch.float32 and buffer == 4. and 0 < fraction < 1 and n_times == 2
    assert set(pool[:,2].tolist()) == {-1.,1.}
    assert (pool[:len(pool)//2,:2] == pool[len(pool)//2:,:2]).all()


@pytest.mark.parametrize('region,buffer', [('unknown',None),('nearest_beam',50.),
    ('beam_hull_buffer',None),('beam_hull_buffer',-1),('beam_hull_buffer',float('nan'))])
def test_invalid_region_rejected(region,buffer):
    with pytest.raises(ValueError):
        collocation_definition(region,buffer)


@pytest.mark.parametrize('key,value', [('collocation_buffer_km',25.),
    ('fixed_lambda_warmup',False),('lambda_curv_xy',1e-6)])
def test_new_checkpoint_identity_rejects_wrong_region_or_schedule(key,value):
    training = dict(num_steps=30000, collocation_region='beam_hull_buffer',
        collocation_buffer_km=50., fixed_lambda_warmup=True, reference_loss_weights=False,
        lambda_curv_xy=1e-5, lambda_curv_t=7.74e-7, lambda_warmup_steps=500, reg_ramp_frac=1/30)
    contract = Contract(Path('unused.json'), {'schema':'shared-xyt-reconstruction-1','training':training})
    assert contract.validate_checkpoint_loss({'step':30000,'config':training}) == contract.loss_definition()
    with pytest.raises(ValueError):
        contract.validate_checkpoint_loss({'step':30000,'config':dict(training,**{key:value})})


def test_legacy_checkpoint_with_omitted_selectors_still_matches():
    contract = Contract(Path('unused.json'), {'schema':'shared-xyt-reconstruction-1','training':{'num_steps':3}})
    assert contract.validate_checkpoint_loss({'step':3,'config':{'num_steps':3}}) == contract.loss_definition()
