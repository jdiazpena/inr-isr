"""Known-expression/order tests supplement exact real training and resume traces."""
import pytest
import torch
from inr_isr.training.steps import data_objective,compose_xyt_loss,run_operations,parameter_update


class PointSum(torch.nn.Module):
    """Deterministic no-parameter pointwise output for data expression tests."""
    def forward(self,points):
        """Sum coordinate columns, returning [N,1] without RNG/state mutation."""
        return points.sum(dim=1,keepdim=True)


@pytest.mark.parametrize('recipe,dim',[('xyt',3),('xyzt',4)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_each_native_data_expression_retains_exact_value(recipe,dim,dtype):
    """Each recipe is compared with its own original expression, not the other mode."""
    points=torch.linspace(-.7,.8,7*dim,dtype=dtype).reshape(7,dim)
    targets=torch.linspace(-.2,.3,7,dtype=dtype).reshape(7,1)
    prediction,loss=data_objective(PointSum(),points,targets,recipe=recipe)
    expected=torch.nn.functional.mse_loss(prediction,targets) if recipe=='xyt' else torch.mean((prediction-targets)**2)
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)


def test_xyt_update_clears_backwards_probes_then_updates():
    """A real gradient reaches the same optimizer only after the optional probe."""
    events=[];parameter=torch.tensor([2.],requires_grad=True);diagnostics={}
    class Optimizer:
        def zero_grad(self,*,set_to_none):
            assert set_to_none;events.append('clear');parameter.grad=None
        def step(self):
            assert parameter.grad.item()==4.;events.append('step')
    def norm(*,parameters):
        assert parameters[0] is parameter;events.append('probe');return parameter.grad.item()
    parameter_update(Optimizer(),recipe='xyt',total_loss=parameter.square().sum(),
        compute_grad_norm=True,parameters=[parameter],diagnostics=diagnostics,grad_norm=norm)
    assert events==['clear','probe','step'] and diagnostics=={'grad_norm_total':4.}


def test_xyzt_scaler_step_precedes_scale_update():
    """The microbatch recipe does not clear or repeat already-accumulated backwards."""
    events=[];optimizer=object()
    class Scaler:
        def step(self,actual):assert actual is optimizer;events.append('step')
        def update(self):events.append('update')
    parameter_update(optimizer,recipe='xyzt',scaler=Scaler())
    assert events==['step','update']


def test_ordered_operations_propagate_failure_without_later_operation():
    """A failed real stage stops execution rather than silently retrying/updating."""
    events=[]
    def fail():events.append('fail');raise RuntimeError('known failure')
    with pytest.raises(RuntimeError,match='known failure'):
        run_operations([('first',lambda:events.append('first')),('fail',fail),('later',lambda:events.append('later'))])
    assert events==['first','fail']


@pytest.mark.parametrize('function',['data','update'])
def test_unknown_recipe_does_not_silently_select_a_mode(function):
    """Recipe comes from explicit configuration, never inferred from array shape."""
    with pytest.raises(ValueError,match='Unknown native training recipe'):
        if function=='data':data_objective(PointSum(),torch.ones(2,3),torch.zeros(2,1),recipe='invalid')
        else:parameter_update(object(),recipe='invalid')


def test_loss_addition_is_not_reassociated():
    """Floating-point cancellation distinguishes (data+xy)+t from data+(xy+t)."""
    data,xy,time=[torch.tensor(x,dtype=torch.float32) for x in [1.e20,-1.e20,3.]]
    assert compose_xyt_loss(data,xy,time).item()==3.
    assert (data+(xy+time)).item()==0.
