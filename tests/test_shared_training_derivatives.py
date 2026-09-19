"""Analytic protections for the shared named-axis training kernels.

Integer quadratic derivatives give exact values, independent of learned weights.
Both native adapters must retain their scientific reductions and input errors.
Old/new model gradients are additionally protected by the saved98-case packet.
"""
import pytest
import torch
from inr_isr.training.derivatives import coordinate_gradient,hessian_rows
from training_common import curvature_losses_xy_t,derivative_diagnostics_xy_t
from inr_isr_4d.regularization import second_derivatives_4d


class Quadratic(torch.nn.Module):
    """Known x/y/time curvature; four inputs additionally include altitude."""
    def forward(self,points):
        """Return [N,1] with diagonal Hessian2,4,(6),8 and xy mixed5."""
        x,y=points[:,0:1],points[:,1:2]
        t=points[:,-1:]
        result=x*x+2*y*y+4*t*t+5*x*y
        if points.shape[1]==4:
            z=points[:,2:3]
            result=result+3*z*z+6*x*z+7*y*z
        return result


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('axes',[('x','y','t'),('x','y','z','t')])
def test_time_is_selected_by_name_not_fixed_column(dtype,axes):
    """Time curvature is8 for both modes; altitude curvature is6 only in xyzt."""
    points=torch.ones(5,len(axes),dtype=dtype,requires_grad=True)
    first=coordinate_gradient(Quadratic()(points),points)
    rows=hessian_rows(first,points,axes=axes,selected=('t',))
    assert list(rows)==['t']
    torch.testing.assert_close(rows['t'][:,-1:],torch.full((5,1),8.,dtype=dtype),rtol=0,atol=0)
    if len(axes)==4:
        torch.testing.assert_close(rows['t'][:,2:3],torch.zeros(5,1,dtype=dtype),rtol=0,atol=0)


@pytest.mark.parametrize('xy,temporal',[(False,False),(True,False),(False,True),(True,True)])
def test_xyt_branches_keep_pointwise_hessian_reduction(xy,temporal):
    """The mixed term has multiplicity2: spatial70, time64; disabled losses zero."""
    points=torch.ones(5,3,dtype=torch.float64)
    horizontal,time=curvature_losses_xy_t(Quadratic(),points,xy,temporal)
    assert horizontal.item()==(70. if xy else 0.)
    assert time.item()==(64. if temporal else 0.)
    diagnostics=derivative_diagnostics_xy_t(Quadratic(),points,xy,temporal,1.e-12)
    if xy:
        assert diagnostics['diag_curv_xy_probe']==70.
        assert diagnostics['fxy_rms']==5.
    if temporal:
        assert diagnostics['diag_curv_t_probe']==64.


def test_xyzt_retains_all_sixteen_hessian_entries():
    """All ordered entries and original row-major key order remain available."""
    derivatives=second_derivatives_4d(Quadratic(),torch.ones(5,4,dtype=torch.float64))
    names=('x','y','z','t')
    assert list(derivatives)==[a+b for a in names for b in names]
    expected=torch.tensor([[2.,5.,6.,0.],[5.,4.,7.,0.],[6.,7.,6.,0.],[0.,0.,0.,8.]],dtype=torch.float64)
    for i,a in enumerate(names):
        for j,b in enumerate(names):
            torch.testing.assert_close(derivatives[a+b],expected[i,j].expand(5,1),rtol=0,atol=0)


def test_unknown_named_row_is_not_silently_remapped():
    """An unsupported axis must not become a different derivative by position."""
    points=torch.ones(5,3,requires_grad=True)
    first=coordinate_gradient(Quadratic()(points),points)
    with pytest.raises(ValueError):
        hessian_rows(first,points,axes=('x','y','t'),selected=('z',))
