"""Autograd kernels shared by the explicitly named xyt and xyzt recipes.

Coordinates are normalized [N,D], with axes supplied by the caller; the third
column is never assumed to mean time. These kernels construct differentiable
coordinate derivatives, not a plasma equation. Native adapters own validation,
physical scaling, loss reductions and enabled-component choices.
"""
from __future__ import annotations
import torch


def coordinate_gradient(output: torch.Tensor, points: torch.Tensor, *,
                        only_inputs: bool | None = None) -> torch.Tensor:
    """Differentiate an output column with respect to all coordinate columns.

For the pointwise INR, ones_like(output) selects the sum of independent row
outputs, producing one gradient row per point. create_graph=True keeps this
derivative differentiable for Hessians and parameter backward; retain_graph=True
allows later rows/components. No coercion, RNG, parameter .grad mutation or
extra forward occurs. Native xyt explicitly passes only_inputs=True; xyzt retains
the historical omitted keyword. Shape/output validation stays with the adapter.
"""
    options=dict(grad_outputs=torch.ones_like(output),create_graph=True,retain_graph=True)
    if only_inputs is not None:
        options['only_inputs']=only_inputs
    return torch.autograd.grad(output,points,**options)[0]


def hessian_rows(first: torch.Tensor, points: torch.Tensor, *,
                 axes: tuple[str,...], selected: tuple[str,...],
                 only_inputs: bool | None = None) -> dict[str,torch.Tensor]:
    """Return requested second-derivative rows [N,D], in selected name order.

first [N,D] is the graph-bearing coordinate gradient at points. For axes=(x,y,t),
the t row differentiates column2; for axes=(x,y,z,t), it differentiates column3.
Only selected rows are evaluated: xyt can disable spatial or temporal priors
without silently adding computations. Every row uses the same shared autograd
kernel. No detached copies or physical/target scaling are introduced here.
Unknown names raise the tuple lookup error; native adapters use fixed valid names.
"""
    rows={}
    for name in selected:
        axis=axes.index(name)
        rows[name]=coordinate_gradient(first[:,axis:axis+1],points,only_inputs=only_inputs)
    return rows
