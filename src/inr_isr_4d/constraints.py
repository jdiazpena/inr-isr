"""Tested interface for future independently validated scientific constraints.

No plasma-transport, precipitation, or imaging constraint is implemented here.
Implementing this interface does not establish that any such physics exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import torch


@dataclass(frozen=True)
class ConstraintEvaluation:
    loss: torch.Tensor
    diagnostics: Mapping[str, float] = field(default_factory=dict)


class Constraint4D(Protocol):
    """Contract that a future constraint must satisfy before trainer integration."""

    name: str
    version: str

    def evaluate(
        self,
        model: torch.nn.Module,
        normalized_coordinates: torch.Tensor,
        context: Mapping[str, object],
    ) -> ConstraintEvaluation:
        ...


def constraint_identities(constraints: tuple[Constraint4D, ...]) -> list[dict[str, str]]:
    identities = []
    names = set()
    for constraint in constraints:
        name = str(constraint.name).strip()
        version = str(constraint.version).strip()
        if not name or not version or name in names:
            raise ValueError("Constraint names/versions must be non-empty and names unique.")
        names.add(name)
        identities.append({"name": name, "version": version})
    return identities


def evaluate_constraint(
    constraint: Constraint4D,
    model: torch.nn.Module,
    coordinates: torch.Tensor,
    context: Mapping[str, object],
) -> ConstraintEvaluation:
    result = constraint.evaluate(model, coordinates, context)
    if result.loss.ndim != 0 or not bool(torch.isfinite(result.loss)):
        raise ValueError(f"Constraint {constraint.name} returned a non-finite or non-scalar loss.")
    if any(not isinstance(value, (int, float)) for value in result.diagnostics.values()):
        raise ValueError(f"Constraint {constraint.name} diagnostics must be scalar numbers.")
    return result
