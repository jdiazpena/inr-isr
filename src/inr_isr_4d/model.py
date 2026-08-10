"""Four-dimensional neural-field model definitions."""

from __future__ import annotations

from models import ActivationName, MLPINR


class SIREN4D(MLPINR):
    """SIREN mapping normalized ``(x, y, z, t)`` to one or more fields.

    This is the direct four-input specialization of the copied working 3D model.
    It deliberately adds no independent initialization or forward implementation.
    """

    def __init__(
        self,
        *,
        out_features: int = 1,
        hidden_features: int = 256,
        hidden_layers: int = 3,
        activation: ActivationName = "sine",
        first_omega_0: float = 5.0,
        hidden_omega_0: float = 5.0,
        outermost_linear: bool = True,
    ) -> None:
        super().__init__(
            in_features=4,
            out_features=out_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            activation=activation,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            outermost_linear=outermost_linear,
        )
