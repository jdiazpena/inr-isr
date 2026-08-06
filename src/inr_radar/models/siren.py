# -*- coding: utf-8 -*-
"""
siren.py

Implicit Neural Representation (INR) models for ionospheric field reconstruction.
Implements the 3-Layer SIREN MLP with periodic sine activations and SIREN weight initializations.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

ActivationName = Literal["relu", "tanh", "softplus", "sine"]


class Sine(nn.Module):
    """
    Sine activation for SIREN-style networks.
    y = sin(w0 * x)
    """

    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = float(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


def get_activation(name: ActivationName, w0: float = 1.0) -> nn.Module:
    """Return activation module by name."""
    name = name.lower()

    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "tanh":
        return nn.Tanh()
    if name == "softplus":
        return nn.Softplus()
    if name == "sine":
        return Sine(w0=w0)

    raise ValueError(f"Unsupported activation: {name}")


def init_linear(
    layer: nn.Linear,
    activation: ActivationName,
    is_first: bool = False,
    w0: float = 1.0,
) -> None:
    """Initialize Linear layer for SIREN or standard activations."""
    if not isinstance(layer, nn.Linear):
        raise TypeError("init_linear expects an nn.Linear layer.")

    with torch.no_grad():
        in_features = layer.in_features

        if activation == "sine":
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / w0

            layer.weight.uniform_(-bound, bound)

        elif activation == "relu":
            nn.init.kaiming_uniform_(layer.weight, a=0.0, nonlinearity="relu")
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1.0 / math.sqrt(fan_in)
                layer.bias.uniform_(-bound, bound)

        elif activation in ("tanh", "softplus"):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        else:
            raise ValueError(f"Unsupported activation for initialization: {activation}")


class MLPINR(nn.Module):
    """
    Coordinate-based INR MLP architecture for ionospheric plasma field reconstruction.
    Maps continuous space-time coordinates (x_norm, y_norm, z_norm, t_norm) to plasma fields.
    """

    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 1,
        hidden_features: int = 256,
        hidden_layers: int = 3,
        activation: ActivationName = "sine",
        first_omega_0: float = 5.0,
        hidden_omega_0: float = 5.0,
        outermost_linear: bool = True,
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.hidden_features = int(hidden_features)
        self.hidden_layers = int(hidden_layers)
        self.activation_name = activation
        self.first_omega_0 = float(first_omega_0)
        self.hidden_omega_0 = float(hidden_omega_0)
        self.outermost_linear = bool(outermost_linear)

        layers: list[nn.Module] = []

        # First layer
        first_linear = nn.Linear(self.in_features, self.hidden_features)
        init_linear(first_linear, activation=activation, is_first=True, w0=self.first_omega_0)
        layers.append(first_linear)

        if activation == "sine":
            layers.append(get_activation("sine", w0=self.first_omega_0))
        else:
            layers.append(get_activation(activation))

        # Hidden layers
        for _ in range(self.hidden_layers):
            hidden_linear = nn.Linear(self.hidden_features, self.hidden_features)
            init_linear(hidden_linear, activation=activation, is_first=False, w0=self.hidden_omega_0)
            layers.append(hidden_linear)

            if activation == "sine":
                layers.append(get_activation("sine", w0=self.hidden_omega_0))
            else:
                layers.append(get_activation(activation))

        # Output layer
        final_linear = nn.Linear(self.hidden_features, self.out_features)
        if activation == "sine":
            init_linear(final_linear, activation="sine", is_first=False, w0=self.hidden_omega_0)
        else:
            init_linear(final_linear, activation=activation, is_first=False, w0=1.0)
        layers.append(final_linear)

        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


class SIREN4D(MLPINR):
    """
    Convenience wrapper for 4D Implicit Neural Representation SIREN model.
    Maps continuous 4D space-time coordinates (x_norm, y_norm, z_norm, t_norm) to log10_Ne.
    """

    def __init__(
        self,
        in_features: int = 4,
        out_features: int = 1,
        hidden_features: int = 256,
        hidden_layers: int = 3,
        activation: ActivationName = "sine",
        first_omega_0: float = 5.0,
        hidden_omega_0: float = 5.0,
        outermost_linear: bool = True,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            activation=activation,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            outermost_linear=outermost_linear,
        )

