# -*- coding: utf-8 -*-
"""
synthetic_plasma_4d.py

Flat compatibility wrapper for 4D synthetic plasma field generator.
"""

from __future__ import annotations

from inr_radar.datasets.synthetic_generator_4d import (
    MovingGaussianPatch4D,
    vertical_f2_profile,
    evaluate_synthetic_plasma_4d,
    make_observation_geometry_4d,
    generate_synthetic_beam_dataset_4d,
)

__all__ = [
    "MovingGaussianPatch4D",
    "vertical_f2_profile",
    "evaluate_synthetic_plasma_4d",
    "make_observation_geometry_4d",
    "generate_synthetic_beam_dataset_4d",
]
