# -*- coding: utf-8 -*-
"""
conformal.py

Re-export conformal calibration and beam splitting utilities from inr_radar.uq.conformal.
"""

from inr_radar.uq.conformal import (
    SplitConformalCalibrator,
    ConformalCalibrator4D,
    split_beams,
    split_beams_by_strategy,
    compute_conformal_quantile,
)

__all__ = [
    "SplitConformalCalibrator",
    "ConformalCalibrator4D",
    "split_beams",
    "split_beams_by_strategy",
    "compute_conformal_quantile",
]
