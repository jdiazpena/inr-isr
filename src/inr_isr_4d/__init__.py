"""Additive four-dimensional INR and uncertainty-quantification package.

The copied three-dimensional implementation remains available through its original
modules.  This package is intentionally separate so that 4D work cannot silently
change the verified 3D baseline.
"""

from .data import AffineScaler, FieldBundle4D, SampleDataset4D
from .model import SIREN4D

__all__ = ["AffineScaler", "FieldBundle4D", "SIREN4D", "SampleDataset4D"]
