"""Optional scalar data-loss tolerance; independent of target or spatial units."""
import math
import torch


def validate_data_loss_floor(value=0.0):
    if isinstance(value, bool):
        raise ValueError('data_loss_floor must be a finite nonnegative number')
    value=float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('data_loss_floor must be a finite nonnegative number')
    return value


def apply_data_loss_floor(raw_loss, floor=0.0):
    """Keep raw loss above tolerance; zero its gradient at/below tolerance.

    The caller still computes raw_loss every update. A later rise above floor
    automatically restores its gradient. Disabled returns the exact same tensor.
    """
    floor=validate_data_loss_floor(floor)
    if floor == 0.0:
        return raw_loss
    # Nonfinite raw values must remain visible rather than being hidden by a floor.
    return torch.where(raw_loss <= floor, raw_loss.new_tensor(floor), raw_loss)
