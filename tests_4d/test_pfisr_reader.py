"""PFISR reader metadata, filtering, uncertainty, and actual-product checks."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inr_isr_4d.pfisr import (
    PFISRReadConfig,
    common_physical_window,
    inspect_pfisr_hdf5,
    read_pfisr_4d,
)


REAL_2MIN = Path("/home/jdiaz/postdoc/codex-inr-radar/data/20120122.001_lp_2min-fitcal.h5")
REAL_5MIN = Path("/home/jdiaz/postdoc/codex-inr-radar/data/20120122.001_lp_5min.h5")


def write_minimal(path: Path, durations: tuple[float, float]) -> None:
    starts = np.array([1000.0, 1200.0])
    unix = np.column_stack([starts, starts + np.asarray(durations)])
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "BeamCodes",
            data=np.array([[1001.0, 0.0, 60.0, 0.0], [1002.0, 90.0, 45.0, 0.0]]),
        )
        fitted = handle.create_group("FittedParams")
        ne = np.array(
            [
                [[1e10, 2e10, np.nan], [4e10, 5e10, 6e10]],
                [[2e10, 3e10, 4e10], [5e10, 6e10, 7e10]],
            ]
        )
        fitted.create_dataset("Ne", data=ne)
        dne = np.full(ne.shape, 1e9)
        dne[0, 0, 1] = np.inf
        fitted.create_dataset("dNe", data=dne)
        fitted.create_dataset(
            "Range", data=np.array([[150e3, 300e3, 600e3], [150e3, 300e3, 600e3]])
        )
        fitted.create_dataset(
            "Altitude", data=np.array([[130e3, 260e3, 520e3], [110e3, 220e3, 440e3]])
        )
        time = handle.create_group("Time")
        time.create_dataset("UnixTime", data=unix)


def test_reader_uses_actual_record_duration_and_preserves_raw_uncertainty(tmp_path: Path) -> None:
    path = tmp_path / "minimal.h5"
    write_minimal(path, (127.0, 303.0))
    metadata = inspect_pfisr_hdf5(path)
    np.testing.assert_array_equal(metadata.integration_duration_sec, [127.0, 303.0])
    case = read_pfisr_4d(
        path,
        PFISRReadConfig(
            altitude_min_km=100.0,
            altitude_max_km=500.0,
            minimum_ne_m3=1e8,
        ),
    )
    assert case.bundle.size == 10
    assert np.isinf(case.uncertainty_ne_m3).any()
    assert set(np.unique(case.integration_duration_sec)) == {127.0, 303.0}
    assert np.all(np.diff(np.unique(case.unix_mid)) > 0)
    assert case.exclusions["nonfinite_density"] == 1
    # Exclusion telemetry is sequential, so the non-finite point outside the
    # altitude bound is counted once under its first rejection reason.
    assert case.exclusions["outside_altitude"] == 1


def test_optional_uncertainty_quality_filter_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "minimal.h5"
    write_minimal(path, (120.0, 300.0))
    unfiltered = read_pfisr_4d(path, PFISRReadConfig())
    filtered = read_pfisr_4d(
        path, PFISRReadConfig(max_relative_uncertainty=0.08)
    )
    assert filtered.bundle.size < unfiltered.bundle.size
    assert filtered.exclusions["invalid_uncertainty_quality"] > 0
    assert np.all(np.isfinite(filtered.relative_uncertainty))
    assert np.all((filtered.relative_uncertainty > 0) & (filtered.relative_uncertainty <= 0.08))


def test_actual_two_and_five_minute_products_have_verified_metadata_and_bounded_reads() -> None:
    two = inspect_pfisr_hdf5(REAL_2MIN)
    five = inspect_pfisr_hdf5(REAL_5MIN)
    assert (two.records, two.beams, two.ranges) == (1122, 42, 18)
    assert (five.records, five.beams, five.ranges) == (471, 42, 18)
    assert set(np.unique(two.integration_duration_sec)) == {79.0, 127.0, 128.0}
    assert set(np.unique(five.integration_duration_sec)) == {108.0, 303.0, 304.0}
    assert two.state_dict()["integration_duration_sec"]["counts"] == {
        "79.0": 1,
        "127.0": 687,
        "128.0": 434,
    }
    assert five.state_dict()["integration_duration_sec"]["counts"] == {
        "108.0": 1,
        "303.0": 102,
        "304.0": 368,
    }
    start, end = common_physical_window(two, five)
    assert start == 1327270741.0 and end == 1327413629.0

    two_case = read_pfisr_4d(
        REAL_2MIN, PFISRReadConfig(record_start_index=0, record_count=2)
    )
    five_case = read_pfisr_4d(
        REAL_5MIN, PFISRReadConfig(record_start_index=0, record_count=2)
    )
    assert two_case.metadata.path == str(REAL_2MIN.resolve())
    assert five_case.metadata.path == str(REAL_5MIN.resolve())
    assert np.unique(two_case.record_indices).tolist() == [0, 1]
    assert np.unique(five_case.record_indices).tolist() == [0, 1]
    assert np.all(np.diff(np.unique(two_case.unix_mid)) > 0)
    assert np.all(np.diff(np.unique(five_case.unix_mid)) > 0)
    assert set(np.unique(two_case.integration_duration_sec)) <= {79.0, 127.0, 128.0}
    assert set(np.unique(five_case.integration_duration_sec)) <= {108.0, 303.0, 304.0}

    final_two = read_pfisr_4d(
        REAL_2MIN,
        PFISRReadConfig(
            record_start_index=1120,
            record_count=2,
            minimum_integration_fraction_of_file_median=0.8,
        ),
    )
    final_five = read_pfisr_4d(
        REAL_5MIN,
        PFISRReadConfig(
            record_start_index=469,
            record_count=2,
            minimum_integration_fraction_of_file_median=0.8,
        ),
    )
    assert np.unique(final_two.record_indices).tolist() == [1120]
    assert np.unique(final_five.record_indices).tolist() == [469]
    assert final_two.exclusions["incomplete_integration_record"] > 0
    assert final_five.exclusions["incomplete_integration_record"] > 0
