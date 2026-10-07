from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maxilab.analysis import HARDNESS_PAIRS, HardnessRatioAnalysis, LightCurveAnalysis


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mjd": [59000.5, 59001.5, 59002.5],
            "flux_2_20": [1.0, 1.0, 1.0],
            "err_2_20": [0.1, 0.1, 0.1],
            "flux_2_4": [2.0, 1.0, -1.0],
            "err_2_4": [0.2, 0.1, 0.2],
            "flux_4_10": [4.0, -1.0, 3.0],
            "err_4_10": [0.4, 0.2, 0.3],
            "flux_10_20": [6.0, 2.0, 4.0],
            "err_10_20": [0.6, 0.2, 0.4],
        }
    )


def test_hardness_definition_and_error_propagation():
    analysis = HardnessRatioAnalysis(LightCurveAnalysis(sample_frame(), source_name="Vela X-1"))
    result = analysis.ratio("4-10", "2-4")

    expected_ratio = (4.0 - 2.0) / (4.0 + 2.0)
    expected_error = 2.0 * np.sqrt((2.0 * 0.4) ** 2 + (4.0 * 0.2) ** 2) / 6.0**2

    assert result.hardness[0] == pytest.approx(expected_ratio)
    assert result.hardness_error[0] == pytest.approx(expected_error)
    assert result.source_name == "Vela X-1"
    assert result.label == "H: 4–10 keV; S: 2–4 keV"


def test_zero_denominator_is_returned_as_nan():
    result = HardnessRatioAnalysis.from_frame(sample_frame()).ratio("4-10", "2-4")
    assert np.isnan(result.hardness[1])
    assert np.isnan(result.hardness_error[1])
    assert not result.finite[1]


def test_negative_flux_is_not_clipped():
    result = HardnessRatioAnalysis.from_frame(sample_frame()).ratio("4-10", "2-4")
    assert result.hardness[2] == pytest.approx(2.0)
    assert result.hardness[2] > 1.0


def test_all_standard_ratios_match_declared_pairs():
    ratios = HardnessRatioAnalysis.from_frame(sample_frame()).all_standard_ratios()
    assert tuple(ratios) == HARDNESS_PAIRS


def test_overlapping_broad_band_is_rejected():
    analysis = HardnessRatioAnalysis.from_frame(sample_frame())
    with pytest.raises(ValueError, match="must not overlap"):
        analysis.ratio("2-20", "2-4")


def test_reversed_hard_and_soft_bands_are_rejected():
    analysis = HardnessRatioAnalysis.from_frame(sample_frame())
    with pytest.raises(ValueError, match="higher energy"):
        analysis.ratio("2-4", "4-10")


def test_identical_bands_are_rejected():
    analysis = HardnessRatioAnalysis.from_frame(sample_frame())
    with pytest.raises(ValueError, match="must be different"):
        analysis.ratio("4-10", "4-10")


def test_constructor_requires_lightcurve_analysis():
    with pytest.raises(TypeError, match="LightCurveAnalysis"):
        HardnessRatioAnalysis(sample_frame())  # type: ignore[arg-type]


def test_to_frame_records_definition_and_bands():
    result = HardnessRatioAnalysis.from_frame(sample_frame()).ratio("10-20", "4-10")
    frame = result.to_frame()
    assert frame.attrs["definition"] == "(H-S)/(H+S)"
    assert frame.attrs["hard_band_keV"] == (10.0, 20.0)
    assert frame.attrs["soft_band_keV"] == (4.0, 10.0)
