from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maxilab.analysis import ENERGY_BANDS, LightCurveAnalysis, get_energy_band


def sample_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "mjd": [59000.5, 59001.5, 59002.5],
            "flux_2_20": [1.00, 1.10, 0.90],
            "err_2_20": [0.10, 0.10, 0.10],
            "flux_2_4": [0.20, 0.25, -0.02],
            "err_2_4": [0.03, 0.03, 0.04],
            "flux_4_10": [0.50, 0.55, 0.45],
            "err_4_10": [0.04, 0.04, 0.04],
            "flux_10_20": [0.30, 0.30, 0.47],
            "err_10_20": [0.05, 0.05, 0.06],
        }
    )
    frame.attrs["flux_unit"] = "ph s^-1 cm^-2"
    return frame


def test_all_standard_bands_are_available():
    analysis = LightCurveAnalysis(sample_frame(), source_name="Vela X-1")
    assert tuple(band.key for band in analysis.available_bands) == tuple(ENERGY_BANDS)


def test_lightcurve_extracts_band_without_modifying_values():
    frame = sample_frame()
    analysis = LightCurveAnalysis(frame, source_name="Vela X-1")
    curve = analysis.lightcurve("2-4")

    np.testing.assert_allclose(curve.mjd, frame["mjd"])
    np.testing.assert_allclose(curve.flux, frame["flux_2_4"])
    np.testing.assert_allclose(curve.flux_error, frame["err_2_4"])
    assert curve.band.key == "2-4"
    assert curve.band.label == "2–4 keV"
    assert curve.source_name == "Vela X-1"
    assert curve.flux_unit == "ph s^-1 cm^-2"
    assert curve.n_points == 3
    assert curve.finite.all()


def test_negative_background_subtracted_flux_is_preserved():
    curve = LightCurveAnalysis(sample_frame()).lightcurve("2-4")
    assert curve.flux[-1] == pytest.approx(-0.02)
    assert curve.finite[-1]


def test_to_frame_returns_independent_compact_table():
    curve = LightCurveAnalysis(sample_frame()).lightcurve("4-10")
    result = curve.to_frame()

    assert list(result.columns) == ["mjd", "flux", "flux_error"]
    assert result.attrs["band_keV"] == (4.0, 10.0)
    result.loc[0, "flux"] = 999.0
    assert curve.flux[0] != 999.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2-4", "2-4"),
        ("2–4 keV", "2-4"),
        ("4_10", "4-10"),
        (" 10 - 20 KEV ", "10-20"),
    ],
)
def test_band_parser_accepts_common_notation(value: str, expected: str):
    assert get_energy_band(value).key == expected


def test_unknown_band_is_rejected():
    with pytest.raises(ValueError, match="Unknown MAXI energy band"):
        get_energy_band("3-8")


def test_missing_band_columns_are_reported():
    frame = sample_frame().drop(columns=["err_4_10"])
    analysis = LightCurveAnalysis(frame)
    with pytest.raises(ValueError, match="err_4_10"):
        analysis.lightcurve("4-10")


def test_input_validation_rejects_bad_tables():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        LightCurveAnalysis([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="'mjd'"):
        LightCurveAnalysis(pd.DataFrame({"flux_2_4": [1.0]}))
    with pytest.raises(ValueError, match="at least one row"):
        LightCurveAnalysis(pd.DataFrame(columns=["mjd"]))
    with pytest.raises(ValueError, match="monotonically increasing"):
        LightCurveAnalysis(pd.DataFrame({"mjd": [2.0, 1.0]}))
