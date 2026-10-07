from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from maxilab.analysis import LightCurveAnalysis
from maxilab.plotting import LightCurvePlotter, PublicationStyle, save_publication_figure


def sample_curve():  # type: ignore[no-untyped-def]
    frame = pd.DataFrame(
        {
            "mjd": [58000.5, 58365.5, 58730.5],
            "flux_2_4": [0.10, 0.20, 0.15],
            "err_2_4": [0.01, 0.02, 0.01],
        }
    )
    return LightCurveAnalysis(frame, source_name="Vela X-1").lightcurve("2-4")


def test_lightcurve_plot_has_canonical_labels_and_zero_line():
    figure, axis = LightCurvePlotter().plot(sample_curve())
    try:
        assert axis.get_xlabel() == "Time [MJD]"
        assert "MAXI/GSC flux" in axis.get_ylabel()
        assert "2–4 keV" in axis.get_ylabel()
        texts = {text.get_text() for text in axis.texts}
        assert "Vela X-1" in texts
        assert "MAXI/GSC · 2–4 keV" in texts
        assert any(line.get_ydata()[0] == pytest.approx(0.0) for line in axis.lines)
    finally:
        plt.close(figure)


def test_plotting_does_not_modify_global_font_size():
    before = matplotlib.rcParams["font.size"]
    style = PublicationStyle(font_size=19.0, small_font_size=17.0)
    figure, _ = LightCurvePlotter(style).plot(sample_curve())
    plt.close(figure)
    assert matplotlib.rcParams["font.size"] == before


def test_publication_save_writes_pdf_and_600dpi_png(tmp_path: Path):
    figure, _ = LightCurvePlotter().plot(sample_curve(), show_year_guides=False)
    try:
        written = save_publication_figure(figure, tmp_path / "vela_x1_2_4keV")
        paths = tuple(written)
        assert {path.suffix for path in paths} == {".pdf", ".png"}
        assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    finally:
        plt.close(figure)
