from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from maxilab.analysis import HardnessRatioAnalysis
from maxilab.plotting import HardnessRatioPlotter


def sample_hardness():  # type: ignore[no-untyped-def]
    frame = pd.DataFrame(
        {
            "mjd": [58000.5, 58365.5, 58730.5],
            "flux_2_4": [0.10, 0.12, 0.08],
            "err_2_4": [0.01, 0.01, 0.01],
            "flux_4_10": [0.20, 0.25, 0.18],
            "err_4_10": [0.02, 0.02, 0.02],
        }
    )
    return HardnessRatioAnalysis.from_frame(frame, source_name="Vela X-1").ratio(
        "4-10", "2-4"
    )


def test_hardness_plot_has_definition_and_band_labels():
    figure, axis = HardnessRatioPlotter().plot(sample_hardness())
    try:
        assert axis.get_xlabel() == "Time [MJD]"
        assert "(H-S)/(H+S)" in axis.get_ylabel()
        texts = {text.get_text() for text in axis.texts}
        assert "Vela X-1" in texts
        assert "H: 4–10 keV · S: 2–4 keV" in texts
    finally:
        plt.close(figure)
