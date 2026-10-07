from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from maxilab.analysis import LightCurveAnalysis
from maxilab.runlog import RunLog


def test_runlog_uses_timestamped_filename_and_records_operations(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "mjd": [59000.5, 59001.5],
            "flux_2_4": [0.2, -0.1],
            "err_2_4": [0.03, 0.04],
        }
    )

    with RunLog(output_dir=tmp_path) as run:
        run.info("Custom operation", purpose="unit-test")
        LightCurveAnalysis(frame, source_name="Vela X-1").lightcurve("2-4")
        path = run.path

    assert path is not None
    assert re.fullmatch(r"MAXILab_\d{8}_\d{6}(?:_\d{2})?\.log", path.name)
    assert path.exists()

    content = path.read_text(encoding="utf-8")
    assert "MAXILab run started" in content
    assert "Custom operation" in content
    assert "Extracted light curve" in content
    assert "negative_flux=1" in content
    assert "preserved_pending_scientific_filter_policy" in content
    assert "MAXILab run completed" in content
