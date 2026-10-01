from __future__ import annotations

import os
from pathlib import Path

import pytest

from maxilab import MAXISource


pytestmark = pytest.mark.live


@pytest.mark.skipif(
    os.environ.get("MAXILAB_RUN_LIVE") != "1",
    reason="Set MAXILAB_RUN_LIVE=1 to contact the live MAXI website.",
)
def test_live_vela_download(tmp_path: Path):
    source = MAXISource("Vela X-1", data_dir=tmp_path, timeout=60)
    assert source.source_name == "Vela X-1"
    assert source.latest_mjd > 55000
    assert len(source.data) > 100
    assert source.product_url.endswith(".dat")
