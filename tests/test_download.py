from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from maxilab import (
    LIGHTCURVE_COLUMNS,
    MAXIConnectionError,
    MAXIDataFormatError,
    MAXILabError,
    MAXISource,
    SourceNotFoundError,
)
from maxilab.download import (
    _atomic_write_text,
    _loose_name,
    _normalise_name,
    _parse_coordinates,
    _safe_slug,
)


CATALOGUE = """
<html><body><table>
<tr><th>source name</th><th>R.A., Dec</th><th>standard</th></tr>
<tr>
  <td>Vela X-1</td><td>135.529, -40.555</td>
  <td><a href="../star_data/J0902-405/J0902-405.html">yes</a></td>
</tr>
<tr>
  <td>Cyg X-1</td><td>299.590, 35.202</td>
  <td><a href="../star_data/J1958+352/J1958+352.html">yes</a></td>
</tr>
<tr>
  <td>No Standard Product</td><td>1.0, 2.0</td><td>--</td>
</tr>
</table></body></html>
"""

SOURCE_PAGE = """
<html><body>
<a href="J0902-405_g_lc_1day.png">1day image</a>
<a href="J0902-405_g_lc_1orb_all.dat">1orbit LC</a>
<a href="J0902-405_g_lc_1day_all.dat">1day LC</a>
</body></html>
"""

LIGHTCURVE = """\
# optional comment
55055.500000 0.706956 0.028919 0.142204 0.014056 0.310069 0.014474 0.154730 0.015810
55056.500000 0.513911 0.021521 0.142107 0.010961 0.204903 0.010590 0.118102 0.012053
"""


class FakeResponse:
    def __init__(self, body: str, status_code: int = 200):
        self.content = body.encode("utf-8")
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, routes: dict[str, FakeResponse] | None = None):
        self.urls: list[str] = []
        self.closed = False
        self.routes = routes

    def get(self, url: str, timeout: float):
        self.urls.append(url)
        if self.routes is not None:
            try:
                return self.routes[url]
            except KeyError as exc:
                raise AssertionError(f"Unexpected URL: {url}") from exc
        if url.endswith("/top/slist.html"):
            return FakeResponse(CATALOGUE)
        if url.endswith("/star_data/J0902-405/J0902-405.html"):
            return FakeResponse(SOURCE_PAGE)
        if url.endswith("_g_lc_1orb_all.dat") or url.endswith("_g_lc_1day_all.dat"):
            return FakeResponse(LIGHTCURVE)
        raise AssertionError(f"Unexpected URL: {url}")

    def close(self) -> None:
        self.closed = True


def test_orbit_download_on_construction_and_provenance(tmp_path: Path):
    session = FakeSession()
    source = MAXISource("Vela X-1", data_dir=tmp_path, session=session)

    assert source.source_name == "Vela X-1"
    assert source.coordinates == (135.529, -40.555)
    assert source.source_page_url.endswith("/star_data/J0902-405/J0902-405.html")
    assert source.product_url.endswith("J0902-405_g_lc_1orb_all.dat")
    assert source.is_loaded
    assert isinstance(source.data, pd.DataFrame)
    assert tuple(source.data.columns) == LIGHTCURVE_COLUMNS
    assert source.latest_mjd == pytest.approx(55056.5)
    assert source.downloaded_at_utc.tzinfo is not None
    assert len(source.sha256) == 64
    assert source.raw_path.read_text(encoding="utf-8") == LIGHTCURVE
    assert source.metadata_path.exists()
    assert source.raw_path.name == "maxi_orbit.dat"

    metadata = json.loads(source.metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_name"] == "Vela X-1"
    assert metadata["binning"] == "orbit"
    assert metadata["rows"] == 2
    assert metadata["latest_mjd"] == pytest.approx(55056.5)
    assert metadata["sha256"] == source.sha256
    assert source.data.attrs["flux_unit"] == "ph s^-1 cm^-2"
    assert source.data.attrs["time_scale"] == "MJD (ISS time)"
    assert "rows=2" in repr(source)


def test_day_download_ignores_image_link(tmp_path: Path):
    session = FakeSession()
    source = MAXISource("Vela X-1", binning="day", data_dir=tmp_path, session=session)
    assert source.product_url.endswith("J0902-405_g_lc_1day_all.dat")
    assert source.raw_path.name == "maxi_day.dat"


def test_name_matching_is_case_space_and_punctuation_tolerant(tmp_path: Path):
    source = MAXISource("  vela x1  ", data_dir=tmp_path, session=FakeSession())
    assert source.source_name == "Vela X-1"
    assert _normalise_name("A–B") == "a-b"
    assert _loose_name("Vela X-1") == "velax1"


def test_unknown_source_has_close_matches(tmp_path: Path):
    with pytest.raises(SourceNotFoundError, match="Close matches: Vela X-1"):
        MAXISource("Vela X-2", data_dir=tmp_path, session=FakeSession())


def test_unknown_source_without_suggestion(tmp_path: Path):
    with pytest.raises(SourceNotFoundError, match="was not found") as exc:
        MAXISource("zzzzzzzz", data_dir=tmp_path, session=FakeSession())
    assert "Close matches" not in str(exc.value)


def test_ambiguous_loose_name_is_rejected(tmp_path: Path):
    catalogue = """
    <table>
      <tr><td>A-B</td><td>1, 2</td><td><a href="a.html">yes</a></td></tr>
      <tr><td>AB</td><td>3, 4</td><td><a href="b.html">yes</a></td></tr>
    </table>
    """
    session = FakeSession({
        "https://maxi.riken.jp/top/slist.html": FakeResponse(catalogue),
    })
    with pytest.raises(SourceNotFoundError, match="ambiguous"):
        MAXISource("A B", data_dir=tmp_path, session=session)


def test_catalogue_without_standard_entries_is_rejected(tmp_path: Path):
    bad_catalogue = "<table><tr><td>X</td><td>1, 2</td><td>--</td></tr></table>"
    session = FakeSession({
        "https://maxi.riken.jp/top/slist.html": FakeResponse(bad_catalogue),
    })
    with pytest.raises(MAXIDataFormatError, match="no standard-product"):
        MAXISource("X", data_dir=tmp_path, session=session)


def test_missing_and_multiple_lightcurve_links_are_rejected(tmp_path: Path):
    missing = FakeSession({
        "https://maxi.riken.jp/top/slist.html": FakeResponse(CATALOGUE),
        "https://maxi.riken.jp/star_data/J0902-405/J0902-405.html": FakeResponse(
            '<a href="plot.png">1orbit</a>'
        ),
    })
    with pytest.raises(MAXIDataFormatError, match="Could not find"):
        MAXISource("Vela X-1", data_dir=tmp_path, session=missing)

    multiple_page = """
    <a href="a_1orb.dat">1orbit A</a>
    <a href="b_1orb.dat">1orbit B</a>
    """
    multiple = FakeSession({
        "https://maxi.riken.jp/top/slist.html": FakeResponse(CATALOGUE),
        "https://maxi.riken.jp/star_data/J0902-405/J0902-405.html": FakeResponse(multiple_page),
    })
    with pytest.raises(MAXIDataFormatError, match="multiple"):
        MAXISource("Vela X-1", data_dir=tmp_path, session=multiple)


def test_network_error_is_wrapped(tmp_path: Path):
    class BrokenSession(FakeSession):
        def get(self, url: str, timeout: float):
            raise requests.ConnectionError("offline")

    with pytest.raises(MAXIConnectionError, match="Could not retrieve"):
        MAXISource("Vela X-1", data_dir=tmp_path, session=BrokenSession())


def test_http_error_is_wrapped(tmp_path: Path):
    session = FakeSession({
        "https://maxi.riken.jp/top/slist.html": FakeResponse("nope", status_code=503),
    })
    with pytest.raises(MAXIConnectionError, match="HTTP 503"):
        MAXISource("Vela X-1", data_dir=tmp_path, session=session)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("\n# only comments\n", "no data rows"),
        ("55055.5 1 2 3\n", "exactly 9 columns"),
        ("not-a-number 1 2 3 4 5 6 7 8\n", "Could not parse numeric"),
        (
            "55055.5 1 2 3 4 5 6 7 8\n55055.5 1 2 3 4 5 6 7 8\n",
            "Duplicate MJD",
        ),
        (
            "55056.5 1 2 3 4 5 6 7 8\n55055.5 1 2 3 4 5 6 7 8\n",
            "not monotonically increasing",
        ),
    ],
)
def test_malformed_lightcurves_are_rejected(text: str, message: str):
    with pytest.raises(MAXIDataFormatError, match=message):
        MAXISource._parse_lightcurve(text, "https://example.test/lc.dat")


def test_missing_mjd_is_rejected(monkeypatch: pytest.MonkeyPatch):
    fake = pd.DataFrame([[float("nan")] + [1.0] * 8], columns=LIGHTCURVE_COLUMNS)
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: fake)
    with pytest.raises(MAXIDataFormatError, match="Missing MJD"):
        MAXISource._parse_lightcurve("1 2 3 4 5 6 7 8 9", "https://example.test/lc.dat")


def test_empty_frame_after_parser_is_rejected(monkeypatch: pytest.MonkeyPatch):
    empty = pd.DataFrame(columns=LIGHTCURVE_COLUMNS, dtype=float)
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: empty)
    with pytest.raises(MAXIDataFormatError, match="no data rows"):
        MAXISource._parse_lightcurve("1 2 3 4 5 6 7 8 9", "https://example.test/lc.dat")


def test_deferred_download_properties_and_refresh(tmp_path: Path):
    session = FakeSession()
    source = MAXISource(
        "Vela X-1", data_dir=tmp_path, session=session, autoload=False
    )
    assert not source.is_loaded
    assert not session.urls
    assert "loaded=False" in repr(source)

    properties = [
        lambda: source.data,
        lambda: source.source_name,
        lambda: source.product_url,
        lambda: source.raw_path,
    ]
    for getter in properties:
        with pytest.raises(MAXILabError):
            getter()

    first = source.download()
    second = source.refresh()
    assert len(first) == len(second) == 2
    assert session.urls.count("https://maxi.riken.jp/top/slist.html") == 2


def test_input_validation():
    with pytest.raises(ValueError, match="non-empty"):
        MAXISource("")
    with pytest.raises(ValueError, match="binning"):
        MAXISource("Vela X-1", binning="week")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        MAXISource("Vela X-1", timeout=0)


def test_coordinate_and_slug_helpers():
    assert _parse_coordinates("135.529, -40.555") == (135.529, -40.555)
    assert _parse_coordinates("unknown") == (None, None)
    assert _safe_slug("Vela X-1") == "Vela_X-1"
    assert _safe_slug(" ... ") == "source"


def test_atomic_write_replaces_existing_file(tmp_path: Path):
    target = tmp_path / "nested" / "file.txt"
    _atomic_write_text(target, "first")
    _atomic_write_text(target, "second")
    assert target.read_text(encoding="utf-8") == "second"
    assert not list(target.parent.glob("*.tmp"))


def test_context_manager_does_not_close_injected_session(tmp_path: Path):
    session = FakeSession()
    with MAXISource("Vela X-1", data_dir=tmp_path, session=session) as source:
        assert source.is_loaded
    assert not session.closed


def test_close_closes_owned_session(monkeypatch: pytest.MonkeyPatch):
    fake = SimpleNamespace(close=lambda: setattr(fake, "closed", True), closed=False)
    monkeypatch.setattr(MAXISource, "_build_session", staticmethod(lambda: fake))
    source = MAXISource("Vela X-1", autoload=False)
    source.close()
    assert fake.closed


def test_internal_session_configuration():
    session = MAXISource._build_session()
    try:
        assert session.headers["User-Agent"].startswith("MAXILab/0.1.0")
        adapter = session.get_adapter("https://maxi.riken.jp/")
        assert adapter.max_retries.total == 3
        assert adapter.max_retries.connect == 3
        assert adapter.max_retries.read == 3
        assert 503 in adapter.max_retries.status_forcelist
        assert adapter.max_retries.allowed_methods == frozenset({"GET"})
    finally:
        session.close()


def test_atomic_write_cleans_temp_file_if_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "file.txt"

    def fail_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("maxilab.download.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        _atomic_write_text(target, "data")

    assert not target.exists()
    assert not list(tmp_path.glob(".*.tmp"))
