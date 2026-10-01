"""Download and validate public MAXI/GSC standard light curves.

This module is intentionally limited to data access.  Scientific analysis and
plotting belong in separate MAXILab modules so that changes to the MAXI website
do not become entangled with analysis code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import get_close_matches
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Final, Literal
import unicodedata
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

Binning = Literal["orbit", "day"]

MAXI_BASE_URL: Final = "https://maxi.riken.jp/"
MAXI_SOURCE_LIST_URL: Final = urljoin(MAXI_BASE_URL, "top/slist.html")

LIGHTCURVE_COLUMNS: Final[tuple[str, ...]] = (
    "mjd",
    "flux_2_20",
    "err_2_20",
    "flux_2_4",
    "err_2_4",
    "flux_4_10",
    "err_4_10",
    "flux_10_20",
    "err_10_20",
)

_USER_AGENT: Final = "MAXILab/0.1.0 (independent scientific MAXI/GSC client)"


class MAXILabError(RuntimeError):
    """Base exception for MAXILab data-access errors."""


class MAXIConnectionError(MAXILabError):
    """Raised when a MAXI web resource cannot be retrieved."""


class SourceNotFoundError(MAXILabError):
    """Raised when a source cannot be resolved unambiguously."""


class MAXIDataFormatError(MAXILabError):
    """Raised when a MAXI page or light curve has an unexpected format."""


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Canonical source metadata parsed from the MAXI source catalogue."""

    name: str
    ra_deg: float | None
    dec_deg: float | None
    page_url: str


@dataclass(frozen=True, slots=True)
class DownloadInfo:
    """Provenance for the currently loaded light curve."""

    product_url: str
    downloaded_at_utc: datetime
    raw_path: Path
    metadata_path: Path
    sha256: str


class MAXISource:
    """Resolve a source in MAXI and download its latest standard light curve.

    Parameters
    ----------
    name
        Source name from the MAXI source list. Matching is case-insensitive
        and tolerant of whitespace and punctuation differences such as
        ``"Vela X-1"`` versus ``"vela x1"``. Fuzzy matches are suggested but
        never selected automatically.
    binning
        ``"orbit"`` (default) for the one-ISS-orbit light curve or ``"day"``
        for the one-day light curve.
    data_dir
        Directory in which the raw ``.dat`` file and provenance JSON are
        preserved. Defaults to ``~/.cache/maxilab``.
    timeout
        HTTP timeout in seconds for each request.
    autoload
        When ``True`` (default), construction immediately resolves the source
        and downloads the current light curve.
    session
        Optional ``requests.Session``-compatible object. Primarily useful for
        testing or custom networking configuration.

    Examples
    --------
    >>> from maxilab import MAXISource
    >>> vela = MAXISource("Vela X-1")
    >>> vela.data.head()
    >>> vela.latest_mjd
    """

    def __init__(
        self,
        name: str,
        *,
        binning: Binning = "orbit",
        data_dir: str | Path | None = None,
        timeout: float = 30.0,
        autoload: bool = True,
        session: requests.Session | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string.")
        if binning not in {"orbit", "day"}:
            raise ValueError("binning must be either 'orbit' or 'day'.")
        if timeout <= 0:
            raise ValueError("timeout must be positive.")

        self.query_name = name.strip()
        self.binning: Binning = binning
        self.data_dir = (
            Path(data_dir).expanduser()
            if data_dir is not None
            else Path.home() / ".cache" / "maxilab"
        )
        self.timeout = float(timeout)
        self._session = session if session is not None else self._build_session()
        self._owns_session = session is None

        self._source: SourceInfo | None = None
        self._data: pd.DataFrame | None = None
        self._download: DownloadInfo | None = None

        if autoload:
            self.download()

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.4,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"User-Agent": _USER_AGENT})
        return session

    def download(self) -> pd.DataFrame:
        """Download the latest selected MAXI light curve and return it.

        The source catalogue and source page are resolved afresh on every call,
        so MAXILab does not assume that MAXI source identifiers or product URLs
        remain fixed forever. Object state is updated only after the complete
        download, validation, and local write succeed.
        """
        source = self._resolve_source(self.query_name)
        product_url = self._resolve_lightcurve_url(source.page_url, self.binning)
        raw_text = self._get_text(product_url)
        frame = self._parse_lightcurve(raw_text, product_url)

        downloaded_at = datetime.now(timezone.utc)
        digest = sha256(raw_text.encode("utf-8")).hexdigest()
        raw_path, metadata_path = self._save_download(
            source=source,
            raw_text=raw_text,
            product_url=product_url,
            downloaded_at=downloaded_at,
            digest=digest,
            rows=len(frame),
            latest_mjd=float(frame["mjd"].iloc[-1]),
        )

        self._source = source
        self._data = frame
        self._download = DownloadInfo(
            product_url=product_url,
            downloaded_at_utc=downloaded_at,
            raw_path=raw_path,
            metadata_path=metadata_path,
            sha256=digest,
        )
        return frame

    def refresh(self) -> pd.DataFrame:
        """Alias for :meth:`download`, emphasizing retrieval of current data."""
        return self.download()

    def _resolve_source(self, query: str) -> SourceInfo:
        html = self._get_text(MAXI_SOURCE_LIST_URL)
        soup = BeautifulSoup(html, "html.parser")
        sources: list[SourceInfo] = []

        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 3:
                continue

            name = cells[0].get_text(" ", strip=True)
            standard_link = cells[2].find("a", href=True)
            if not name or standard_link is None:
                continue

            ra_deg, dec_deg = _parse_coordinates(cells[1].get_text(" ", strip=True))
            sources.append(
                SourceInfo(
                    name=name,
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    page_url=urljoin(MAXI_SOURCE_LIST_URL, standard_link["href"]),
                )
            )

        if not sources:
            raise MAXIDataFormatError(
                "The MAXI source list was retrieved, but no standard-product "
                "sources could be parsed. The MAXI page structure may have changed."
            )

        exact_key = _normalise_name(query)
        exact = [source for source in sources if _normalise_name(source.name) == exact_key]
        if len(exact) == 1:
            return exact[0]

        loose_key = _loose_name(query)
        loose = [source for source in sources if _loose_name(source.name) == loose_key]
        if len(loose) == 1:
            return loose[0]
        if len(loose) > 1:
            options = ", ".join(source.name for source in loose)
            raise SourceNotFoundError(
                f"Source name {query!r} is ambiguous in the MAXI source list: "
                f"{options}. Use the catalogue spelling."
            )

        suggestions = get_close_matches(query, [s.name for s in sources], n=5, cutoff=0.45)
        hint = f" Close matches: {', '.join(suggestions)}." if suggestions else ""
        raise SourceNotFoundError(
            f"Source {query!r} was not found among MAXI sources with a standard product."
            f"{hint}"
        )

    def _resolve_lightcurve_url(self, source_page_url: str, binning: Binning) -> str:
        html = self._get_text(source_page_url)
        soup = BeautifulSoup(html, "html.parser")
        token = "1orb" if binning == "orbit" else "1day"

        candidates: list[str] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            href_key = href.casefold()
            text_key = _normalise_name(anchor.get_text(" ", strip=True))
            matches_binning = token in href_key or (
                binning == "orbit" and "1orbit" in text_key
            ) or (binning == "day" and "1day" in text_key)
            if matches_binning and href_key.endswith(".dat"):
                candidates.append(urljoin(source_page_url, href))

        candidates = list(dict.fromkeys(candidates))
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise MAXIDataFormatError(
                f"Could not find a machine-readable {binning!r} light-curve file on "
                f"{source_page_url}. The MAXI page structure may have changed."
            )
        raise MAXIDataFormatError(
            f"Found multiple {binning!r} light-curve files on {source_page_url}: "
            + ", ".join(candidates)
        )

    def _get_text(self, url: str) -> str:
        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise MAXIConnectionError(f"Could not retrieve {url}: {exc}") from exc
        return response.content.decode("utf-8", errors="replace")

    @staticmethod
    def _parse_lightcurve(text: str, url: str) -> pd.DataFrame:
        rows = [
            line
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not rows:
            raise MAXIDataFormatError(f"MAXI light curve from {url} contains no data rows.")

        for row_number, line in enumerate(rows, start=1):
            column_count = len(line.split())
            if column_count != len(LIGHTCURVE_COLUMNS):
                raise MAXIDataFormatError(
                    f"Expected exactly {len(LIGHTCURVE_COLUMNS)} columns in MAXI light "
                    f"curve from {url}; data line {row_number} has {column_count}."
                )

        try:
            frame = pd.read_csv(
                StringIO("\n".join(rows)),
                sep=r"\s+",
                header=None,
                names=list(LIGHTCURVE_COLUMNS),
                dtype=float,
                engine="python",
            )
        except (TypeError, ValueError) as exc:
            raise MAXIDataFormatError(
                f"Could not parse numeric MAXI light curve from {url}."
            ) from exc

        if frame.empty:
            raise MAXIDataFormatError(f"MAXI light curve from {url} contains no data rows.")
        if frame["mjd"].isna().any():
            raise MAXIDataFormatError(f"Missing MJD values found in MAXI data from {url}.")
        if frame["mjd"].duplicated().any():
            raise MAXIDataFormatError(f"Duplicate MJD rows found in MAXI data from {url}.")
        if not frame["mjd"].is_monotonic_increasing:
            raise MAXIDataFormatError(f"MJD values are not monotonically increasing in {url}.")

        frame.attrs.update(
            {
                "source_url": url,
                "flux_unit": "ph s^-1 cm^-2",
                "time_scale": "MJD (ISS time)",
                "bands_keV": {
                    "flux_2_20": (2.0, 20.0),
                    "flux_2_4": (2.0, 4.0),
                    "flux_4_10": (4.0, 10.0),
                    "flux_10_20": (10.0, 20.0),
                },
            }
        )
        return frame

    def _save_download(
        self,
        *,
        source: SourceInfo,
        raw_text: str,
        product_url: str,
        downloaded_at: datetime,
        digest: str,
        rows: int,
        latest_mjd: float,
    ) -> tuple[Path, Path]:
        directory = self.data_dir / _safe_slug(source.name)
        raw_path = directory / f"maxi_{self.binning}.dat"
        metadata_path = directory / f"maxi_{self.binning}.json"

        _atomic_write_text(raw_path, raw_text)
        metadata = {
            "query_name": self.query_name,
            "source_name": source.name,
            "ra_deg": source.ra_deg,
            "dec_deg": source.dec_deg,
            "source_page_url": source.page_url,
            "product_url": product_url,
            "binning": self.binning,
            "downloaded_at_utc": downloaded_at.isoformat(),
            "rows": rows,
            "latest_mjd": latest_mjd,
            "sha256": digest,
            "raw_path": str(raw_path),
        }
        _atomic_write_text(metadata_path, json.dumps(metadata, indent=2) + "\n")
        return raw_path, metadata_path

    @property
    def data(self) -> pd.DataFrame:
        """Validated light curve as a pandas DataFrame."""
        return self._require_data()

    @property
    def source_name(self) -> str:
        """Canonical source name returned by the MAXI source list."""
        return self._require_source().name

    @property
    def coordinates(self) -> tuple[float | None, float | None]:
        """Catalogue coordinates as ``(RA, Dec)`` in degrees."""
        source = self._require_source()
        return source.ra_deg, source.dec_deg

    @property
    def source_page_url(self) -> str:
        """MAXI standard-product page for the resolved source."""
        return self._require_source().page_url

    @property
    def product_url(self) -> str:
        """URL of the currently loaded MAXI light-curve file."""
        return self._require_download().product_url

    @property
    def latest_mjd(self) -> float:
        """Largest MJD in the currently loaded light curve."""
        return float(self._require_data()["mjd"].iloc[-1])

    @property
    def downloaded_at_utc(self) -> datetime:
        """UTC timestamp at which the current file was downloaded."""
        return self._require_download().downloaded_at_utc

    @property
    def raw_path(self) -> Path:
        """Path to the preserved raw MAXI ``.dat`` file."""
        return self._require_download().raw_path

    @property
    def metadata_path(self) -> Path:
        """Path to the provenance JSON accompanying the raw data."""
        return self._require_download().metadata_path

    @property
    def sha256(self) -> str:
        """SHA-256 checksum of the downloaded raw text."""
        return self._require_download().sha256

    @property
    def is_loaded(self) -> bool:
        """Whether source resolution and download completed successfully."""
        return self._source is not None and self._data is not None and self._download is not None

    def _require_source(self) -> SourceInfo:
        if self._source is None:
            raise MAXILabError("The source is not resolved. Call download() first.")
        return self._source

    def _require_data(self) -> pd.DataFrame:
        if self._data is None:
            raise MAXILabError("No light curve is loaded. Call download() first.")
        return self._data

    def _require_download(self) -> DownloadInfo:
        if self._download is None:
            raise MAXILabError("No light curve is loaded. Call download() first.")
        return self._download

    def close(self) -> None:
        """Close the HTTP session when MAXILab created it internally."""
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> MAXISource:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def __repr__(self) -> str:
        if not self.is_loaded:
            return f"MAXISource(name={self.query_name!r}, binning={self.binning!r}, loaded=False)"
        return (
            f"MAXISource(name={self.source_name!r}, binning={self.binning!r}, "
            f"rows={len(self.data)}, latest_mjd={self.latest_mjd:.5f})"
        )


def _parse_coordinates(text: str) -> tuple[float | None, float | None]:
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)", text)
    if match is None:
        return None, None
    return float(match.group(1)), float(match.group(2))


def _normalise_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.translate(str.maketrans({"−": "-", "–": "-", "—": "-"}))
    return " ".join(value.casefold().split())


def _loose_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalise_name(value))


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    return slug or "source"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)
