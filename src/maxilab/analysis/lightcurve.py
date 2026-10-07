"""Light-curve extraction from validated MAXI/GSC tables.

This module contains scientific data preparation only.  Figure construction
belongs in :mod:`maxilab.plotting`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from maxilab.runlog import get_run_logger

from .bands import ENERGY_BANDS, EnergyBand, get_energy_band


_DEFAULT_FLUX_UNIT: Final = "ph s^-1 cm^-2"


@dataclass(frozen=True, slots=True)
class LightCurveSeries:
    """One MAXI/GSC light curve prepared for analysis or plotting.

    Parameters
    ----------
    mjd
        Observation times in Modified Julian Date.
    flux
        Background-subtracted photon flux values.
    flux_error
        One-sigma statistical uncertainties associated with ``flux``.
    band
        Energy-band metadata.
    source_name
        Optional canonical source name.
    flux_unit
        Unit string inherited from the downloaded MAXI table.
    """

    mjd: np.ndarray
    flux: np.ndarray
    flux_error: np.ndarray
    band: EnergyBand
    source_name: str | None
    flux_unit: str

    def __post_init__(self) -> None:
        lengths = {len(self.mjd), len(self.flux), len(self.flux_error)}
        if len(lengths) != 1:
            raise ValueError("mjd, flux, and flux_error must have identical lengths.")

    @property
    def finite(self) -> np.ndarray:
        """Boolean mask selecting finite values with non-negative uncertainties."""
        return (
            np.isfinite(self.mjd)
            & np.isfinite(self.flux)
            & np.isfinite(self.flux_error)
            & (self.flux_error >= 0.0)
        )

    @property
    def n_points(self) -> int:
        """Number of rows in the extracted light curve."""
        return len(self.mjd)

    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the series as a compact pandas DataFrame."""
        frame = pd.DataFrame(
            {
                "mjd": self.mjd.copy(),
                "flux": self.flux.copy(),
                "flux_error": self.flux_error.copy(),
            }
        )
        frame.attrs.update(
            {
                "band_keV": (self.band.lower_keV, self.band.upper_keV),
                "flux_unit": self.flux_unit,
                "source_name": self.source_name,
            }
        )
        return frame


class LightCurveAnalysis:
    """Prepare standard MAXI/GSC light curves for scientific use.

    Parameters
    ----------
    data
        Validated MAXI light-curve table, normally ``MAXISource.data``.
    source_name
        Optional canonical source name, normally ``MAXISource.source_name``.

    Notes
    -----
    The class deliberately does not modify the input table and does not apply
    implicit signal-to-noise cuts.  Negative background-subtracted fluxes are
    valid MAXI measurements and are therefore preserved.
    """

    def __init__(self, data: pd.DataFrame, *, source_name: str | None = None) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if "mjd" not in data.columns:
            raise ValueError("MAXI light-curve data must contain an 'mjd' column.")
        if data.empty:
            raise ValueError("MAXI light-curve data must contain at least one row.")

        mjd = pd.to_numeric(data["mjd"], errors="coerce")
        if mjd.isna().any():
            raise ValueError("MAXI light-curve MJD values must be finite numeric values.")
        if not mjd.is_monotonic_increasing:
            raise ValueError("MAXI light-curve MJD values must be monotonically increasing.")

        self._data = data
        self.source_name = source_name.strip() if isinstance(source_name, str) else source_name

        get_run_logger().info(
            "Initialised light-curve analysis | source=%s | rows=%d | mjd_min=%.5f | mjd_max=%.5f",
            self.source_name or "unknown",
            len(data),
            float(mjd.iloc[0]),
            float(mjd.iloc[-1]),
        )

    @property
    def available_bands(self) -> tuple[EnergyBand, ...]:
        """Standard bands whose flux and uncertainty columns are available."""
        return tuple(
            band
            for band in ENERGY_BANDS.values()
            if band.flux_column in self._data.columns and band.error_column in self._data.columns
        )

    @property
    def flux_unit(self) -> str:
        """Flux unit attached to the MAXI table, with the standard unit as fallback."""
        return str(self._data.attrs.get("flux_unit", _DEFAULT_FLUX_UNIT))

    def lightcurve(self, band: str | EnergyBand) -> LightCurveSeries:
        """Extract one standard MAXI/GSC energy-band light curve.

        Parameters
        ----------
        band
            Standard MAXI/GSC band, for example ``"2-4"`` or ``"4–10 keV"``.

        Returns
        -------
        LightCurveSeries
            Independent NumPy arrays containing MJD, flux, and uncertainty.

        Raises
        ------
        ValueError
            If columns required for the selected band are missing or contain
            non-numeric values.
        """
        resolved = get_energy_band(band)
        self._require_columns(resolved)

        mjd = self._numeric_array("mjd")
        flux = self._numeric_array(resolved.flux_column)
        error = self._numeric_array(resolved.error_column)

        finite_input = np.isfinite(mjd) & np.isfinite(flux) & np.isfinite(error)
        negative_flux = np.isfinite(flux) & (flux < 0.0)
        negative_error = np.isfinite(error) & (error < 0.0)
        get_run_logger().info(
            "Extracted light curve | band=%s | rows=%d | finite_rows=%d | negative_flux=%d | negative_error=%d | filtering=none",
            resolved.label,
            len(mjd),
            int(np.count_nonzero(finite_input)),
            int(np.count_nonzero(negative_flux)),
            int(np.count_nonzero(negative_error)),
        )
        if np.any(negative_flux):
            get_run_logger().warning(
                "Negative background-subtracted flux values detected | band=%s | count=%d | action=preserved_pending_scientific_filter_policy",
                resolved.label,
                int(np.count_nonzero(negative_flux)),
            )

        return LightCurveSeries(
            mjd=mjd,
            flux=flux,
            flux_error=error,
            band=resolved,
            source_name=self.source_name,
            flux_unit=self.flux_unit,
        )

    def _require_columns(self, band: EnergyBand) -> None:
        missing = [
            column
            for column in (band.flux_column, band.error_column)
            if column not in self._data.columns
        ]
        if missing:
            raise ValueError(
                f"MAXI data do not contain the columns required for {band.label}: "
                + ", ".join(missing)
            )

    def _numeric_array(self, column: str) -> np.ndarray:
        values = pd.to_numeric(self._data[column], errors="coerce").to_numpy(
            dtype=float,
            copy=True,
        )
        return values
