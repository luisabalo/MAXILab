"""Hardness-ratio calculations for standard MAXI/GSC energy bands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from maxilab.runlog import get_run_logger

from .bands import HARDNESS_PAIRS, EnergyBand, bands_overlap, get_energy_band
from .lightcurve import LightCurveAnalysis



@dataclass(frozen=True, slots=True)
class HardnessRatioSeries:
    """Hardness ratio and propagated statistical uncertainty.

    The ratio is defined as ``(H - S) / (H + S)``.  Uncertainties assume that
    the hard- and soft-band flux errors are independent.
    """

    mjd: np.ndarray
    hardness: np.ndarray
    hardness_error: np.ndarray
    hard_band: EnergyBand
    soft_band: EnergyBand
    source_name: str | None

    def __post_init__(self) -> None:
        lengths = {len(self.mjd), len(self.hardness), len(self.hardness_error)}
        if len(lengths) != 1:
            raise ValueError("mjd, hardness, and hardness_error must have identical lengths.")

    @property
    def finite(self) -> np.ndarray:
        """Boolean mask selecting finite hardness measurements and errors."""
        return (
            np.isfinite(self.mjd)
            & np.isfinite(self.hardness)
            & np.isfinite(self.hardness_error)
            & (self.hardness_error >= 0.0)
        )

    @property
    def n_points(self) -> int:
        """Number of rows in the hardness-ratio series."""
        return len(self.mjd)

    @property
    def label(self) -> str:
        """Compact label identifying hard and soft bands."""
        return f"H: {self.hard_band.label}; S: {self.soft_band.label}"

    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the hardness-ratio series as a DataFrame."""
        frame = pd.DataFrame(
            {
                "mjd": self.mjd.copy(),
                "hardness": self.hardness.copy(),
                "hardness_error": self.hardness_error.copy(),
            }
        )
        frame.attrs.update(
            {
                "definition": "(H-S)/(H+S)",
                "hard_band_keV": (self.hard_band.lower_keV, self.hard_band.upper_keV),
                "soft_band_keV": (self.soft_band.lower_keV, self.soft_band.upper_keV),
                "source_name": self.source_name,
            }
        )
        return frame


class HardnessRatioAnalysis:
    """Calculate MAXI/GSC hardness ratios from a common light-curve table.

    Parameters
    ----------
    lightcurves
        Prepared light-curve analysis object.  All hardness ratios are derived
        row-by-row from the same validated MAXI table and therefore share the
        same MJD sampling.
    """

    def __init__(self, lightcurves: LightCurveAnalysis) -> None:
        if not isinstance(lightcurves, LightCurveAnalysis):
            raise TypeError("lightcurves must be a LightCurveAnalysis instance.")
        self._lightcurves = lightcurves

    @classmethod
    def from_frame(
        cls,
        data: pd.DataFrame,
        *,
        source_name: str | None = None,
    ) -> HardnessRatioAnalysis:
        """Construct directly from a validated MAXI DataFrame."""
        return cls(LightCurveAnalysis(data, source_name=source_name))

    def ratio(
        self,
        hard_band: str | EnergyBand,
        soft_band: str | EnergyBand,
    ) -> HardnessRatioSeries:
        """Calculate ``(H - S) / (H + S)`` and its propagated uncertainty.

        Parameters
        ----------
        hard_band, soft_band
            Non-overlapping standard MAXI/GSC bands.  The hard band must lie at
            higher energies than the soft band.

        Returns
        -------
        HardnessRatioSeries
            Ratio values and one-sigma propagated uncertainties.

        Notes
        -----
        No scientific negative-flux filtering policy is applied yet.  Negative
        background-subtracted inputs are preserved so the filtering decision
        remains explicit and can be implemented as a documented next step.
        Consequently, the current raw ratio may fall outside [-1, 1]. Rows for
        which ``H + S`` is numerically zero, or for which an input is non-finite,
        are returned as NaN.
        """
        hard = get_energy_band(hard_band)
        soft = get_energy_band(soft_band)
        _validate_hardness_pair(hard, soft)

        hard_curve = self._lightcurves.lightcurve(hard)
        soft_curve = self._lightcurves.lightcurve(soft)

        if not np.array_equal(hard_curve.mjd, soft_curve.mjd, equal_nan=True):
            raise ValueError("Hard- and soft-band light curves do not share identical MJD sampling.")

        h = hard_curve.flux
        s = soft_curve.flux
        h_err = hard_curve.flux_error
        s_err = soft_curve.flux_error
        denominator = h + s

        ratio = np.full_like(denominator, np.nan, dtype=float)
        ratio_error = np.full_like(denominator, np.nan, dtype=float)

        valid = (
            np.isfinite(h)
            & np.isfinite(s)
            & np.isfinite(h_err)
            & np.isfinite(s_err)
            & (h_err >= 0.0)
            & (s_err >= 0.0)
            & _numerically_nonzero(denominator, h, s)
        )

        ratio[valid] = (h[valid] - s[valid]) / denominator[valid]
        ratio_error[valid] = (
            2.0
            * np.sqrt(
                (s[valid] * h_err[valid]) ** 2
                + (h[valid] * s_err[valid]) ** 2
            )
            / denominator[valid] ** 2
        )

        negative_hard = np.isfinite(h) & (h < 0.0)
        negative_soft = np.isfinite(s) & (s < 0.0)
        get_run_logger().info(
            "Calculated raw hardness ratio | hard=%s | soft=%s | rows=%d | computed=%d | undefined=%d | negative_hard=%d | negative_soft=%d | negative_flux_filter=none",
            hard.label,
            soft.label,
            len(ratio),
            int(np.count_nonzero(np.isfinite(ratio))),
            int(np.count_nonzero(~np.isfinite(ratio))),
            int(np.count_nonzero(negative_hard)),
            int(np.count_nonzero(negative_soft)),
        )
        if np.any(negative_hard) or np.any(negative_soft):
            get_run_logger().warning(
                "Hardness ratio includes rows with negative background-subtracted inputs | hard=%s | soft=%s | action=review_filter_policy_before_final_science_product",
                hard.label,
                soft.label,
            )

        return HardnessRatioSeries(
            mjd=hard_curve.mjd.copy(),
            hardness=ratio,
            hardness_error=ratio_error,
            hard_band=hard,
            soft_band=soft,
            source_name=hard_curve.source_name,
        )

    def all_standard_ratios(self) -> dict[tuple[str, str], HardnessRatioSeries]:
        """Calculate all three non-overlapping MAXI/GSC hardness combinations."""
        return {
            pair: self.ratio(hard_band=pair[0], soft_band=pair[1])
            for pair in HARDNESS_PAIRS
        }


def _validate_hardness_pair(hard: EnergyBand, soft: EnergyBand) -> None:
    if hard.key == soft.key:
        raise ValueError("Hard and soft bands must be different.")
    if bands_overlap(hard, soft):
        raise ValueError(
            f"Hardness-ratio bands must not overlap; {hard.label} and {soft.label} overlap."
        )
    if hard.lower_keV < soft.upper_keV:
        raise ValueError(
            f"hard_band must be at higher energy than soft_band; got {hard.label} and {soft.label}."
        )


def _numerically_nonzero(
    denominator: np.ndarray,
    hard: np.ndarray,
    soft: np.ndarray,
) -> np.ndarray:
    scale = np.maximum(np.abs(hard) + np.abs(soft), 1.0)
    threshold = np.finfo(float).eps * scale
    return np.abs(denominator) > threshold
