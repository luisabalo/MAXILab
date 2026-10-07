"""Energy-band definitions shared by MAXILab analysis tools."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping


@dataclass(frozen=True, slots=True)
class EnergyBand:
    """Description of one standard MAXI/GSC light-curve energy band.

    Parameters
    ----------
    key
        Compact identifier used by the public MAXILab API.
    lower_keV, upper_keV
        Lower and upper energy boundaries in keV.
    flux_column, error_column
        Column names in the validated MAXI light-curve table.
    """

    key: str
    lower_keV: float
    upper_keV: float
    flux_column: str
    error_column: str

    @property
    def label(self) -> str:
        """Human-readable energy-band label."""
        return f"{_format_energy(self.lower_keV)}–{_format_energy(self.upper_keV)} keV"

    @property
    def filename_token(self) -> str:
        """Filesystem-friendly representation of the band."""
        return self.key.replace("-", "_")


_BANDS: Final[dict[str, EnergyBand]] = {
    "2-20": EnergyBand("2-20", 2.0, 20.0, "flux_2_20", "err_2_20"),
    "2-4": EnergyBand("2-4", 2.0, 4.0, "flux_2_4", "err_2_4"),
    "4-10": EnergyBand("4-10", 4.0, 10.0, "flux_4_10", "err_4_10"),
    "10-20": EnergyBand("10-20", 10.0, 20.0, "flux_10_20", "err_10_20"),
}

ENERGY_BANDS: Final[Mapping[str, EnergyBand]] = MappingProxyType(_BANDS)
"""Read-only mapping of the standard MAXI/GSC energy bands."""

HARDNESS_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("4-10", "2-4"),
    ("10-20", "2-4"),
    ("10-20", "4-10"),
)
"""Scientifically meaningful non-overlapping ``(hard, soft)`` band pairs."""


def get_energy_band(value: str | EnergyBand) -> EnergyBand:
    """Return a canonical :class:`EnergyBand` from a user-supplied value.

    The string matcher accepts common variants such as ``"2–4 keV"`` and
    ``"2_4"`` while preserving one canonical key internally.

    Parameters
    ----------
    value
        Energy-band object or string representation.

    Raises
    ------
    ValueError
        If the requested band is not one of the standard MAXI/GSC bands.
    """
    if isinstance(value, EnergyBand):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("energy band must be a non-empty string or EnergyBand.")

    key = _normalise_band_key(value)
    try:
        return ENERGY_BANDS[key]
    except KeyError as exc:
        allowed = ", ".join(ENERGY_BANDS)
        raise ValueError(f"Unknown MAXI energy band {value!r}. Choose from: {allowed}.") from exc


def bands_overlap(first: EnergyBand, second: EnergyBand) -> bool:
    """Return whether two bands overlap over a finite energy interval."""
    return max(first.lower_keV, second.lower_keV) < min(first.upper_keV, second.upper_keV)


def _normalise_band_key(value: str) -> str:
    key = value.strip().casefold()
    key = key.replace("kev", "")
    key = key.translate(str.maketrans({"−": "-", "–": "-", "—": "-", "_": "-"}))
    key = "".join(key.split())
    return key


def _format_energy(value: float) -> str:
    return f"{value:g}"
