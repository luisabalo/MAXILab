"""Scientific analysis tools for MAXI/GSC light curves."""

from .bands import ENERGY_BANDS, HARDNESS_PAIRS, EnergyBand, get_energy_band
from .hardness import HardnessRatioAnalysis, HardnessRatioSeries
from .lightcurve import LightCurveAnalysis, LightCurveSeries

__all__ = [
    "ENERGY_BANDS",
    "HARDNESS_PAIRS",
    "EnergyBand",
    "HardnessRatioAnalysis",
    "HardnessRatioSeries",
    "LightCurveAnalysis",
    "LightCurveSeries",
    "get_energy_band",
]
