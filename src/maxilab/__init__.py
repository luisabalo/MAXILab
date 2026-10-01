"""MAXILab: MAXI Light-curve Analysis and Browsing.

Version 0.1 contains only the MAXI/GSC light-curve download layer.
"""

from .download import (
    LIGHTCURVE_COLUMNS,
    MAXI_BASE_URL,
    MAXI_SOURCE_LIST_URL,
    DownloadInfo,
    MAXIConnectionError,
    MAXIDataFormatError,
    MAXILabError,
    MAXISource,
    SourceInfo,
    SourceNotFoundError,
)

__all__ = [
    "DownloadInfo",
    "LIGHTCURVE_COLUMNS",
    "MAXI_BASE_URL",
    "MAXI_SOURCE_LIST_URL",
    "MAXIConnectionError",
    "MAXIDataFormatError",
    "MAXILabError",
    "MAXISource",
    "SourceInfo",
    "SourceNotFoundError",
]

__version__ = "0.1.0"
