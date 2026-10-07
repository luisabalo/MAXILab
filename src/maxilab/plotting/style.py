"""Shared publication-style configuration for MAXILab figures."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from maxilab.runlog import get_run_logger


_MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class PublicationStyle:
    """Visual defaults for publication-ready MAXILab figures.

    The defaults follow the figure language used in the MAXILab author's
    publication plots: readable 14-pt axis labels, restrained Okabe-Ito colors,
    black marker edges and error bars, inward ticks, and vector-friendly output.
    """

    font_size: float = 14.0
    small_font_size: float = 11.0
    figure_size: tuple[float, float] = (10.0, 4.8)
    marker_size: float = 3.2
    marker_edge_width: float = 0.25
    error_line_width: float = 0.55
    error_capsize: float = 0.0
    point_alpha: float = 0.72
    blue: str = "#0072B2"
    orange: str = "#E69F00"
    green: str = "#009E73"
    vermilion: str = "#D55E00"
    purple: str = "#CC79A7"
    year_line_color: str = "0.86"
    zero_line_color: str = "0.45"

    @property
    def rc_params(self) -> dict[str, object]:
        """Matplotlib rc parameters applied only inside MAXILab plotting calls."""
        return {
            "font.size": self.small_font_size,
            "axes.labelsize": self.font_size,
            "xtick.labelsize": self.font_size,
            "ytick.labelsize": self.font_size,
            "legend.fontsize": self.small_font_size,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 6.0,
            "ytick.major.size": 6.0,
            "xtick.minor.size": 3.0,
            "ytick.minor.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "savefig.bbox": "tight",
        }

    @contextmanager
    def context(self) -> Iterator[None]:
        """Apply the MAXILab style without mutating global Matplotlib state."""
        with mpl.rc_context(self.rc_params):
            yield


@dataclass(frozen=True, slots=True)
class FigureFiles:
    """Paths written by :func:`save_publication_figure`."""

    files: tuple[Path, ...]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.files)


def create_figure(style: PublicationStyle) -> tuple[Figure, Axes]:
    """Create one single-panel figure using the supplied publication style."""
    figure, axis = plt.subplots(figsize=style.figure_size)
    return figure, axis


def format_time_axis(
    axis: Axes,
    *,
    mjd_min: float,
    mjd_max: float,
    style: PublicationStyle,
    show_year_guides: bool = True,
) -> None:
    """Apply common MJD-axis formatting and optional calendar-year guides."""
    axis.set_xlim(mjd_min, mjd_max)
    axis.minorticks_on()
    axis.tick_params(which="both", direction="in", top=True, right=True)
    axis.ticklabel_format(axis="x", style="plain", useOffset=False)

    for spine in axis.spines.values():
        spine.set_linewidth(0.8)

    if show_year_guides:
        _add_year_guides(axis, mjd_min=mjd_min, mjd_max=mjd_max, style=style)


def add_source_labels(
    axis: Axes,
    *,
    source_name: str | None,
    right_label: str,
    style: PublicationStyle,
) -> None:
    """Add compact in-axis source/instrument labels without a figure title."""
    if source_name:
        axis.text(
            0.015,
            0.965,
            source_name,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=style.font_size,
        )
    axis.text(
        0.985,
        0.965,
        right_label,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=style.small_font_size,
    )


def save_publication_figure(
    figure: Figure,
    output_stem: str | Path,
    *,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 600,
) -> FigureFiles:
    """Save a figure in vector PDF and/or high-resolution raster formats.

    Parameters
    ----------
    figure
        Matplotlib figure to save.
    output_stem
        Destination path without a suffix, for example ``"vela_x1_2_4keV"``.
    formats
        File suffixes to create.  PDF and PNG are the intended defaults.
    dpi
        Raster resolution.  This does not affect vector content in PDF output.
    """
    stem = Path(output_stem)
    if stem.suffix:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for file_format in formats:
        suffix = file_format.casefold().lstrip(".")
        if not suffix:
            raise ValueError("figure formats must be non-empty strings.")
        output = stem.with_suffix(f".{suffix}")
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        written.append(output)
        get_run_logger().info(
            "Saved publication figure | path=%s | format=%s | dpi=%d",
            output,
            suffix,
            dpi,
        )

    return FigureFiles(tuple(written))


def _add_year_guides(
    axis: Axes,
    *,
    mjd_min: float,
    mjd_max: float,
    style: PublicationStyle,
) -> None:
    start_year = _mjd_to_datetime(mjd_min).year
    end_year = _mjd_to_datetime(mjd_max).year

    for year in range(start_year, end_year + 1):
        year_mjd = _datetime_to_mjd(datetime(year, 1, 1, tzinfo=timezone.utc))
        if year_mjd <= mjd_min or year_mjd >= mjd_max:
            continue
        axis.axvline(
            year_mjd,
            color=style.year_line_color,
            linestyle="--",
            linewidth=0.65,
            zorder=-100,
        )
        axis.text(
            year_mjd + 10.0,
            0.955,
            str(year),
            transform=axis.get_xaxis_transform(),
            rotation=90,
            color="0.48",
            fontsize=style.small_font_size - 1.0,
            ha="left",
            va="top",
            zorder=-90,
        )


def _mjd_to_datetime(mjd: float) -> datetime:
    return _MJD_EPOCH + timedelta(days=float(mjd))


def _datetime_to_mjd(value: datetime) -> float:
    return (value - _MJD_EPOCH).total_seconds() / 86400.0
