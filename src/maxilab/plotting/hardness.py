"""Publication plotting for MAXI/GSC hardness ratios."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from maxilab.analysis import HardnessRatioSeries
from maxilab.runlog import get_run_logger

from .style import PublicationStyle, add_source_labels, create_figure, format_time_axis



class HardnessRatioPlotter:
    """Create publication-ready ``(H-S)/(H+S)`` figures."""

    def __init__(self, style: PublicationStyle | None = None) -> None:
        self.style = style or PublicationStyle()

    def plot(
        self,
        hardness: HardnessRatioSeries,
        *,
        axis: Axes | None = None,
        show_year_guides: bool = True,
    ) -> tuple[Figure, Axes]:
        """Plot one hardness-ratio time series with propagated uncertainties."""
        if not isinstance(hardness, HardnessRatioSeries):
            raise TypeError("hardness must be a HardnessRatioSeries instance.")

        valid = hardness.finite
        if not np.any(valid):
            raise ValueError("hardness series contains no finite measurements to plot.")

        get_run_logger().info(
            "Preparing hardness-ratio figure | source=%s | hard=%s | soft=%s | total_rows=%d | plotted_rows=%d | skipped_rows=%d",
            hardness.source_name or "unknown",
            hardness.hard_band.label,
            hardness.soft_band.label,
            hardness.n_points,
            int(np.count_nonzero(valid)),
            int(hardness.n_points - np.count_nonzero(valid)),
        )

        with self.style.context():
            if axis is None:
                figure, axis = create_figure(self.style)
            else:
                figure = axis.figure

            axis.errorbar(
                hardness.mjd[valid],
                hardness.hardness[valid],
                yerr=hardness.hardness_error[valid],
                fmt="o",
                linestyle="none",
                markersize=self.style.marker_size,
                markerfacecolor=self.style.vermilion,
                markeredgecolor="black",
                markeredgewidth=self.style.marker_edge_width,
                ecolor="black",
                elinewidth=self.style.error_line_width,
                capsize=self.style.error_capsize,
                alpha=self.style.point_alpha,
                zorder=5,
            )
            axis.axhline(
                0.0,
                color=self.style.zero_line_color,
                linewidth=0.8,
                linestyle="-",
                zorder=-50,
            )

            axis.set_xlabel("Time [MJD]")
            axis.set_ylabel(r"Hardness ratio $(H-S)/(H+S)$")

            mjd_min = float(np.nanmin(hardness.mjd[valid]))
            mjd_max = float(np.nanmax(hardness.mjd[valid]))
            format_time_axis(
                axis,
                mjd_min=mjd_min,
                mjd_max=mjd_max,
                style=self.style,
                show_year_guides=show_year_guides,
            )
            add_source_labels(
                axis,
                source_name=hardness.source_name,
                right_label=(
                    f"H: {hardness.hard_band.label} · "
                    f"S: {hardness.soft_band.label}"
                ),
                style=self.style,
            )
            figure.tight_layout()

        get_run_logger().info(
            "Created hardness-ratio figure | source=%s | hard=%s | soft=%s | mjd_min=%.5f | mjd_max=%.5f | year_guides=%s",
            hardness.source_name or "unknown",
            hardness.hard_band.label,
            hardness.soft_band.label,
            mjd_min,
            mjd_max,
            show_year_guides,
        )
        return figure, axis
