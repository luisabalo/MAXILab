"""Publication plotting for MAXI/GSC light curves."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from maxilab.analysis import LightCurveSeries
from maxilab.runlog import get_run_logger

from .style import PublicationStyle, add_source_labels, create_figure, format_time_axis



class LightCurvePlotter:
    """Create publication-ready single-band MAXI/GSC light-curve figures.

    Parameters
    ----------
    style
        Optional :class:`PublicationStyle`.  Defaults reproduce the canonical
        MAXILab publication layout.
    """

    def __init__(self, style: PublicationStyle | None = None) -> None:
        self.style = style or PublicationStyle()

    def plot(
        self,
        lightcurve: LightCurveSeries,
        *,
        axis: Axes | None = None,
        show_year_guides: bool = True,
    ) -> tuple[Figure, Axes]:
        """Plot one energy-band light curve with one-sigma error bars.

        No signal-to-noise filtering is applied.  The plotted points are the
        finite measurements returned by the analysis layer.
        """
        if not isinstance(lightcurve, LightCurveSeries):
            raise TypeError("lightcurve must be a LightCurveSeries instance.")

        valid = lightcurve.finite
        if not np.any(valid):
            raise ValueError("lightcurve contains no finite measurements to plot.")

        get_run_logger().info(
            "Preparing light-curve figure | source=%s | band=%s | total_rows=%d | plotted_rows=%d | skipped_rows=%d",
            lightcurve.source_name or "unknown",
            lightcurve.band.label,
            lightcurve.n_points,
            int(np.count_nonzero(valid)),
            int(lightcurve.n_points - np.count_nonzero(valid)),
        )

        with self.style.context():
            if axis is None:
                figure, axis = create_figure(self.style)
            else:
                figure = axis.figure

            axis.errorbar(
                lightcurve.mjd[valid],
                lightcurve.flux[valid],
                yerr=lightcurve.flux_error[valid],
                fmt="o",
                linestyle="none",
                markersize=self.style.marker_size,
                markerfacecolor=self.style.blue,
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
            axis.set_ylabel(
                "MAXI/GSC flux "
                r"[ph s$^{-1}$ cm$^{-2}$]"
                f"\n{lightcurve.band.label}"
            )

            mjd_min = float(np.nanmin(lightcurve.mjd[valid]))
            mjd_max = float(np.nanmax(lightcurve.mjd[valid]))
            format_time_axis(
                axis,
                mjd_min=mjd_min,
                mjd_max=mjd_max,
                style=self.style,
                show_year_guides=show_year_guides,
            )
            add_source_labels(
                axis,
                source_name=lightcurve.source_name,
                right_label=f"MAXI/GSC · {lightcurve.band.label}",
                style=self.style,
            )
            figure.tight_layout()

        get_run_logger().info(
            "Created light-curve figure | source=%s | band=%s | mjd_min=%.5f | mjd_max=%.5f | year_guides=%s",
            lightcurve.source_name or "unknown",
            lightcurve.band.label,
            mjd_min,
            mjd_max,
            show_year_guides,
        )
        return figure, axis
