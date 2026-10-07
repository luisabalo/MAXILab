"""Example: publication-ready 2–4 keV MAXI/GSC light curve for Vela X-1."""

from maxilab import MAXISource
from maxilab.analysis import LightCurveAnalysis
from maxilab.plotting import LightCurvePlotter, save_publication_figure
from maxilab.runlog import RunLog


with RunLog(output_dir="logs") as run:
    run.info("Requesting MAXI source", source="Vela X-1", binning="day")
    source = MAXISource("Vela X-1", binning="day")
    run.info(
        "MAXI source ready",
        source=source.source_name,
        rows=len(source.data),
        binning="day",
        coordinates_deg=source.coordinates,
        source_page_url=source.source_page_url,
        product_url=source.product_url,
        downloaded_at_utc=source.downloaded_at_utc.isoformat(),
        latest_mjd=source.latest_mjd,
        raw_path=source.raw_path,
        metadata_path=source.metadata_path,
        sha256=source.sha256,
        flux_unit=source.data.attrs.get("flux_unit", "unknown"),
        time_scale=source.data.attrs.get("time_scale", "unknown"),
    )

    analysis = LightCurveAnalysis(source.data, source_name=source.source_name)
    lightcurve = analysis.lightcurve("2-4")

    plotter = LightCurvePlotter()
    figure, axis = plotter.plot(lightcurve)

    files = save_publication_figure(
        figure,
        "vela_x1_MAXI_2_4keV",
        formats=("pdf", "png"),
        dpi=600,
    )
    run.info("Example workflow finished", outputs=", ".join(str(path) for path in files))
