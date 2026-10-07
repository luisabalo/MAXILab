"""Per-run provenance logging for MAXILab analysis workflows.

The run log is intentionally separate from scientific calculations.  New
MAXILab modules emit concise operation records to the ``maxilab.run`` logger;
wrapping a script in :class:`RunLog` writes those records to one timestamped
``.log`` file that can later be consulted while preparing a paper or checking
how a figure was produced.
"""

from __future__ import annotations

import logging
import os
import platform
import shlex
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final

_LOGGER_NAME: Final = "maxilab.run"


def get_run_logger() -> logging.Logger:
    """Return the package logger used for MAXILab run provenance.

    The logger is silent unless a :class:`RunLog` context is active, so library
    users are never forced to create log files.  Analysis and plotting modules
    use this function to report operations without owning the output file.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.disabled = False
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


@dataclass(slots=True)
class RunLog:
    """Write one timestamped, human-readable log for a MAXILab run.

    Parameters
    ----------
    output_dir
        Directory in which the log file is created.  The default ``logs``
        directory is intended to be ignored by Git.
    prefix
        Filename prefix.  The default creates names such as
        ``MAXILab_20261007_161530.log``.

    Notes
    -----
    Use this object as a context manager.  It records basic reproducibility
    metadata at startup, receives operation messages from MAXILab analysis and
    plotting modules, and writes a completion/exception record when the run
    finishes.
    """

    output_dir: str | Path = "logs"
    prefix: str = "MAXILab"
    path: Path | None = field(init=False, default=None)
    _handler: logging.FileHandler | None = field(init=False, default=None, repr=False)
    _started_monotonic: float | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    def __enter__(self) -> RunLog:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now().astimezone()
        self.path = _unique_log_path(self.output_dir, self.prefix, now)

        logger = get_run_logger()
        self._handler = logging.FileHandler(self.path, encoding="utf-8")
        self._handler.setLevel(logging.INFO)
        self._handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(self._handler)
        self._started_monotonic = time.monotonic()

        self.info("MAXILab run started")
        self.info(
            "Run environment",
            started=now.isoformat(timespec="seconds"),
            python=platform.python_version(),
            platform=platform.platform(),
            maxilab_version=_maxilab_version(),
            working_directory=os.getcwd(),
            command=" ".join(shlex.quote(arg) for arg in sys.argv),
        )
        self.info("Log file created", path=self.path)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:  # type: ignore[no-untyped-def]
        elapsed = (
            time.monotonic() - self._started_monotonic
            if self._started_monotonic is not None
            else float("nan")
        )
        logger = get_run_logger()

        if exc_type is None:
            self.info("MAXILab run completed", status="success", elapsed_seconds=f"{elapsed:.3f}")
        else:
            logger.exception(
                "MAXILab run failed | status=error | elapsed_seconds=%.3f",
                elapsed,
                exc_info=(exc_type, exc_value, traceback),
            )

        if self._handler is not None:
            logger.removeHandler(self._handler)
            self._handler.close()
            self._handler = None
        return False

    def info(self, message: str, /, **fields: Any) -> None:
        """Append a structured informational entry to the active run log."""
        get_run_logger().info(_format_message(message, fields))

    def warning(self, message: str, /, **fields: Any) -> None:
        """Append a structured warning entry to the active run log."""
        get_run_logger().warning(_format_message(message, fields))


def _format_message(message: str, fields: dict[str, Any]) -> str:
    if not fields:
        return message
    rendered = " | ".join(f"{key}={_render_value(value)}" for key, value in fields.items())
    return f"{message} | {rendered}"


def _render_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _unique_log_path(output_dir: Path, prefix: str, timestamp: datetime) -> Path:
    safe_prefix = "".join(char if char.isalnum() or char in "-_" else "_" for char in prefix)
    safe_prefix = safe_prefix.strip("_") or "MAXILab"
    stem = f"{safe_prefix}_{timestamp:%Y%m%d_%H%M%S}"
    candidate = output_dir / f"{stem}.log"
    counter = 1
    while candidate.exists():
        candidate = output_dir / f"{stem}_{counter:02d}.log"
        counter += 1
    return candidate


def _maxilab_version() -> str:
    try:
        return version("maxilab")
    except PackageNotFoundError:
        return "unknown (package metadata unavailable)"
