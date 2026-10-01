# MAXILab

**MAXI Light-curve Analysis and Browsing**

MAXILab is a Python package for accessing public MAXI/GSC light curves.

Version **0.1.1** contains the data-access layer only: give MAXILab a source name and it resolves the source in the live MAXI catalogue, downloads the latest standard light curve, validates it, and returns it as a `pandas.DataFrame`.

MAXILab is an independent project and is not an official MAXI-team package. 

## Install

From PyPI, once released:

```bash
pip install maxilab
```

For development from a local clone:

```bash
python -m pip install -e '.[test]'
```

## Quick start

```python
from maxilab import MAXISource

vela = MAXISource("Vela X-1")

print(vela.data.head())
print(vela.data.tail())
print(vela.latest_mjd)
```

The default product is the **1-ISS-orbit** MAXI/GSC light curve.

For the 1-day product:

```python
vela = MAXISource("Vela X-1", binning="day")
```

The returned table contains:

```text
mjd
flux_2_20   err_2_20
flux_2_4    err_2_4
flux_4_10   err_4_10
flux_10_20  err_10_20
```

Raw MAXI data and a provenance JSON file are cached locally under:

```text
~/.cache/maxilab/
```

## Testing

Install the test dependencies and run:

```bash
python -m pip install -e '.[test]'
pytest -m 'not live'
```

The test suite enforces **>90% coverage**. The current release has 100% statement and branch coverage in the deterministic test suite.

To additionally test a real connection to the MAXI website:

```bash
MAXILAB_RUN_LIVE=1 pytest -m live --no-cov
```

## Current scope

Version 0.1.1 only handles source resolution, download, validation, local storage, and provenance. Analysis and plotting tools will be added in later releases.

MAXI source catalogue: https://maxi.riken.jp/top/slist.html