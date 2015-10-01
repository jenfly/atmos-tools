# atmos-tools

Collection of modules for working with atmospheric data. Modules are in the `atmos` sub-directory and can be imported individually or as a package using `import atmos` from the atmos-tools directory.

This package makes extensive use of the `xray` Python package for data structures (DataArray and Dataset) and for reading/writing data (NetCDF and HDF formats).

Module | Description
------- | --------- |
utils.py | General purpose utilities used by other modules
xrhelper.py | Helper functions for working with xray DataArrays and Datasets
data.py | Data wrangling utilities for file I/O, lat-lon geophysical data, pressure-level data and topography
constants.py | Constants for calculating atmospheric fields
variables.py | Functions to calculate atmospheric fields (e.g. streamfunction, potential temperature, etc.)
plots.py | Lat-lon maps and latitude-pressure plots.
analysis.py | Timeseries and linear regression analyses



:cat: :cat: :cat:
