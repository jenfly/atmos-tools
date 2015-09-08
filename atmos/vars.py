import numpy as np
import xray
import atmos.utils as utils
import atmos.data as dat
from atmos.constants import const as constants

# ----------------------------------------------------------------------
def potential_temp(T, p, p0=1e5):
    """Return potential temperature.

    Parameters
    ----------
    T : ndarray or xray.DataArray
        Atmospheric temperatures in Kelvins.
    p : ndarray
        Atmospheric pressures, on same grid as T, or a 1-D array
        corresponding to the vertical levels of T.
    p0 : float, optional
        Reference pressure to use.  Must be in the same units as p.

    Returns
    -------
    theta : ndarray or xray.DataArray
        Potential temperatures in Kelvins.  If T is a DataArray, then
        theta is returned as a DataArray with the same coordinates.
        Otherwise theta is returned as an ndarray.
    """

    R = constants.R_air
    Cp = constants.Cp

    scale = (p0/plev) ** (R/Cp).values
    theta = T * dat.biggify(scale, T)

    if isinstance(theta, xray.DataArray):
        theta.attrs['long_name'] = 'Atmospheric Potential Temperature'
        theta.attrs['units'] = 'K'

    return theta


# ----------------------------------------------------------------------
