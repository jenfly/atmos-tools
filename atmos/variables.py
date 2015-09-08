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

    scale = (p0/p) ** (R/Cp).values
    theta = T * dat.biggify(scale, T)

    if isinstance(T, xray.DataArray):
        theta.attrs['long_name'] = 'Potential Temperature'
        theta.attrs['units'] = 'K'

    return theta


# ----------------------------------------------------------------------
def equiv_potential_temp(T, p, q, p0=1e5):
    """Return potential temperature.

    Parameters
    ----------
    T : ndarray or xray.DataArray
        Atmospheric temperatures in Kelvins.
    p : ndarray
        Atmospheric pressures, on same grid as T, or a 1-D array
        corresponding to the vertical levels of T.
    q : ndarray
        Atmospheric specific humidity, on same grid as T, in units
        of kg/kg.
    p0 : float, optional
        Reference pressure to use.  Must be in the same units as p.

    Returns
    -------
    theta_e : ndarray or xray.DataArray
        Equivalent potential temperatures in Kelvins.  If T is a
        DataArray, then theta_e is returned as a DataArray with the
        same coordinates. Otherwise theta_e is returned as an ndarray.

    Notes
    -----
    Equivalent potential temperature is computed according to the
    definition in Equation (4-30) in:
    Marshall, John, and R. Alan Plumb. Atmosphere, Ocean, and Climate
        Dynamics: An Introductory Text. Amsterdam: Elsevier Academic
        Press, 2008.
    """

    L = constants.Lv
    Cp = constants.Cp

    theta = potential_temp(T, p, p0)
    theta_e = theta * np.exp((L*q) / (Cp*T))

    if isinstance(T, xray.DataArray):
        theta_e.attrs['long_name'] = 'Equivalent Potential Temperature'
        theta_e.attrs['units'] = 'K'

    return theta_e


# ----------------------------------------------------------------------
