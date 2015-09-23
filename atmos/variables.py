import numpy as np
import xray
import atmos.utils as utils
import atmos.data as dat
import atmos.xrhelper as xr
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
def moisture_flux_conv(uq, vq, lat=None, lon=None, plev=None, pdim=-3,
                       pmin=0, pmax=1e6, return_comp=False):
    """Return the vertically integrated moisture flux convergence.

    Parameters
    ----------
    uq : ndarray or xray.DataArray
        Zonal moisture flux, with latitude as the second-last dimension,
        longitude as the last dimension.
        i.e. zonal wind (m/s) * specific humidity (kg/kg)
    vq : ndarray or xray.DataArray
        Meridional moisture flux, with latitude as the second-last dimension,
        longitude as the last dimension.
        i.e. meridional wind (m/s) * specific humidity (kg/kg)
    lat, lon : ndarray, optional
        Latitudes and longitudes in degrees.  If omitted, then uq and
        vq must be xray.DataArrays and the coordinates are extracted
        from them.
    plev : ndarray, optional
        Pressure levels in Pascals.  If omitted, then extracted from
        DataArray inputs.
    pdim : int, optional
        Dimension of pressure levels in uq and vq.
    pmin, pmax : float, optional
        Lower and upper bounds (inclusive) of pressure levels (Pa)
        to include in integration.
    return_comp, bool, optional
        If True, return additional components, otherwise just total
        moisture flux convergence.

    Returns
    -------
    If return_comp is False:
    mfc : ndarray or xray.DataArray
        Vertically integrated moisture flux convergence in mm/day.

    If return_comp is True:
    mfc, mfc_x, mfc_y, uq_int, vq_int : ndarrays or xray.DataArrays
        Vertically integrated moisture flux convergence in mm/day
        (total, x- and y- components) and vertically integrated
        moisture fluxes.

    """

    # Convert from (kg/m^2)/s to mm/day
    SCALE = 60 * 60 * 24

    uq_int = dat.int_pres(uq, plev, pdim=pdim, pmin=pmin, pmax=pmax)
    vq_int = dat.int_pres(vq, plev, pdim=pdim, pmin=pmin, pmax=pmax)

    mfc, mfc_x, mfc_y = dat.divergence_spherical_2d(uq_int, vq_int, lat, lon,
                                                    return_comp=True)

    # Convert from divergence to convergence, and to mm/day
    mfc, mfc_x, mfc_y = -SCALE * mfc, -SCALE * mfc_x, -SCALE * mfc_y

    if isinstance(mfc, xray.DataArray):
        mfc.name = 'Vertically integrated moisture flux convergence'
        mfc.attrs['units'] = 'mm/day'

    if return_comp:
        return mfc, mfc_x, mfc_y, uq_int, vq_int
    else:
        return mfc


# ----------------------------------------------------------------------
def streamfunction(v, lat=None, pres=None, pdim=-3, scale=1e-9, topdown=True):
    """Return the Eulerian mass streamfunction.

    Parameters
    ----------
    v : ndarray or xray.DataArray
        Meridional wind speed in m/s.
    lat : ndarray, optional
        Array of latitude in degrees.  If omitted, lat is extracted
        from xray.DataArray input v.
    pres : ndarray, optional
        Pressures in Pa.  Can be a vector for pressure-level data, or
        a grid of pressures of the same shape as v.  If omitted, pres
        is extracted from xray.DataArray input v.
    pdim : {-3, -2}, optional
        Dimension of v corresponding to vertical levels.  Can be either
        the second-last or third-last dimension.
    scale : float, optional
        Scale factor for output, e.g. 1e-9 to output streamfunction in
        10^9 kg/s.
    topdown : bool, optional
        If True, integrate from the top of atmosphere down, assuming
        that vertical levels are indexed from the surface up (i.e.
        level 0 is the surface). Otherwise integrate from bottom up.

    Returns
    -------
    psi : ndarray or xray.DataArray
        Eulerian mass stream function in units of 1/scale kg/s.
    """

    R = constants.radius_earth.values
    g = constants.g.values

    if isinstance(v, xray.DataArray):
        i_DataArray = True
        coords, attrs, name = xr.meta(v)
        if lat is None:
            lat = dat.get_coord(v, 'lat')
        if pres is None:
            pres = dat.get_coord(v, 'plev')
            pname = dat.get_coord(v, 'plev', 'name')
            pres = dat.pres_convert(pres, v[pname].units, 'Pa')
        v = v.values.copy()
    else:
        i_DataArray = False
        if lat is None or pres is None:
            raise ValueError('Inputs lat and pres must be provided when '
                'v is an ndarray.')

    # Standardize the shape of v
    if pdim == -2:
        v = np.expand_dims(v, axis=-1)
        pdim = -3
    elif pdim != -3:
        raise ValueError('Invalid pdim %d.  Must be -2 or -3.' % pdim)

    dims = list(v.shape)
    nlevel = dims[pdim]
    dims[pdim] += 1
    nlat = dims[-2]

    # Tile the pressure levels to make a grid
    pmid = np.zeros(dims, dtype=float)
    pmid[...,:-1,:,:] = dat.biggify(pres, v, tile=True)

    # Cosine-weighted meridional velocity
    coslat = np.cos(np.radians(lat))
    vcos = np.zeros(v.shape, dtype=float)
    for j in range(nlat):
        vcos[...,j,:] = v[...,j,:] * coslat[j]

    # Compute streamfunction
    sfctn = np.zeros(dims, dtype=float)
    if topdown:
        # Integrate from top of atmosphere down
        for k in range(nlevel-1, -1, -1):
            dp = pmid[...,k+1,:,:] - pmid[...,k,:,:]
            sfctn[...,k,:,:] = sfctn[...,k+1,:,:] + vcos[...,k,:,:] * dp
    else:
        # Integrate from the surface up
        for k in range(1, nlevel):
            dp = pmid[...,k-1,:,:] - pmid[...,k,:,:]
            sfctn[...,k,:,:] = sfctn[...,k-1,:,:] + vcos[...,k,:,:] * dp

    # Scale the output and remove the added dimension(s)
    sfctn *= scale * 2 * np.pi * R / g
    psi = sfctn[...,:-1,:,:]
    if psi.ndim > v.ndim:
        psi = psi[...,0]

    if i_DataArray:
        psi = xray.DataArray(psi, name='Eulerian mass streamfunction',
            coords=coords)
        psi.attrs['units'] = '%.1e kg/s' % (1/scale)

    return psi
# ----------------------------------------------------------------------
# Dry static energy, moist static energy
# Vorticity
