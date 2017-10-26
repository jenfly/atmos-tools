"""Compute dynamic and thermodynamic atmospheric variables."""

from __future__ import division
import numpy as np
import xarray as xray
import atmos.utils as utils
import atmos.data as dat
import atmos.xrhelper as xrh
from atmos.constants import const as constants
from atmos.data import get_coord

# ----------------------------------------------------------------------
def coriolis(lat, degrees=True):
    """Return the Coriolis parameter 2*Omega*sin(lat).

    Input latitude array is assumed to be in degrees unless
    degrees=False is specified in inputs."""

    Omega = constants.Omega.values
    if degrees:
        lat_rad = np.radians(lat)
    else:
        lat_rad = lat

    return 2 * Omega * np.sin(lat_rad)


# ----------------------------------------------------------------------
def divergence_spherical_2d(Fx, Fy, lat=None, lon=None):
    """Return the 2-D spherical divergence.

    Parameters
    ----------
    Fx, Fy : ndarrays or xray.DataArrays
        Longitude, latitude components of a vector function in
        spherical coordinates.  Latitude and longitude should be the
        second-last and last dimensions, respectively, of Fx and Fy.
        Maximum size 5-D.
    lat, lon : ndarrays, optional
        Longitude and latitude in degrees.  If these are omitted, then
        Fx and Fy must be xray.DataArrays with latitude and longitude
        in degrees within the coordinates.

    Returns
    -------
    d, d1, d2 : ndarrays or xray.DataArrays
        d1 = dFx/dx, d2 = dFy/dy, and d = d1 + d2.

    Reference
    ---------
    Atmospheric and Oceanic Fluid Dynamics: Fundamentals and
    Large-Scale Circulation, by Geoffrey K. Vallis, Cambridge
    University Press, 2006 -- Equation 2.30.
    """

    nmax = 5
    ndim = Fx.ndim
    if ndim > nmax:
        raise ValueError('Input data has too many dimensions. Max 5-D.')

    if isinstance(Fx, xray.DataArray):
        i_DataArray = True
        name, attrs, coords, _ = xrh.meta(Fx)
        if lat is None:
            lat = get_coord(Fx, 'lat')
        if lon is None:
            lon = get_coord(Fx, 'lon')
    else:
        i_DataArray = False
        if lat is None or lon is None:
            raise ValueError('Lat/lon inputs must be provided when input '
                'data is an ndarray.')

    R = constants.radius_earth.values
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    # Add singleton dimensions for looping, if necessary
    for i in range(ndim, nmax):
        Fx = np.expand_dims(Fx, axis=0)
        Fy = np.expand_dims(Fy, axis=0)

    dims = Fx.shape
    nlon = dims[-1]
    nlat = dims[-2]

    d1 = np.zeros(dims, dtype=float)
    d2 = np.zeros(dims, dtype=float)

    for i in range(nlat):
        dx = np.gradient(lon_rad)
        coslat = np.cos(lat_rad[i])
        for k1 in range(dims[0]):
            for k2 in range(dims[1]):
                for k3 in range(dims[2]):
                    sub = Fx[k1,k2,k3,i,:]
                    d1[k1,k2,k3,i,:] = np.gradient(sub, dx) / (R*coslat)
    for j in range(nlon):
        dy = np.gradient(lat_rad)
        coslat = np.cos(lat_rad)
        # Set to NaN at poles to keep from blowing up
        coslat[abs(lat) > 89] = np.nan
        for k1 in range(dims[0]):
            for k2 in range(dims[1]):
                for k3 in range(dims[2]):
                    sub = Fy[k1,k2,k3,:,j] * coslat
                    d2[k1,k2,k3,:,j] = np.gradient(sub, dy) / (R*coslat)

    # Collapse any additional dimensions that were added
    for i in range(ndim, d1.ndim):
        d1, d2 = d1[0], d2[0]

    d = d1 + d2

    if i_DataArray:
        d = xray.DataArray(d, coords=coords)
        d1 = xray.DataArray(d1, coords=coords)
        d2 = xray.DataArray(d2, coords=coords)

    return d, d1, d2


# ----------------------------------------------------------------------
def vorticity(u, v, lat=None, lon=None):
    """Return the relative and absolute vorticity (vertical component).

    Parameters
    ----------
    u, v : ndarrays or xray.DataArrays
        Zonal and meridional winds in m/s. Latitude and longitude
        should be the second-last and last dimensions, respectively,
        of u and v.
    lat, lon : ndarrays, optional
        Latitudes and longitudes in degrees.  If omitted, then u and
        v must be xray.DataArrays and lat, lon are extracted from
        the metadata.

    Returns
    -------
    rel_vort, abs_vort : ndarrays or xray.DataArrays
        Vertical component of relative vorticity dv/dx - du/dy
        and absolute vorticity f + dv/dx - du/dy, calculated in
        spherical coordinates.
    f : ndarray
        Array of Coriolis parameters corresponding to latitude
        grid.

    Reference
    ---------
    Atmospheric and Oceanic Fluid Dynamics: Fundamentals and
    Large-Scale Circulation, by Geoffrey K. Vallis, Cambridge
    University Press, 2006 -- Equation 2.33.
    """

    # Relative vorticity
    _, dvdx, dudy = divergence_spherical_2d(v, u, lat, lon)
    rel_vort = dvdx - dudy

    # Coriolis parameter
    if lat is None:
        if isinstance(u, xray.DataArray):
            lat = get_coord(u, 'lat')
        else:
            raise ValueError('Lat/lon inputs must be provided when input '
                'data is an ndarray.')
    f = coriolis(lat)

    # Absolute vorticity
    abs_vort = rel_vort + dat.biggify(f, rel_vort)

    if isinstance(u, xray.DataArray):
        abs_vort.name = 'abs_vort'
        abs_vort.attrs['long_name'] = 'Absolute vorticity'
        rel_vort.name = 'rel_vort'
        rel_vort.attrs['long_name'] = 'Relative vorticity'

    return rel_vort, abs_vort, f


# ----------------------------------------------------------------------
def rossby_num(u, v, lat=None, lon=None):
    """Return the local Rossby number.

    Parameters
    ----------
    u, v : ndarrays or xray.DataArrays
        Zonal and meridional winds in m/s. Latitude and longitude
        should be the second-last and last dimensions, respectively,
        of u and v.
    lat, lon : ndarrays, optional
        Latitudes and longitudes in degrees.  If omitted, then u and
        v must be xray.DataArrays and lat, lon are extracted from
        the metadata.

    Returns
    -------
    Ro : ndarray or xray.DataArray
        Local Rossby number (Ro = -relative_vorticity / f), calculated
        in spherical coordinates.

    Reference
    ---------
    Schneider, T. (2006). The General Circulation of the Atmosphere.
    Annual Review of Earth and Planetary Sciences, 34(1), 655-688.
    doi:10.1146/annurev.earth.34.031405.125144
    """

    rel_vort, _, f = vorticity(u, v, lat, lon)
    Ro = - rel_vort / dat.biggify(f, rel_vort)

    if isinstance(u, xray.DataArray):
        Ro.name = 'Ro'
        Ro.attrs['long_name'] = 'Local Rossby number'
    return Ro


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
        theta.name = 'theta'
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
        theta_e.name = 'theta_e'
        theta_e.attrs['long_name'] = 'Equivalent Potential Temperature'
        theta_e.attrs['units'] = 'K'

    return theta_e


# ----------------------------------------------------------------------
def dry_static_energy(T, z):
    return constants.Cp * T + constants.g * z


# ----------------------------------------------------------------------
def moist_static_energy(T, z, q):
    return dry_static_energy(T, z) + constants.Lv * q


# ----------------------------------------------------------------------
def moisture_flux_conv(uq, vq, lat=None, lon=None, plev=None, pdim=-3,
                       pmin=0, pmax=1e6, return_comp=False, already_int=False):
    """Return the vertically integrated moisture flux convergence.

    Parameters
    ----------
    uq : ndarray or xray.DataArray
        Zonal moisture flux, with latitude as the second-last dimension,
        longitude as the last dimension.
        i.e. zonal wind (m/s) * specific humidity (kg/kg)
        If already_int is True, then uq is already vertically integrated.
        Otherwise, uq is on vertical pressure levels.
    vq : ndarray or xray.DataArray
        Meridional moisture flux, with latitude as the second-last dimension,
        longitude as the last dimension.
        i.e. meridional wind (m/s) * specific humidity (kg/kg)
        If already_int is True, then vq is already vertically integrated.
        Otherwise, vq is on vertical pressure levels.
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
    return_comp : bool, optional
        If True, return additional components, otherwise just total
        moisture flux convergence.
    already_int : bool, optional
        If True, then uq and vq inputs have already been vertically
        integrated.  Otherwise, the vertical integration is
        calculated here.

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

    if already_int:
        uq_int, vq_int = uq, vq
    else:
        uq_int = dat.int_pres(uq, plev, pdim=pdim, pmin=pmin, pmax=pmax)
        vq_int = dat.int_pres(vq, plev, pdim=pdim, pmin=pmin, pmax=pmax)

    mfc, mfc_x, mfc_y = divergence_spherical_2d(uq_int, vq_int, lat, lon)

    # Convert from divergence to convergence, and to mm/day
    mfc = -dat.precip_convert(mfc, 'kg/m2/s', 'mm/day')
    mfc_x = -dat.precip_convert(mfc_x, 'kg/m2/s', 'mm/day')
    mfc_y = -dat.precip_convert(mfc_y, 'kg/m2/s', 'mm/day')

    if isinstance(mfc, xray.DataArray):
        mfc.name = 'Vertically integrated moisture flux convergence'
        mfc.attrs['units'] = 'mm/day'

    if return_comp:
        return mfc, mfc_x, mfc_y, uq_int, vq_int
    else:
        return mfc


# ----------------------------------------------------------------------
def streamfunction(v, lat=None, pres=None, pdim=None, scale=1e-9,
                   sector_scale=None):
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
        the second-last or third-last dimension.  If omitted, pdim is
        extracted from xray.DataArray v.
    scale : float, optional
        Scale factor for output, e.g. 1e-9 to output streamfunction in
        10^9 kg/s.
    sector_scale : float, optional
        Scaling factor for the streamfunction over a sector lon1-lon2
        (i.e. should be set to (lon2 - lon1)/360.0 or None if full longitude
        range is going to be included).
    Returns
    -------
    psi : ndarray or xray.DataArray
        Eulerian mass stream function in units of 1/scale kg/s.
    """

    R = constants.radius_earth.values
    g = constants.g.values

    if isinstance(v, xray.DataArray):
        i_DataArray = True
        name, attrs, coords, _ = xrh.meta(v)
        if lat is None:
            lat = dat.get_coord(v, 'lat')
        if pres is None:
            pres = dat.get_coord(v, 'plev')
            pname = dat.get_coord(v, 'plev', 'name')
            pres = dat.pres_convert(pres, v[pname].units, 'Pa')
        if pdim is None:
            pdim = dat.get_coord(v, 'plev', 'dim') - v.ndim
        v = v.values.copy()
    else:
        i_DataArray = False
        if lat is None or pres is None or pdim is None:
            raise ValueError('Inputs lat, pres and pdim must be provided when '
                'v is an ndarray.')

    # Standardize the shape of v
    pdim_in = pdim
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

    # Compute streamfunction, integrating from top of atmosphere down
    sfctn = np.zeros(dims, dtype=float)
    for k in range(nlevel-1, -1, -1):
        dp = pmid[...,k+1,:,:] - pmid[...,k,:,:]
        sfctn[...,k,:,:] = sfctn[...,k+1,:,:] + vcos[...,k,:,:] * dp

    # Scale the output and remove the added dimension(s)
    sfctn *= scale * 2 * np.pi * R / g
    psi = sfctn[...,:-1,:,:]
    if pdim_in == -2:
        psi = psi[...,0]
    if sector_scale is not None:
        psi = psi * sector_scale

    if i_DataArray:
        psi = xray.DataArray(psi, name='Eulerian mass streamfunction',
            coords=coords)
        psi.attrs['units'] = '%.1e kg/s' % (1/scale)

    return psi
# ----------------------------------------------------------------------
# Dry static energy, moist static energy
# Vorticity
