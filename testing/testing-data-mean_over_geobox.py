"""Testing mask_oceans and mean_over_geobox"""

import numpy as np
import matplotlib.pyplot as plt
import xray

import atmos.plots as ap
import atmos.data as dat

# ----------------------------------------------------------------------
# Read data
url = ('http://goldsmr3.sci.gsfc.nasa.gov/opendap/MERRA_MONTHLY/'
    'MAIMCPASM.5.2.0/1979/MERRA100.prod.assim.instM_3d_asm_Cp.197901.hdf')

ds = xray.open_dataset(url)
T = ds['T']
ps = ds['PS']
q = ds['QV']
plev = dat.get_plev(T, units='Pa')
lat, lon = dat.get_lat(ps), dat.get_lon(ps)


# ----------------------------------------------------------------------
# Masking ocean

T_land = dat.mask_oceans(T, inlands=True)

t, k = 0, 10
plt.figure()
ap.pcolor_latlon(T_land[t,k], cmap='jet')

# ----------------------------------------------------------------------
# Averaging over box - constant array

lon1, lon2 = 20, 80
lat1, lat2 = 65, 85

data = 2.5 * np.ones(ps.shape)

avg1 = dat.mean_over_geobox(data, lat1, lat2, lon1, lon2, lat=lat, lon=lon,
    area_wtd=False)
avg2 = dat.mean_over_geobox(data, lat1, lat2, lon1, lon2, lat=lat, lon=lon,
    area_wtd=True)
avg3 = dat.mean_over_geobox(data, lat1, lat2, lon1, lon2, lat=lat, lon=lon,
    area_wtd=True, land_only=True)

print(avg1)
print(avg2)
print(avg3)

# ----------------------------------------------------------------------
# Averaging over box
def testing(data, lat1, lat2, lon1, lon2, t, k):

    latname = dat.get_lat(data, return_name=True)
    lonname = dat.get_lon(data, return_name=True)

    data_sub = dat.subset(data, latname, lat1, lat2, lonname, lon1, lon2)

    plt.figure()
    ap.pcolor_latlon(data_sub[t,k], axlims=(lat1,lat2,lon1,lon2), cmap='jet')

    avg0 = data_sub.mean(axis=-1).mean(axis=-1)
    avg1 = dat.mean_over_geobox(data, lat1, lat2, lon1, lon2, area_wtd=False)
    avg2 = dat.mean_over_geobox(data, lat1, lat2, lon1, lon2, area_wtd=True)
    avg3 = dat.mean_over_geobox(data, lat1, lat2, lon1, lon2, area_wtd=True,
                            land_only=True)

    print(avg0[t, k].values)
    print(avg1[t, k].values)
    print(avg2[t, k].values)
    print(avg3[t, k].values)
# ----------------------------------------------------------------------

t, k = 0, 6
lat1, lat2 = -50, -15
lon1, lon2 = 15, 30
testing(T, lat1, lat2, lon1, lon2, t, k)

lat1, lat2 = 10, 30
lon1, lon2 = 80, 95
testing(T, lat1, lat2, lon1, lon2, t, k)

lat1, lat2 = 10, 50
lon1, lon2 = 60, 100
testing(T, lat1, lat2, lon1, lon2, t, k)
