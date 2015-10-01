
import matplotlib.pyplot as plt
import xray
import atmos as atm

# ----------------------------------------------------------------------
# Read some data from OpenDAP url

url = ('http://goldsmr3.sci.gsfc.nasa.gov/opendap/MERRA_MONTHLY/'
    'MAIMCPASM.5.2.0/1979/MERRA100.prod.assim.instM_3d_asm_Cp.197901.hdf')

ds = xray.open_dataset(url)
ps = ds['PS'] / 100

plt.figure()
atm.pcolor_latlon(ps, cmap='jet')

lon1, lon2 = 0, 100
lat1, lat2 = -45, 45
ps_sub = atm.subset(ps, 'lon', lon1, lon2, 'lat', lat1, lat2)
plt.figure()
atm.pcolor_latlon(ps_sub, cmap='jet')
