import numpy as np
import xray

# My modules:
import atmos.data as dat
from atmos.utils import disptime

# ----------------------------------------------------------------------
# Opening multiple OpenDAP files into xray dataset

url_dir = ('http://goldsmr3.sci.gsfc.nasa.gov/opendap/MERRA/'
    'MAI3CPASM.5.2.0/1979/01/')
start = 'MERRA100.prod.assim.inst3_3d_asm_Cp.197901'
#end = '.hdf?U,V,XDim,YDim,Height,TIME'
end = '.hdf'

paths = ['%s%s%02d%s' % (url_dir, start, i, end) for i in range(1,3)]

concat_dim = 'TIME'

u = dat.load_concat(paths, 'U', concat_dim=concat_dim, verbose=True)
v = dat.load_concat(paths, 'V', concat_dim=concat_dim, verbose=True)

ds = u.to_dataset(name='u')
ds['v'] = v
ds['uu'] = u * u
ds['uv'] = u * v
ds = ds.mean(dim='TIME')

ds['u'].attrs = u.attrs
ds['v'].attrs = v.attrs
ds['uu'].attrs = {'long_name' : 'u * u', 'units' : 'm^2 s^-2'}
ds['uv'].attrs = {'long_name' : 'u * v', 'units' : 'm^2 s^-2'}

outfile = 'data/more/merra_uv_197901.nc'
ds.to_netcdf(outfile,mode='w')

# v = ds['V']
# T = ds['T']
# ps = ds['PS']
# q = ds['QV']
# hgt = ds['H']
# omega = ds['OMEGA']
