import numpy as np
import xray

mp = ('Atmosphere, Ocean, and Climate Dynamics: An Introductory Text,'
    'by John Marshall and R. Alan Plumb, 2008')

constants = xray.Dataset()

constants['g'] = xray.DataArray(
    9.81,
    attrs={'name' : "Earth's surface gravity",
           'units' : 'm s^-2',
           'ref' : mp
          })

constants['R'] = xray.DataArray(
    6.37e6,
    attrs={'name' : "Earth's mean radius",
           'units' : 'm',
           'ref' : mp
          })


constants['STP_T'] = xray.DataArray(
    273,
    attrs={'name' : 'Temperature at STP,
           'units' : 'K',
           'ref' : mp
          })


constants['STP_p'] = xray.DataArray(
    101325,
    attrs={'name' : 'Pressure at STP,
           'units' : 'Pa',
           'ref' : mp
          })

constants['Omega'] = xray.DataArray(
    7.27e-5,
    attrs={'name' : "Earth's rotation rate",
           'units' : 's^-1',
           'ref' : mp
          })

constants['Cp'] = xray.DataArray(
    1005,
    attrs={'name' : 'Dry air specific heat at constant pressure, at STP',
           'units' : 'J kg^-1 K^-1',
           'ref' : mp
          })

constants['Cv'] = xray.DataArray(
    718,
    attrs={'name' : 'Dry air specific heat at constant volume, at STP',
           'units' : 'J kg^-1 K^-1',
           'ref' : mp
          })
