import numpy as np
import xray

mp = 'Marshall and Plumb'

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
