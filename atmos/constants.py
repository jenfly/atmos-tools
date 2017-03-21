from __future__ import division
import numpy as np
import xarray as xray

hart = 'Global Physical Climatology, by Dennis Hartmann, 1994'

const = xray.Dataset()

const['g'] = xray.DataArray(
    9.80665,
    attrs={'name' : "Earth's standard gravity",
           'units' : 'm s^-2',
           'ref' : hart
          })

const['radius_earth'] = xray.DataArray(
    6.37e6,
    attrs={'name' : "Earth's mean radius",
           'units' : 'm',
           'ref' : hart
          })


const['mass_earth'] = xray.DataArray(
    5.983e24,
    attrs={'name' : 'Mass of Earth',
           'units' : 'kg',
           'ref' : hart
          })


const['mass_atm'] = xray.DataArray(
    5.3e18,
    attrs={'name' : "Mass of Earth's Atmosphere",
           'units' : 'kg',
           'ref' : hart
          })

const['Omega'] = xray.DataArray(
    7.292e-5,
    attrs={'name' : "Earth's mean rotation rate",
           'units' : 's^-1',
           'ref' : hart
          })

const['S0'] = xray.DataArray(
    1367.,
    attrs={'name' : "Earth's solar constant",
           'units' : 'W m^-2',
           'ref' : hart
          })


const['R_gas'] = xray.DataArray(
    8.3143,
    attrs={'name' : 'Universal gas constant',
           'units' : 'J K^-1 mol^-1',
           'ref' : hart
          })


const['R_air'] = xray.DataArray(
    287.,
    attrs={'name' : 'Gas constant for dry air',
           'units' : 'J K^-1 kg^-1',
           'ref' : hart
          })

const['mm_air'] = xray.DataArray(
    28.97,
    attrs={'name' : 'Mean molar mass of dry air',
           'units' : 'g mol^-1',
           'ref' : hart
          })

const['density_air'] = xray.DataArray(
    1.293,
    attrs={'name' : 'Density of dry air at 0C and 101325 Pa',
           'units' : 'kg m^-3',
           'ref' : hart
          })


const['Cp'] = xray.DataArray(
    1004.,
    attrs={'name' : 'Dry air specific heat at constant pressure',
           'units' : 'J kg^-1 K^-1',
           'ref' : hart
          })

const['Cv'] = xray.DataArray(
    717.,
    attrs={'name' : 'Dry air specific heat at constant volume',
           'units' : 'J kg^-1 K^-1',
           'ref' : hart
          })

const['mm_water'] = xray.DataArray(
    18.016,
    attrs={'name' : 'Molar mass of water',
           'units' : 'g mol^-1',
           'ref' : hart
          })

const['R_water_v'] = xray.DataArray(
    461.,
    attrs={'name' : 'Gas constant for water vapor',
           'units' : 'J K^-1 kg^-1',
           'ref' : hart
          })

const['Cp_water_v'] = xray.DataArray(
    1952.,
    attrs={'name' : 'Specific heat of water vapor at constant pressure',
           'units' : 'J K^-1 kg^-1',
           'ref' : hart
          })


const['Cv_water_v'] = xray.DataArray(
    1463.,
    attrs={'name' : 'Specific heat of water vapor at constant volume',
           'units' : 'J K^-1 kg^-1',
           'ref' : hart
          })

const['C_water_l'] = xray.DataArray(
    4218.,
    attrs={'name' : 'Specific heat of liquid water at 0C',
           'units' : 'J K^-1 kg^-1',
           'ref' : hart
          })

const['C_water_i'] = xray.DataArray(
    2106.,
    attrs={'name' : 'Specific heat of water ice at 0C',
           'units' : 'J K^-1 kg^-1',
           'ref' : hart
          })

const['Lv'] = xray.DataArray(
    2.5e6,
    attrs={'name' : 'Latent heat of vaporization/condensation of water at 0C',
           'units' : 'J kg^-1',
           'ref' : hart
          })

const['L_water_fusion'] = xray.DataArray(
    3.34e5,
    attrs={'name' : 'Latent heat of fusion of water at 0C',
           'units' : 'J kg^-1',
           'ref' : hart
          })
