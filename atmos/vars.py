import numpy as np
import xray

hart = 'Global Physical Climatology, by Dennis Hartmann, 1994'

constants = xray.Dataset()

constants['g'] = xray.DataArray(
    9.80665,
    attrs={'name' : "Earth's standard gravity",
           'units' : 'm s^-2',
           'ref' : hart
          })

constants['radius_earth'] = xray.DataArray(
    6.37e6,
    attrs={'name' : "Earth's mean radius",
           'units' : 'm',
           'ref' : hart
          })


constants['mass_earth'] = xray.DataArray(
    5.983e24,
    attrs={'name' : 'Mass of Earth',
           'units' : 'kg',
           'ref' : hart
          })


constants['mass_atm'] = xray.DataArray(
    5.3e18,
    attrs={'name' : "Mass of Earth's Atmosphere",
           'units' : 'kg',
           'ref' : hart
          })

constants['Omega'] = xray.DataArray(
    7.292e-5,
    attrs={'name' : "Earth's mean rotation rate",
           'units' : 's^-1',
           'ref' : hart
          })

constants['S0'] = xray.DataArray(
    1367,
    attrs={'name' : "Earth's solar constant",
           'units' : 'W m^-2',
           'ref' : hart
          })


constants['R'] = xray.DataArray(
    8.3143,
    attrs={'name' : 'Universal gas constant',
           'units' : 'J K^-1 mol^-1',
           'ref' : hart
          })


constants['R_air'] = xray.DataArray(
    287,
    attrs={'name' : 'Gas constant for dry air',
           'units' : 'J K^-1 kg^-1',
           'ref' : hart
          })

constants['m_air'] = xray.DataArray(
    28.97,
    attrs={'name' : 'Mean molar mass of dry air',
           'units' : 'g mol^-1',
           'ref' : hart
          })

constants['density_air'] = xray.DataArray(
    1.293,
    attrs={'name' : 'Density of dry air at 0C and 101325 Pa',
           'units' : 'kg m^-3',
           'ref' : hart
          })


constants['Cp'] = xray.DataArray(
    1004,
    attrs={'name' : 'Dry air specific heat at constant pressure',
           'units' : 'J kg^-1 K^-1',
           'ref' : hart
          })

constants['Cv'] = xray.DataArray(
    717,
    attrs={'name' : 'Dry air specific heat at constant volume',
           'units' : 'J kg^-1 K^-1',
           'ref' : hart
          })

constants['m_water'] = xray.DataArray(
    18.016,
    attrs={'name' : 'Molar mass of water',
           'units' : 'g mol^-1',
           'ref' : hart
          })

constants['R_wv'] = xray.DataArray(
    461,
    attrs={'name' : 'Gas constant for water vapor',
           'units' : 'J K^-1 kg^-1',
           'ref' : hart
          })

constants['Cp_wv'] = xray.DataArray(
    1952,
    attrs={'name' : 'Specific heat of water vapor at constant pressure',
           'units' : 'J K^-1 kg^-1',
           'ref' : hart
          })


constants['Cv_wv'] = xray.DataArray(
    1463,
    attrs={'name' : 'Specific heat of water vapor at constant volume',
           'units' : 'J K^-1 kg^-1',
           'ref' : hart
          })

constants['C_wl'] = xray.DataArray(
    4218,
    attrs={'name' : 'Specific heat of liquid water at 0C',
           'units' : 'J K^-1 kg^-1',
           'ref' : hart
          })

constants['C_wi'] = xray.DataArray(
    2106,
    attrs={'name' : 'Specific heat of water ice at 0C',
           'units' : 'J K^-1 kg^-1',
           'ref' : hart
          })

constants['Lv'] = xray.DataArray(
    2.5e6,
    attrs={'name' : 'Latent heat of vaporization of water at 0C',
           'units' : 'J kg^-1',
           'ref' : hart
          })


constants['Lv_100'] = xray.DataArray(
    2.25e6,
    attrs={'name' : 'Latent heat of vaporization of water at 100C',
           'units' : 'J kg^-1',
           'ref' : hart
          })


constants['L_fusion'] = xray.DataArray(
    3.34e5,
    attrs={'name' : 'Latent heat of fusion of water at 0C',
           'units' : 'J kg^-1',
           'ref' : hart
          })
