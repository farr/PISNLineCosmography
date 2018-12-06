from pylab import *

from astropy.cosmology import Planck15
import astropy.units as u

true_params = {
    'R0': 100.0, # Gpc^-3 yr^-1
    'alpha': 0.75,
    'beta': 0.0,
    'gamma': 3.0,
    'MMin': 5.0, # MSun
    'MMax': 40.0, # MSun
    'sigma_low': 0.1,
    'sigma_high': 0.1,
    'H0': Planck15.H0.to(u.km/u.s/u.Mpc).value, # km/s/Mpc
    'Om': Planck15.Om0,
    'w': -1.0,
}

uncert = {
    'threshold_snr': 8,
    'Theta': 0.15,
    'mc': 0.017,
    'eta': 0.017
}

ifo = {
    'obstime': 5,
    'duty_cycle': 0.5
}
