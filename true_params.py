from pylab import *

from astropy.cosmology import Planck15
import astropy.units as u

true_params = {
    'R0': 60.0, # Gpc^-3 yr^-1
    'R0_30': 43.88, # R_{0,30} / 30^2 = R_0 p\left( m_1 = 30, m_2 = 30 \right) (to avoid having to normalize the PDF every time)
    'alpha': 0.75,
    'beta': 0.0,
    'gamma': 3.0,
    'MMin': 5.0, # MSun
    'MMax': 50.0, # MSun
    'H0': Planck15.H0.to(u.km/u.s/u.Mpc).value, # km/s/Mpc
    'Om': Planck15.Om0,
    'w': -1.0,
    'w_a': 0.0,
    'sigma_min': 0.1,
    'sigma_max': 0.1
}

uncert = {
    'threshold_snr': 8,
    'Theta': 0.05,
    'mc': 0.03,
    'eta': 0.005
}

ifo = {
    'obstime': 5,
    'duty_cycle': 0.5
}
