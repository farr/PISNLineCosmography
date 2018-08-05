from pylab import *

import astropy.cosmology as cosmo
from astropy.cosmology import Planck15
import astropy.units as u
import pymc3 as pm
import theano
import theano.tensor as tt

def interp1d(xs, xs_interp, ys_interp):
    inds = tt.extra_ops.searchsorted(xs_interp, xs)

    x0 = xs_interp[inds-1]
    x1 = xs_interp[inds]

    r = (xs-x0)/(x1-x0)

    return r*ys_interp[inds] + (1.0-r)*ys_interp[inds-1]

def Ez(z, Om):
    opz = 1.0 + z
    opz2 = opz*opz
    opz3 = opz*opz2

    return tt.sqrt(opz3*Om + (1.0 - Om))

def dzddL(dl, z, dH, Om):
    dc = dl/(1.0+z)
    return 1.0/(dc + (1.0+z)*dH/Ez(z, Om))

def dNdm1obsddl(m1_obs, dl_obs, z_obs, R0, alpha, MMin, MMax, gamma, dH, Om):
    m1 = m1_obs / (1.0 + z_obs)

    dc_obs = dl_obs / (1.0 + z_obs)

    dVdz = 4.0*pi*dH*dc_obs*dc_obs/Ez(z_obs, Om)
    dzddL_ = dzddL(dl_obs, z_obs, dH, Om)

    dNdm1obs = R0*(1.0-alpha)*m1**(-alpha)/(MMax**(1.0-alpha) - MMin**(1.0-alpha))

    dN = dNdm1obs*dVdz*dzddL_*(1.0+z_obs)**(gamma-1.0)

    return tt.switch((m1 > MMin) & (m1 < MMax), dN, 0.0)

def make_model(m1_obs, dl_obs, m1_sel, dl_sel, Vgen, Ngen):
    m1_obs = np.atleast_1d(m1_obs)
    dl_obs = np.atleast_1d(dl_obs)
    m1_sel = np.atleast_1d(m1_sel)
    dl_sel = np.atleast_1d(dl_sel)

    nsel = m1_sel.shape[0]

    dmax = max(np.max(dl_obs), np.max(dl_sel))
    zmax = cosmo.z_at_value(Planck15.luminosity_distance, dmax*u.Gpc)
    zmax = zmax * 2

    dmax_zmax = Planck15.luminosity_distance(zmax).to(u.Gpc).value

    dH_factor_min = dmax / dmax_zmax
    H_factor_max = 1.0/dH_factor_min

    H_max = Planck15.H0.to(u.km/u.s/u.Mpc).value * H_factor_max * 0.95

    z_interp = logspace(log10(1), log10(zmax+1), 200)-1
    du_interp = Planck15.luminosity_distance(z_interp).to(u.Gpc).value / Planck15.hubble_distance.to(u.Gpc).value

    z_interp = theano.shared(z_interp)
    du_interp = theano.shared(du_interp)

    model = pm.Model()

    with model:
        R0 = pm.Lognormal('R0', mu=log(100), sd=1)
        alpha = pm.Bound(pm.Normal, lower=-3, upper=3)('alpha', mu=1, sd=2)
        MMin = pm.Bound(pm.Lognormal, lower=1, upper=10)('MMin', mu=log(5), sd=1)
        MMax = pm.Bound(pm.Lognormal, lower=30, upper=60)('MMax', mu=log(40), sd=1)
        gamma = pm.Bound(pm.Normal, lower=-3, upper=10)('gamma', mu=3, sd=3)
        H0 = pm.Bound(pm.Lognormal, lower=0, upper=H_max)('H0', mu=log(70), sd=1)

        dH = pm.Deterministic('dH', 4.42563416002 * (67.74/H0))

        z_obs = interp1d(dl_obs, du_interp*dH, z_interp)
        z_sel = interp1d(dl_sel, du_interp*dH, z_interp)

        pm.Potential('likelihood', tt.sum(tt.log(tt.mean(dNdm1obsddl(m1_obs, dl_obs, z_obs, R0, alpha, MMin, MMax, gamma, dH, Planck15.Om0), axis=1))))

        pm.Potential('Poisson-norm', -Vgen/Ngen*tt.sum(dNdm1obsddl(m1_sel, dl_sel, z_sel, R0, alpha, MMin, MMax, gamma, dH, Planck15.Om0)))

    return model
