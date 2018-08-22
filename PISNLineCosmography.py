from pylab import *

import astropy.cosmology as cosmo
from astropy.cosmology import Planck15
import astropy.units as u
import pymc3 as pm
import scipy.interpolate as si
import theano
import theano.tensor as tt

def Ez(z):
    Om = Planck15.Om0
    opz = 1.0 + z
    return tt.sqrt(Om*opz*opz*opz + (1.0 - Om))

def dzddl(dl, z, dH):
    dc = dl / (1.0 + z)

    return 1.0/(dc + (1.0 + z)*dH/Ez(z))

def interp1d(x, xs, ys):
    i = tt.extra_ops.searchsorted(xs, x)

    r = (x - xs[i-1])/(xs[i] - xs[i-1])

    return r*ys[i] + (1.0-r)*ys[i-1]

def dNdm1obsdm2obsddldt(m1obs, m2obs, dlobs, zobs, R0, MMin, MMax, alpha, beta, gamma, dH):
    m1 = m1obs / (1.0 + zobs)
    m2 = m2obs / (1.0 + zobs)
    dc = dlobs / (1.0 + zobs)

    pm1m2 = m1**(-alpha) * m2**(beta) * (1.0 - alpha) * (1.0 + beta) / ((MMax**(1.0-alpha) - MMin**(1.0-alpha)) * (m1**(1.0+beta) - MMin**(1.0+beta)))

    dNdm1dm2dVdt = R0*pm1m2*(1+zobs)**(gamma-1)

    dVdz = 4.0*pi*dc*dc*dH/Ez(zobs)

    dzddl_ = dzddl(dlobs, zobs, dH)

    dmdmobs = 1.0/(1.0+zobs)

    dN = dNdm1dm2dVdt*dmdmobs*dmdmobs*dVdz*dzddl_

    return tt.switch((m1 < MMax) & (m2 > MMin), dN, 0.0)

def make_model(m1obs, m2obs, dlobs, m1sel, m2sel, dlsel, Tobs, Vgen, Ngen):
    dlmax = 5.0*max(np.max(dlobs), np.max(dlsel))
    zmax = cosmo.z_at_value(Planck15.luminosity_distance, dlmax*u.Gpc)

    zinterp = linspace(0, zmax, 500)
    dluinterp = Planck15.luminosity_distance(zinterp)/Planck15.hubble_distance

    zs = si.interp1d(dluinterp*Planck15.hubble_distance.to(u.Gpc).value, zinterp)(dlobs)

    zinterp = theano.shared(zinterp)
    dluinterp = theano.shared(dluinterp)

    m = pm.Model()

    H0_init = Planck15.H0.to(u.km/u.s/u.Mpc).value
    MMax_init = 1.2*np.max(np.min(m1obs/(1+zs), axis=1))

    print('MMax_init = {:.1f}'.format(MMax_init))

    with m:
        R0 = pm.Lognormal('R0', mu=log(100), sd=1, testval=100.0)
        H0 = pm.Bound(pm.Lognormal, lower=50, upper=100)('H0', mu=log(70), sd=15.0/70.0, testval=H0_init)

        MMin = pm.Bound(pm.Normal, lower=3, upper=10)('MMin', mu=5, sd=3, testval=5.0)
        MMax = pm.Bound(pm.Normal, lower=30, upper=100)('MMax', mu=40, sd=10, testval=MMax_init)

        alpha = pm.Normal('alpha', mu=1, sd=2, testval=1.1)
        beta = pm.Normal('beta', mu=0, sd=2, testval=0.0)
        gamma = pm.Normal('gamma', mu=3, sd=2, testval=3.0)

        dH = pm.Deterministic('dH', 4.42563416002 * (67.74/H0))

        dlinterp = dH*dluinterp

        zobs = interp1d(dlobs, dlinterp, zinterp)
        zsel = interp1d(dlsel, dlinterp, zinterp)

        Nex = pm.Deterministic('Nex', Tobs*Vgen/Ngen*tt.sum(dNdm1obsdm2obsddldt(m1sel, m2sel, dlsel, zsel, R0, MMin, MMax, alpha, beta, gamma, dH)))

        pm.Potential('norm', -Nex)

        pm.Potential('likelihood', tt.sum(tt.log(tt.mean(dNdm1obsdm2obsddldt(m1obs, m2obs, dlobs, zobs, R0, MMin, MMax, alpha, beta, gamma, dH), axis=1))))

    return m
