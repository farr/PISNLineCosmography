from pylab import *

import astropy.cosmology as cosmo
from astropy.cosmology import Planck15
import astropy.units as u
import pymc3 as pm
import theano
import theano.tensor as tt

def Ez(z, Om):
    zp1 = 1.0 + z
    Ol = 1.0 - Om

    return tt.sqrt(Ol + zp1*zp1*zp1*Om)

def dNdm1obsdqddl(m1obs, dl, z, R0, alpha, MMin, MMax, gamma, dH, Om):
    m1 = m1obs/(1.0 + z)

    dc = dl/(1.0+z)

    dVdz = 4.0*pi*dH*dc*dc/Ez(z, Om)
    dzddl = 1.0/(dc + dH*(1.0+z)/Ez(z, Om))

    mnorm = (1.0-alpha)/(MMax**(1.0-alpha) - MMin**(1.0 - alpha))

    dN = R0*mnorm*m1**(-alpha)/(1.0+z)*(1.0+z)**(gamma-1.0)*dVdz*dzddl

    return tt.switch((m1 > MMin) & (m1 < MMax), dN, 0.0)

def interp1d(x, xs, ys):
    i = tt.extra_ops.searchsorted(xs, x)

    x1 = xs[i-1]
    x2 = xs[i]

    y1 = ys[i-1]
    y2 = ys[i]

    r = (x - x1) / (x2 - x1)

    return r*y2 + (1.0-r)*y1

def ddcudz(z, Om):
    return 1.0/Ez(z, Om)

def cumtrapz(ys, xs):
    dxs = tt.extra_ops.diff(xs)
    terms = 0.5*dxs*(ys[1:] + ys[:-1])

    cterms = tt.extra_ops.cumsum(terms)

    return tt.concatenate([tt.zeros((1,)), cterms])

def dls_at_zs(zs, dH, Om):
    dddz = ddcudz(zs, Om)

    dcs = dH*cumtrapz(dddz, zs)

    dls = (1.0+zs)*dcs

    return dls

def make_model(m1obs, dlobs, m1obs_sel, dlobs_sel, ngen, Vgen):
    ndet = m1obs_sel.shape[0]

    # Right now, we don't fit Om, so just do the interpolant outside the model
    zmax = 2.0*cosmo.z_at_value(Planck15.luminosity_distance, np.max(dlobs)*u.Gpc)

    Om = Planck15.Om0

    zs = theano.shared(linspace(0, zmax, 200))
    dls = dls_at_zs(zs, 1.0, Om)

    # If H0 gets too large, we will be outside the valid domain of our dl vs z interpolation
    # So, we need to impose a limit so that we always stay within the relevant domain
    hmax = 4.42563416002*0.6774*dls.eval()[-1]/np.max(dlobs)*0.9 # 10% safety factor
    print('Setting hmax to {:g}'.format(hmax))

    model = pm.Model()
    with model:
        h = pm.Bound(pm.Lognormal, lower=0, upper=hmax)('h', mu=log(0.7), sd=1.0)
        H0 = pm.Deterministic('H0', 100.0*h)
        dH = pm.Deterministic('dH', 4.42563416002 * (67.74/H0))

        r = pm.Lognormal('r', mu=log(1.0), sd=1.0)
        R0 = pm.Deterministic('R0', 100.0*r)

        MMin = pm.Bound(pm.Lognormal, lower=1.0, upper=10.0)('MMin', mu=log(5.0), sd=1)
        MMax = pm.Bound(pm.Lognormal, lower=30.0, upper=60.0)('MMax', mu=log(40.0), sd=1)

        alpha = pm.Bound(pm.Normal, lower=-3, upper=3)('alpha', mu=0.0, sd=1.5)
        gamma = pm.Bound(pm.Normal, lower=-5, upper=5)('gamma', mu=0.0, sd=3)

        zobs = interp1d(dlobs/dH, dls, zs)

        # Likelihood terms for each observation
        dNs = dNdm1obsdqddl(m1obs, dlobs, zobs, R0, alpha, MMin, MMax, gamma, dH, Om)
        pm.Potential('likelihood', tt.sum(tt.log(tt.mean(dNs, axis=1))))

        # Now the expected number of detections
        zsel = interp1d(dlobs_sel/dH, dls, zs)
        fdet = dNdm1obsdqddl(m1obs_sel, dlobs_sel, zsel, R0, alpha, MMin, MMax, gamma, dH, Om)
        Ndet = Vgen/ngen*tt.sum(fdet)
        pm.Potential('poisson-norm', -Ndet)

    return model
