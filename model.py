from pylab import *

import astropy.cosmology as cosmo
from astropy.cosmology import Planck15
import astropy.units as u
import pymc3 as pm
from scipy.interpolate import interp1d
import theano
import theano.tensor as tt
import theano.tensor.extra_ops as tte
import theano.tensor.slinalg as tts

def trapz(ys, xs):
    dx = tte.diff(xs)
    dI = 0.5*(ys[1:] + ys[:-1])*dx

    return tt.sum(dI)

def cumtrapz(ys, xs, initial=0):
    dx = tte.diff(xs)
    dI = 0.5*(ys[1:] + ys[:-1])*dx

    return tt.concatenate((tt.zeros(1)+initial, tte.cumsum(dI)))

def Efunc(z, Om, w):
    Ol = 1.0-Om
    opz = 1.0 + z
    return tt.sqrt(Om*opz*opz*opz + Ol*opz**(3*(1+w)))

def dH_of_H0(H0):
    return 4.42563416002 * (67.74/H0)

def comoving_distance(zs, H0, Om, w):
    dH = dH_of_H0(H0)

    cdi = 1.0/Efunc(zs, Om, w)

    return dH*cumtrapz(cdi, zs)

def interp1d_theano(x, xs, ys):
    ih = tte.searchsorted(xs, x)
    il = ih - 1

    xh = xs[ih]
    xl = xs[il]
    yh = ys[ih]
    yl = ys[il]

    r = (x-xl)/(xh-xl)

    return (1.0-r)*yl + r*yh

def dzddl(dls, zs, H0, Om, w):
    opz = 1.0+zs
    return 1.0/(dls/opz + opz*dH_of_H0(H0)/Efunc(zs, Om, w))

def log_norm_pl(alpha, xl, xh):
    return tt.switch(alpha > -1,
                     -tt.log1p(alpha) + (1+alpha)*tt.log(xh) + tt.log1p(-(xl/xh)**(1+alpha)),
                     -tt.log(-(1+alpha)) + (1+alpha)*tt.log(xl) + tt.log1p(-(xh/xl)**(1+alpha)))

def log_dNdm1dm2dz(m1s, m2s, dls, zs, MMin, MMax, alpha, beta, gamma, H0, Om, w):
    log_norm_alpha = log_norm_pl(-alpha, MMin, MMax)
    log_norm_beta = log_norm_pl(beta*ones_like(m1s), MMin, m1s)
    opz = 1.0+zs

    log_dNdm1dm2dVdt = tt.switch((MMin < m2s) & (m2s < m1s) & (m1s < MMax),
                                 -alpha*tt.log(m1s) + beta*tt.log(m2s) - log_norm_alpha - log_norm_beta + (gamma-1)*tt.log1p(zs),
                                 np.NINF)

    return log_dNdm1dm2dVdt + tt.log(4.0*pi) + 2.0*tt.log(dls/opz) + tt.log(dH_of_H0(H0)/Efunc(zs, Om, w))

def kde_log_likelihood(m1, m2, dl, m1det, m2det, dldet, chol_cov):
    pt = tt.stack([m1, m2, dl])
    pts = tt.stack([m1det, m2det, dldet], axis=1)

    dp = pt-pts

    r = tt.slinalg.solve_lower_triangular(chol_cov, dp.T)
    chi2 = tt.sum(r*r, axis=0)

    return pm.logsumexp(-0.5*chi2)

def make_model(m1det, m2det, dldet, m1sel, m2sel, dlsel, log_wtsel, Ndraw, Tobs, cosmo_prior=False, zmax=8):
    nobs, nsamp = m1det.shape

    cms_chol = []
    for m1d, m2d, dld in zip(m1det, m2det, dldet):
        cm = cov(row_stack((m1d, m2d, dld)), rowvar=True)
        cm = cm / len(m1d)**(2.0/7.0) # Scott-like rule for KDE bandwidth
        cms_chol.append(np.linalg.cholesky(cm))
    cms_chol = np.array(cms_chol)

    zinterp = expm1(linspace(log(1), log(1+zmax)+0.1, 1000))
    dli = Planck15.luminosity_distance(zinterp).to(u.Gpc).value
    z_of_dl = interp1d(dli, zinterp)

    m1init = []
    m2init = []
    dlinit = []
    zinit = []
    for m1, m2, dl in zip(m1det, m2det, dldet):
        i = randint(len(m1))

        dlinit.append(dl[i])
        z = z_of_dl(dl[i])
        zinit.append(z)
        m1init.append(m1[i]/(1+z))
        m2init.append(m2[i]/(1+z))

    m1init = array(m1init)
    m2init = array(m2init)
    dlinit = array(dlinit)
    zinit = array(zinit)

    MMin_init = max(np.min(m2init) - 1.0, 3.1)
    MMax_init = min(np.max(m1init)+1.0, 69.0)

    m1init_frac = (m1init - MMin_init)/(MMax_init-MMin_init)
    m2init_frac = (m2init - MMin_init)/(m1init - MMin_init)

    s = m1init_frac < 0
    m1init_frac[s] = 0.01
    s = m1init_frac > 1
    m1init_frac[s] = 0.99

    s = m2init_frac < 0
    m2init_frac[s] = 0.01
    s = m2init_frac > 1
    m2init_frac[s] = 0.99

    m1det = tt.as_tensor_variable(m1det)
    m2det = tt.as_tensor_variable(m2det)
    dldet = tt.as_tensor_variable(dldet)

    m1sel = tt.as_tensor_variable(m1sel)
    m2sel = tt.as_tensor_variable(m2sel)
    dlsel = tt.as_tensor_variable(dlsel)
    log_wtsel = tt.as_tensor_variable(log_wtsel)

    cms_chol = tt.as_tensor_variable(cms_chol)

    zinterp = tt.as_tensor_variable(zinterp)

    m = pm.Model()
    with m:
        # Cosmo structure variables
        if cosmo_prior:
            pH0 = Planck15.H0.to(u.km/u.s/u.Mpc).value
            H0 = pm.Bound(pm.Normal, lower=35, upper=140)('H0', mu=pH0, sd=0.01*pH0)
            Omh2 = pm.Bound(pm.Normal, lower=0, upper=0.3)('Omh2', mu=0.02225+0.1198, sd=sqrt(0.00016**2 + 0.0015**2))
            Om = pm.Deterministic('Om', Omh2/(H0/100)**2)
        else:
            Om = pm.Bound(pm.Normal, lower=0, upper=1)('Om', mu=0.3, sd=0.15)
            H0 = pm.Bound(pm.Normal, lower=35, upper=140)('H0', mu=70.0, sd=12.0)
            Omh2 = pm.Deterministic('Omh2', Om*(H0/100)**2)
        w = pm.Bound(pm.Normal, lower=-2, upper=0)('w', mu=-1.0, sd=0.5)

        # Mass+redshift dist variables
        RUnit = pm.Normal('RUnit', mu=0, sd=1)
        MMin = pm.Bound(pm.Normal, lower=3, upper=10)('MMin', mu=5.0, sd=2.0)
        MMax = pm.Bound(pm.Normal, lower=30, upper=70)('MMax', mu=50.0, sd=10.0)
        alpha = pm.Bound(pm.Normal, lower=-1, upper=3)('alpha', mu=1, sd=1, testval=1.1) # Need testval because alpha = 1 gives numerical singularity
        beta = pm.Bound(pm.Normal, lower=-2, upper=2)('beta', mu=0, sd=1)
        gamma = pm.Bound(pm.Normal, lower=0, upper=6)('gamma', mu=3, sd=1.5)

        # Source variables: m1, m2, z, dL
        m1_frac = pm.Uniform('m1_frac', lower=0, upper=1, shape=(nobs,))
        m2_frac = pm.Uniform('m2_frac', lower=0, upper=1, shape=(nobs,))
        zs = pm.Uniform('zs', lower=0, upper=zmax, shape=(nobs,))

        m1s = pm.Deterministic('m1s', MMin + (MMax-MMin)*m1_frac)
        m2s = pm.Deterministic('m2s', MMin + (m1s-MMin)*m2_frac)

        dcinterp = comoving_distance(zinterp, H0, Om, w)
        dlinterp = dcinterp*(1+zinterp)

        dls = pm.Deterministic('dls', interp1d_theano(zs, zinterp, dlinterp))

        # Population is a "prior" on source-frame.  Our function is density in
        # m1, m2, z, but we sample in m1_frac, m2_frac, z, so need Jacobian
        # d(m1)/d(m1_frac) d(m2)/d(m2_frac) = (MMax-MMin)*(m1s-MMin)
        pm.Potential('population-distribution', tt.sum(log_dNdm1dm2dz(m1s, m2s, dls, zs, MMin, MMax, alpha, beta, gamma, H0, Om, w)) + nobs*tt.log(MMax-MMin) + tt.sum(tt.log(m1s-MMin)))

        # Selection effects

        # Ensure that we can interpolate to find the redshift
        s = dlsel < dlinterp[-1]
        m1sels = m1sel[s]
        m2sels = m2sel[s]
        dlsels = dlsel[s]
        zsels = interp1d_theano(dlsels, dlinterp, zinterp)
        logwtsels = log_wtsel[s]

        log_sel_wts = log_dNdm1dm2dz(m1sels/(1+zsels), m2sels/(1+zsels), dlsels, zsels, MMin, MMax, alpha, beta, gamma, H0, Om, w) - 2.0*tt.log1p(zsels) + tt.log(dzddl(dlsels, zsels, H0, Om, w)) - logwtsels
        log_sel_wts2 = 2.0*log_sel_wts

        log_mu = pm.Deterministic('log_mu', pm.logsumexp(log_sel_wts) - tt.log(Ndraw))
        mu = exp(log_mu)
        mu2 = tt.exp(2*log_mu)
        sigma2 = tt.exp(pm.logsumexp(log_sel_wts2))/(Ndraw*Ndraw) - mu2/Ndraw
        Neff = pm.Deterministic('Neff_det', mu2/sigma2)

        pm.Potential('reject-neff-too-small', tt.switch(Neff > 4.0*nobs, 0.0, np.NINF))

        pm.Potential('normalization', -nobs*log_mu + (3.0*nobs + nobs*nobs)/(2*Neff))

        mu_R = nobs/(Tobs*mu)*(1.0 + nobs/Neff*(1.0 + 2.0*nobs/Neff))
        sigma_R = tt.sqrt(nobs)/(Tobs*mu)*(1.0 + nobs/Neff*(3.0/2.0 + 31.0/8.0*nobs/Neff))
        R = pm.Deterministic('R', mu_R + sigma_R*RUnit)

        # Likelihood
        log_lls, _ = theano.map(kde_log_likelihood, sequences=[m1s*(1+zs), m2s*(1+zs), dls, m1det, m2det, dldet, cms_chol])
        pm.Potential('log-likelihood', tt.sum(log_lls))

    return m
