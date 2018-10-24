#!/usr/bin/env python

from pylab import *

import astropy.cosmology as cosmo
from astropy.cosmology import Planck15
import astropy.units as u
import h5py
import multiprocessing as multi
import pymc3 as pm
import scipy.stats as ss
import sys
import theano
import theano.tensor as tt
import theano.tensor.extra_ops as te
from tqdm import tqdm
from true_params import uncert

with h5py.File('thetas.h5', 'r') as inp:
    ts = array(inp['Theta'])

mu_t = mean(ts)
sigma_t = std(ts)

def interp2d(x, y, xs, ys, zs):
    xs = tt.as_tensor_variable(xs)
    ys = tt.as_tensor_variable(ys)
    zs = tt.as_tensor_variable(zs)

    i = te.searchsorted(xs, x)
    j = te.searchsorted(ys, y)

    i = tt.switch(i < 1, 1, i)
    i = tt.switch(i >= xs.shape[0], xs.shape[0]-1, i)

    j = tt.switch(j < 1, 1, j)
    j = tt.switch(j >= ys.shape[0], ys.shape[0]-1, j)

    r = (x - xs[i-1])/(xs[i]-xs[i-1])
    s = (y - ys[j-1])/(ys[j]-ys[j-1])

    return (1-r)*(1-s)*zs[i-1,j-1] + r*(1-s)*zs[i,j-1] + (1-r)*s*zs[i-1, j] + r*s*zs[i,j]

with h5py.File('optimal_snr.h5', 'r') as inp:
    ms = array(inp['ms'])
    osnrs = array(inp['SNR'])

m = pm.Model()

# Placeholders for mc, eta, rho, theta
mco = theano.shared(1.0)
eto = theano.shared(1.0)
ro = theano.shared(1.0)
to = theano.shared(1.0)

sigma_mc = theano.shared(1.0)
sigma_eta = theano.shared(1.0)
sigma_rho = theano.shared(1.0)
sigma_theta = theano.shared(1.0)

with m:
    mcdet = pm.Uniform('mcdet', lower=3, upper=120)
    eta = pm.Uniform('eta', lower=0, upper=0.25)

    disc = eta**(4.0/5.0)*mcdet**2 - 4.0*eta**(9.0/5.0)*mcdet**2

    m1det = pm.Deterministic('m1det', (eta**(2.0/5.0)*mcdet + sqrt(disc))/(2*eta))
    m2det = pm.Deterministic('m2det', (eta**(2.0/5.0)*mcdet - sqrt(disc))/(2*eta))

    dl = pm.Uniform('dl', lower=0, upper=Planck15.luminosity_distance(2).to(u.Gpc).value)
    theta = pm.Bound(pm.Normal, lower=0, upper=1)('theta', mu=mu_t, sd=sigma_t)

    pm.Potential('m1m2-mceta-jac', tt.log(m1det)+tt.log(m2det)-tt.log(m1det-m2det)-(8.0/5.0)*tt.log(eta))

    rho_optimal = interp2d(m1det, m2det, ms, ms, osnrs)/dl
    rho_true = rho_optimal*theta

    theta_obs = pm.Normal('theta_obs', mu=theta, sd=sigma_theta, observed=to)
    pm.Potential('theta_obs_cum', -tt.log(0.5)-tt.log(tt.erf(theta/(tt.sqrt(2)*sigma_theta)) - tt.erf((theta-1)/(tt.sqrt(2)*sigma_theta))))

    rho_obs = pm.Normal('rho_obs', mu=rho_true, sd=sigma_rho, observed=ro)

    eta_obs = pm.Normal('eta_obs', mu=eta, sd=sigma_eta, observed=eto)
    pm.Potential('eta_obs_cum', -tt.log(0.5)-tt.log(tt.erf(eta/(tt.sqrt(2)*sigma_eta)) - tt.erf((eta-0.25)/(tt.sqrt(2)*sigma_eta))))

    mc_obs = pm.Lognormal('mc_obs', mu=tt.log(mcdet), sd=sigma_mc, observed=mco)

with h5py.File('observations.h5', 'r') as inp:
    m1s = array(inp['m1s'])
    m2s = array(inp['m2s'])
    zs = array(inp['zs'])
    thetas = array(inp['thetas'])

    mcobs = array(inp['mcobs'])
    etaobs = array(inp['etaobs'])
    rhoobs = array(inp['rhoobs'])
    thetaobs = array(inp['thetaobs'])

    smcs = array(inp['sigma_mc'])
    sets = array(inp['sigma_eta'])
    srhs = array(inp['sigma_rho'])
    sths = array(inp['sigma_t'])

def sample(i, progressbar=False, njobs=1):
    mco.set_value(mcobs[i])
    eto.set_value(etaobs[i])
    ro.set_value(rhoobs[i])
    to.set_value(thetaobs[i])

    sigma_mc.set_value(smcs[i])
    sigma_eta.set_value(sets[i])
    sigma_rho.set_value(srhs[i])
    sigma_theta.set_value(sths[i])

    retries = 0

    with m:
        factor = 1
        while True:
            start = {
                'mcdet': mcobs[i] + smcs[i]*randn(),
                'eta': max(0.0001, min(etaobs[i] + sets[i]*randn(), 0.2499)),
                'dl': random.uniform(low=0.5, high=2),
                'theta': max(0.0001, min(thetaobs[i] + sths[i]*randn(), 0.9999))
            }

            while retries < 5:
                try:
                    trace = pm.sample(draws=1000*factor, tune=1000*factor, init='adapt_diag', start=start, njobs=njobs, chains=4, progressbar=progressbar, nuts_kwargs={'target_accept': 0.95})
                    break
                except ValueError:
                    retries += 1
                    if retries >= 5:
                        raise
                    else:
                        print('Re-trying sampling due to ValueError')

            nef = pm.effective_n(trace)
            ne_min = min([nef[k] for k in ['m1det', 'm2det', 'dl', 'theta']])

            if ne_min > 1000:
                break
            else:
                factor *= 2

    print(pm.summary(trace))

    return i, trace

def thin(arr):
    l = len(arr)

    if l > 4000:
        t = int(round(len(arr)/4000.0))
    else:
        t = 1

    return arr[::t]

if __name__ == '__main__':
    p = multi.Pool()

    nobs = len(m1s)

    with h5py.File('observations.h5', 'a') as f:
        try:
            f['posteriors']
        except:
            g = f.create_group('posteriors')
            g.create_dataset('m1det', data=zeros((nobs, 4000)), compression='gzip', shuffle=True)
            g.create_dataset('m2det', data=zeros((nobs, 4000)), compression='gzip', shuffle=True)
            g.create_dataset('dl', data=zeros((nobs, 4000)), compression='gzip', shuffle=True)
            g.create_dataset('theta', data=zeros((nobs, 4000)), compression='gzip', shuffle=True)

        to_process = []
        for i in range(nobs):
            if np.any(array(f['posteriors']['m1det'][i,:]) == 0):
                to_process.append(i)

        for i, t in tqdm(p.imap_unordered(sample, to_process), total=len(to_process)):
            for k in ['m1det', 'm2det', 'dl', 'theta']:
                f['posteriors'][k][i,:] = thin(t[k])
