from pylab import *

import astropy.cosmology as cosmo
from astropy.cosmology import Planck15
import astropy.units as u
import pymc3 as pm
from pymc3.step_methods.hmc.quadpotential import QuadPotentialFull
import theano
import theano.tensor as tt
import theano.tensor.extra_ops as te

def Ez(zs, Om, w):
    opz = 1 + zs
    opz2 = opz*opz
    opz3 = opz2*opz

    return tt.sqrt(opz3*Om + (1-Om)*opz**(3*(1+w)))

def dzddL(dls, zs, dH, Om, w):
    opz = 1+zs
    return 1/(dls/opz + opz*dH/Ez(zs, Om, w))

def log_dNdm1dm2ddLdt(m1s, m2s, dls, zs, R0, MMin, MMax, alpha, beta, gamma, dH, Om, w, smooth_low, smooth_high):
    oma = 1-alpha
    log_m1norm = tt.log(oma/(MMax**oma - MMin**oma))

    opb = 1+beta
    m2norm = opb/(m1s**opb - MMin**opb)
    m2norm = tt.switch(m2norm < 0, -m2norm, m2norm)
    log_m2norm = tt.log(m2norm)

    log_dNdm1dm2dVdt = tt.log(R0) - alpha*tt.log(m1s) + beta*tt.log(m2s) + log_m1norm + log_m2norm + (gamma-1)*tt.log1p(zs)

    log_dVdz = tt.log(4.0*pi) + 2*tt.log(dls/(1+zs)) + tt.log(dH) - tt.log(Ez(zs, Om, w))
    log_dzddL = tt.log(dzddL(dls, zs, dH, Om, w))

    log_sl = tt.switch(m2s < MMin, -0.5*(m2s-MMin)**2/smooth_low**2, 0.0)
    log_sh = tt.switch(m1s > MMax, -0.5*(m1s-MMax)**2/smooth_high**2, 0.0)

    return log_dNdm1dm2dVdt + log_dVdz + log_dzddL + log_sl + log_sh

def dls_of_zs(zs, dH, Om, w):
    ddcdz = dH/Ez(zs, Om, w)

    dzs = (zs[1:] - zs[:-1])
    ddcdz_ave = 0.5*(ddcdz[1:] + ddcdz[:-1])

    dcs = tt.concatenate((tt.zeros(1), te.cumsum(dzs*ddcdz_ave)))

    return dcs*(1+zs)

def interp1d(xs, xi, yi):
    inds = te.searchsorted(xi, xs)

    xl = xi[inds-1]
    xh = xi[inds]

    yl = yi[inds-1]
    yh = yi[inds]

    r = (xs-xl)/(xh-xl)

    return yl*(1-r) + yh*r

def make_model(m1s, m2s, dls, m1s_det, m2s_det, dls_det, wts_det, N_gen, T_obs, z_safety_factor=10, n_interp=1000, smooth_low=0.1, smooth_high=0.5):
    dmax = max(np.max(dls), np.max(dls_det))
    zmax = cosmo.z_at_value(Planck15.luminosity_distance, dmax*u.Gpc)

    zs_interp = linspace(0, zmax*z_safety_factor, n_interp)
    zs_interp = tt.as_tensor_variable(zs_interp)

    log_wts_det = np.log(wts_det)

    bw_high = std(m1s, axis=1)/m1s.shape[1]**0.2
    bw_low = std(m2s, axis=1)/m2s.shape[1]**0.2

    bw_high = bw_high[:,newaxis]
    bw_high = np.where(bw_high < 0.5, 0.5, bw_high)

    bw_low = bw_low[:,newaxis]
    bw_low = np.where(bw_low < 0.05, 0.05, bw_low)

    N_obs = m1s.shape[0]
    N_samp = m1s.shape[1]
    N_det = wts_det.shape[0]

    m = pm.Model()

    with m:
        MMin = 5.0 #pm.Bound(pm.Normal, lower=3, upper=10)('MMin', mu=5, sd=2)
        MMax = pm.Bound(pm.Normal, lower=30, upper=70)('MMax', mu=40, sd=10)

        R0 = pm.Lognormal('R0', mu=log(100), sd=1)

        alpha = pm.Bound(pm.Normal, lower=-3, upper=3)('alpha', mu=1, sd=1)
        beta = pm.Bound(pm.Normal, lower=-3, upper=3)('beta', mu=0, sd=1)
        gamma = pm.Bound(pm.Normal, lower=0, upper=6)('gamma', mu=3, sd=2)

        Om = pm.Bound(pm.Normal, lower=0, upper=1)('Om', mu=0.3, sd=0.1)
        H0 = pm.Bound(pm.Normal, lower=50, upper=100)('H0', mu=70, sd=15)
        w = pm.Bound(pm.Normal, lower=-2, upper=0)('w', mu=-1, sd=0.5)

        dH = 4.42563416002 * (67.74/H0);
        ds_interp = dls_of_zs(zs_interp, dH, Om, w)

        zs = interp1d(dls, ds_interp, zs_interp)
        zs_det = interp1d(dls_det, ds_interp, zs_interp)

        log_dN_det = log_dNdm1dm2ddLdt(m1s_det/(1+zs_det), m2s_det/(1+zs_det), dls_det, zs_det, R0, MMin, MMax, alpha, beta, gamma, dH, Om, w, smooth_low, smooth_high) - log_wts_det - 2*tt.log1p(zs_det)

        N_sum = tt.exp(pm.logsumexp(log_dN_det))
        N2_sum = tt.exp(pm.logsumexp(2*log_dN_det))

        Nex = pm.Deterministic('Nex', T_obs/N_gen*N_sum)
        sigma_Nex = T_obs/N_gen*tt.sqrt(N2_sum - N_sum*N_sum/N_det)

        Neff_det = pm.Deterministic('neff_det', Nex*Nex/(sigma_Nex*sigma_Nex))

        log_dN_likelihood = log_dNdm1dm2ddLdt(m1s/(1+zs), m2s/(1+zs), dls, zs, R0, MMin, MMax, alpha, beta, gamma, dH, Om, w, bw_low, bw_high) - 2*tt.log1p(zs)

        pm.Potential('log-likelihood', tt.sum(pm.logsumexp(log_dN_likelihood, axis=1) - tt.log(N_samp)))
        pm.Potential('Poisson-norm', -Nex)

    return m

# Uses tricks from DFM's blog post: https://dfm.io/posts/pymc3-mass-matrix/
def get_step_for_trace(trace=None, model=None,
                       regular_window=5, regular_variance=1e-3,
                       **kwargs):
    model = pm.modelcontext(model)

    # If not given, use the trivial metric
    if trace is None:
        cov = np.eye(model.ndim)
        return pm.NUTS(scaling=cov, is_cov=True, **kwargs)

    # Loop over samples and convert to the relevant parameter space;
    # I'm sure that there's an easier way to do this, but I don't know
    # how to make something work in general...
    samples = np.empty((len(trace) * trace.nchains, model.ndim))
    i = 0
    for chain in trace._straces.values():
        for p in chain:
            samples[i] = model.bijection.map(p)
            i += 1

    # Compute the sample covariance
    cov = np.cov(samples, rowvar=0)

    # Stan uses a regularized estimator for the covariance matrix to
    # be less sensitive to numerical issues for large parameter spaces.
    # In the test case for this blog post, this isn't necessary and it
    # actually makes the performance worse so I'll disable it, but I
    # wanted to include the implementation here for completeness
    N = len(samples)
    cov = cov * N / (N + regular_window)
    cov[np.diag_indices_from(cov)] += \
        regular_variance * regular_window / (N + regular_window)

    # Use the sample covariance as the inverse metric
    return pm.NUTS(scaling=cov, is_cov=True, **kwargs)

def sample(model, n_tune, n_draw, n_jobs, *sampler_args, **sampler_kwargs):
    assert n_tune >= 100, 'cannot tune for fewer than 100 steps!'
    n_start = 25

    n_window = [n_start]
    while np.sum(n_window) < n_tune:
        n_window.append(2*n_window[-1])

    with model:
        start = None
        burnin_trace = None
        for steps in n_window[:-1]:
            step = get_step_for_trace(burnin_trace)
            burnin_trace = pm.sample(start=start, tune=steps, draws=2, step=step, compute_convergence_checks=False, discard_tuned_samples=False, njobs=n_jobs, *sampler_args, **sampler_kwargs)
            start = [t[-1] for t in burnin_trace._straces.values()]

        step = get_step_for_trace(burnin_trace)
        dense_trace = pm.sample(draws=n_draw, tune=n_window[-1], step=step, start=start, njobs=n_jobs, *sampler_args, **sampler_kwargs)

    return dense_trace
