from pylab import *

import astropy.cosmology as cosmo
from astropy.cosmology import Planck15
import astropy.units as u
import pymc3 as pm
from pymc3.step_methods.hmc.quadpotential import QuadPotentialFull
import theano
import theano.tensor as tt
import theano.tensor.extra_ops as te

def softened_power_law_pdf_unnorm(xs, alpha, xmin, xmax, sigma_min, sigma_max):
    return tt.exp(softened_power_law_logpdf_unnorm(xs, alpha, xmin, xmax, sigma_min, sigma_max))

def softened_power_law_logpdf_unnorm(xs, alpha, xmin, xmax, sigma_min, sigma_max):
    return alpha*tt.log(xs) + tt.switch(xs > xmin,
                                        tt.switch(xs < xmax,
                                                  0.0,
                                                  -0.5*(tt.log(xs)-tt.log(xmax))**2/sigma_max**2),
                                        -0.5*(tt.log(xs)-tt.log(xmin))**2/sigma_min**2)

def Ez(zs, Om, w):
    opz = 1 + zs
    opz2 = opz*opz
    opz3 = opz2*opz

    return tt.sqrt(opz3*Om + (1-Om)*opz**(3*(1+w)))

def dzddL(dls, zs, dH, Om, w):
    opz = 1+zs
    return 1/(dls/opz + opz*dH/Ez(zs, Om, w))

def interp1d(xs, xi, yi):
    inds = te.searchsorted(xi, xs)

    xl = xi[inds-1]
    xh = xi[inds]

    yl = yi[inds-1]
    yh = yi[inds]

    r = (xs-xl)/(xh-xl)

    return yl*(1-r) + yh*r

def log_dNdm1dm2ddLdt(m1s, m2s, dls, zs, R0, MMin, MMax, alpha, beta, gamma, dH, Om, w, smooth_low, smooth_high, ms_norm):
    dms = ms_norm[1:] - ms_norm[:-1]
    pms_alpha = softened_power_law_pdf_unnorm(ms_norm, -alpha, MMin, MMax, smooth_low, smooth_high)
    pms_beta = softened_power_law_pdf_unnorm(ms_norm, beta, MMin, MMax, smooth_low, smooth_high)

    cum_beta = tt.concatenate((tt.zeros(1), tt.cumsum(0.5*dms*(pms_beta[1:] + pms_beta[:-1]))))

    log_norm_alpha = tt.log(tt.sum(0.5*dms*(pms_alpha[1:] + pms_alpha[:-1])))
    log_norm_beta = tt.log(interp1d(m1s, ms_norm, cum_beta))

    log_dNdm1dm2dVdt = tt.log(R0) + softened_power_law_logpdf_unnorm(m1s, -alpha, MMin, MMax, smooth_low, smooth_high) + softened_power_law_logpdf_unnorm(m2s, beta, MMin, MMax, smooth_low, smooth_high) - log_norm_alpha - log_norm_beta + (gamma-1)*tt.log1p(zs)

    log_dVdz = tt.log(4.0*pi) + 2*tt.log(dls/(1+zs)) + tt.log(dH) - tt.log(Ez(zs, Om, w))
    log_dzddL = tt.log(dzddL(dls, zs, dH, Om, w))

    return log_dNdm1dm2dVdt + log_dVdz + log_dzddL

def dls_of_zs(zs, dH, Om, w):
    ddcdz = dH/Ez(zs, Om, w)

    dzs = (zs[1:] - zs[:-1])
    ddcdz_ave = 0.5*(ddcdz[1:] + ddcdz[:-1])

    dcs = tt.concatenate((tt.zeros(1), te.cumsum(dzs*ddcdz_ave)))

    return dcs*(1+zs)

def make_model(m1s_data, m2s_data, dls_data, m1s_det, m2s_det, dls_det, wts_det, N_gen, T_obs, z_safety_factor=10, n_interp=1000, cosmo_constraints=False, dlogm_interp=1e-3):
    dmax = max(np.max(dls_data), np.max(dls_det))
    zmax = cosmo.z_at_value(Planck15.luminosity_distance, dmax*u.Gpc)

    zs_interp = linspace(0, zmax*z_safety_factor, n_interp)
    zs_interp = tt.as_tensor_variable(zs_interp)

    ms_interp = tt.as_tensor_variable(np.exp(np.arange(log(1), log(200), dlogm_interp)))

    log_wts_det = np.log(wts_det)

    N_obs = m1s_data.shape[0]
    N_samp = m1s_data.shape[1]
    N_det = wts_det.shape[0]

    m = pm.Model()

    bws = []
    for i in range(N_obs):
        pts = column_stack((m1s_data[i,:], m2s_data[i,:], dls_data[i,:]))
        cm = cov(pts, rowvar=False)
        bws.append(cm / N_samp**(2.0/7.0))

    chol_bws = [np.linalg.cholesky(b) for b in bws]

    with m:
        smooth_low = pm.Lognormal('sigma_low', mu=log(0.1), sd=1)
        smooth_high = pm.Lognormal('sigma_high', mu=log(0.1), sd=1)

        MMin = pm.Bound(pm.Normal, lower=3, upper=10)('MMin', mu=5, sd=2)
        MMax = pm.Bound(pm.Normal, lower=30, upper=70)('MMax', mu=40, sd=10)

        R0 = pm.Lognormal('R0', mu=log(100), sd=1)

        #unit_normal = pm.Normal('unit_normal', mu=0, sd=1)

        alpha = pm.Bound(pm.Normal, lower=-3, upper=3)('alpha', mu=0.75, sd=1)
        beta = pm.Bound(pm.Normal, lower=-3, upper=3)('beta', mu=0, sd=1)
        gamma = pm.Bound(pm.Normal, lower=0, upper=6)('gamma', mu=3, sd=2)

        if cosmo_constraints:
            H0 = pm.Bound(pm.Normal, lower=50, upper=100)('H0', mu=Planck15.H0.to(u.km/u.s/u.Mpc).value, sd=0.01*Planck15.H0.to(u.km/u.s/u.Mpc).value)
            Om_h2 = pm.Bound(pm.Normal, lower=0, upper=0.5)('Om_h2', mu=0.02225+0.1198, sd=sqrt(0.00016**2 + 0.0015**2))
            Om = pm.Deterministic('Om', Om_h2/(H0/100)**2)
        else:
            Om = pm.Bound(pm.Normal, lower=0, upper=1)('Om', mu=0.3, sd=0.1)
            H0 = pm.Bound(pm.Normal, lower=50, upper=100)('H0', mu=70, sd=15)
        w = pm.Bound(pm.Normal, lower=-2, upper=0)('w', mu=-1, sd=0.5)

        m1s = pm.Uniform('m1s', 3.0, 100.0, shape=(N_obs,))
        m2_fracs = pm.Uniform('m2_fracs', 0, 1, shape=(N_obs,))
        m2s = pm.Deterministic('m2s', 3.0 + (m1s-3.0)*m2_fracs)
        pm.Potential('m2fracjacobian', tt.sum(tt.log(m1s-3.0)))
        dls = pm.Uniform('dls', 0.0, 2.0*dmax, shape=(N_obs,))

        dH = 4.42563416002 * (67.74/H0);
        ds_interp = dls_of_zs(zs_interp, dH, Om, w)

        zs = pm.Deterministic('zs', interp1d(dls, ds_interp, zs_interp))
        zs_det = interp1d(dls_det, ds_interp, zs_interp)

        # Set R0 = 1.0; we will put the scale back in later
        log_dN_det = log_dNdm1dm2ddLdt(m1s_det/(1+zs_det), m2s_det/(1+zs_det), dls_det, zs_det, 1.0, MMin, MMax, alpha, beta, gamma, dH, Om, w, smooth_low, smooth_high, ms_interp) - log_wts_det - 2*tt.log1p(zs_det)

        N_sum = tt.exp(pm.logsumexp(log_dN_det))
        N2_sum = tt.exp(pm.logsumexp(2*log_dN_det))

        # Note: R0 put back in by hand here
        mu_N_det = R0*T_obs/N_gen*N_sum

        # ...but not here, where the uncertainty is the relative uncertainty
        sigma_rel_det2 = N2_sum/(N_sum*N_sum) - 1.0/N_gen
        sigma_rel_det = tt.sqrt(sigma_rel_det2)

        Neff_det = pm.Deterministic('neff_det', 1.0/sigma_rel_det2)

        Nex = pm.Deterministic('Nex', mu_N_det)

        log_dN_population = log_dNdm1dm2ddLdt(m1s, m2s, dls, zs, R0, MMin, MMax, alpha, beta, gamma, dH, Om, w, smooth_low, smooth_high, ms_interp)
        pm.Potential('population', tt.sum(log_dN_population))

        for i in range(N_obs):
            cbw = chol_bws[i]
            mu = tt.as_tensor_variable([m1s[i]*(1+zs[i]), m2s[i]*(1+zs[i]), dls[i]])

            dx = tt.slinalg.solve_lower_triangular(cbw, (column_stack((m1s_data[i,:], m2s_data[i,:], dls_data[i,:]))-mu).T)

            lls = -0.5*tt.dot(dx.T, dx)

            pm.Potential('log-likelihood-{:d}'.format(i), pm.logsumexp(lls))

        pm.Potential('norm', -Nex)

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
