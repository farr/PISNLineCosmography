from pylab import *

import arviz as az
import astropy.cosmology as cosmo
from astropy.cosmology import Planck15
import astropy.units as u
import corner
import h5py
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde, norm
import seaborn as sns
from true_params import true_params
import xarray as xr

colwidth = 433.62/72.0
figsize_pub=(colwidth, colwidth)

class PubContextManager(object):
    def __init__(self, figsize=None):
        if figsize is None:
            self._figsize = figsize_pub
        else:
            self._figsize = figsize

        pass

    def __enter__(self):
        self._ctx = sns.plotting_context()
        self._rc = mpl.rcParams.copy()

        sns.set_context('paper')
        plt.rcParams.update({
            'figure.figsize': self._figsize,
            'text.usetex': True,
            'axes.unicode_minus': False
        })

    def __exit__(self, exc_type, exc_value, traceback):
        sns.set_context(self._ctx)
        mpl.rcParams.update(self._rc)

        # Don't squash exception
        return None

def pub_plots(figsize=None):
    return PubContextManager(figsize=figsize)

def spd_interval(samps, p):
    samps = sort(samps)
    N = samps.shape[0]
    Nint = int(round(p*N))
    Nout = N-Nint

    starts = samps[0:Nout]
    ends = samps[N-Nout:N]

    i = np.argmin(ends - starts)

    return starts[i], ends[i]

def interval_string(d, prefix='', postfix='', f=0.68):
    s = d
    m = median(s)
    l, h = spd_interval(s, f)

    dl = m-l
    dh = h-m

    il = 10**(int(np.ceil(log10(dl))) - 2)
    ih = 10**(int(np.ceil(log10(dl))) - 2)

    i = min(il, ih)

    m = i*int(round(m/i))
    dl = i*int(round(dl/i))
    dh = i*int(round(dh/i))

    return prefix + '{:g}^{{+{:g}}}_{{-{:g}}}'.format(m, dh, dl) + postfix

def Hz(z, H0, Om, w):
    return H0*np.sqrt(Om*(1+z)**3 + (1.0-Om)*(1+z)**(3*(1+w)))

def traceplot(c):
    fit = az.convert_to_inference_data(c)

    lines = (('H0', {}, true_params['H0']),
             ('Om', {}, true_params['Om']),
             ('w0', {}, true_params['w']),
             ('R0_30', {}, true_params['R0_30']),
             ('MMin', {}, true_params['MMin']),
             ('MMax', {}, true_params['MMax']),
             ('smooth_min', {}, true_params['smooth_min']),
             ('smooth_max', {}, true_params['smooth_max']),
             ('alpha', {}, true_params['alpha']),
             ('beta', {}, true_params['beta']),
             ('gamma', {}, true_params['gamma']))

    az.plot_trace(fit, var_names=['H0', 'Om', 'w0', 'R0_30', 'MMax', 'smooth_max', 'alpha', 'beta', 'gamma'], lines=lines)

def sampled_variables_scatterplot(c):
    az.plot_pair(c, var_names=['H_p', 'Om', 'w0', 'MMax_d_p', 'MMax_d_p_2sigma', 'alpha', 'beta', 'gamma'])

def neff_det_check_plot(c):
    fit = az.convert_to_inference_data(c)

    az.plot_density(fit, var_names=['neff_det'], credible_interval=0.99)

    xlabel(r'$N_\mathrm{eff}$')
    ylabel(r'$p\left( N_\mathrm{eff} \right)$')

    nobs = c.posterior['m1s'].shape[2]
    axvline(4*nobs)

    nemin = percentile(c.posterior['neff_det'], 2.5)
    title(r'Two-sigma lower $N_\mathrm{{eff}}$ is factor {:.2f} above limit'.format(nemin/(4*nobs)))

def cosmo_corner_plot(c, *args, **kwargs):
    pts = column_stack([c.posterior[n].values.flatten() for n in ['H0', 'Om', 'w0']])

    fig = figure()

    corner.corner(pts, labels=[r'$H_0$', r'$\Omega_M$', r'$w$'],
                  truths=[Planck15.H0.to(u.km/u.s/u.Mpc).value,
                          Planck15.Om0,
                          -1],
                  quantiles=[0.16, 0.84],
                  show_titles=True)

def pop_corner_plot(c, *args, **kwargs):
    pts = column_stack([c.posterior[n].values.flatten() for n in ['R0_30', 'MMax', 'smooth_max', 'alpha', 'beta', 'gamma']])

    fig = figure()

    corner.corner(pts,
                  labels=[r'$R_{0,30}$',
                          r'$M_\mathrm{max}$',
                          r'$\sigma_\mathrm{max}$',
                          r'$\alpha$',
                          r'$\beta$',
                          r'$\gamma$'],
                  truths=[true_params['R0_30'],
                          true_params['MMax'],
                          true_params['smooth_max'],
                          true_params['alpha'],
                          true_params['beta'],
                          true_params['gamma']],
                  quantiles=[0.16, 0.84],
                  show_titles=True)

def H0_plot(c, *args, **kwargs):
    H0 = c.posterior['H0'].values.flatten()

    sns.distplot(H0)

    l, h = spd_interval(H0, 0.68)
    axvline(l, ls='--')
    axvline(h, ls='--')
    xlabel(r'$H_0$ ($\mathrm{km} \, \mathrm{s}^{-1} \, \mathrm{Mpc}^{-1}$)')
    ylabel(r'$p\left( H_0 \mid d \right)$')
    axvline(Planck15.H0.to(u.km/u.s/u.Mpc).value, color='k')
    title(interval_string(H0, prefix=r'$H_0 = ', postfix=r' \, \mathrm{km} \, \mathrm{s}^{-1} \, \mathrm{Mpc}^{-1}$'))

def pure_DE_w_plot(c, *args, **kwargs):
    notitle = kwargs.pop('notitle', False)
    nolines = kwargs.pop('nolines', False)
    color = kwargs.pop('color', None)

    wsamps = c.posterior['w0'].values.flatten()

    m = median(wsamps)
    l = percentile(wsamps, 16)
    h = percentile(wsamps, 84)

    sns.distplot(wsamps, *args, color=color, **kwargs)

    if not nolines:
        axvline(m, color=color)
        axvline(h, ls='--', color=color)
        axvline(l, ls='--', color=color)

    axvline(-1, color='k')

    xlabel(r'$w$')
    ylabel(r'$p\left(w\right)$')
    if not notitle:
        title(r'$w = {:.3f}^{{+{:.3f}}}_{{-{:.3f}}}$'.format(m, h-m, m-l))

    return wsamps

def constrained_versus_w0_plot(c):
    nc, ns = c.posterior.w0.values.shape

    sns.kdeplot(c.posterior.w0.values.flatten(), label='This Measurement')

    log_rs = log(rand(nc, ns))

    s = log_rs < c.posterior.constrained_cosmo_log_wts.values

    sns.kdeplot(c.posterior.w0.values[s], label=r'+ 1% $H_0$ + CMB $\omega_M$')

    xlabel(r'$w_0$')
    ylabel(r'$p\left( w_0 \right)$')

    wts = exp(c.posterior.constrained_cosmo_log_wts.values)
    norm = np.sum(wts)

    mu = np.sum(wts*c.posterior.w0.values)/norm
    sigma = np.sqrt(np.sum(wts*np.square(c.posterior.w0.values - mu))/norm)

    title(r'$w_0 = {:.2f} \pm {:.2f}$'.format(mu, sigma))

def MMax_plot(c, *args, **kwargs):
    MMax = c.posterior['MMax'].values.flatten()

    sns.distplot(MMax)
    xlabel(r'$M_\mathrm{max}$ ($M_\odot$)')
    ylabel(r'$p\left( M_\mathrm{max} \right)$')
    axvline(true_params['MMax'], color='k')

def Hz_plot(c, *args, color=None, draw_tracks=True, label=None, **kwargs):
    if color is None:
        color = sns.color_palette()[0]

    zs = linspace(0, 2, 1000)

    plot(zs, Hz(zs, Planck15.H0.to(u.km/u.s/u.Mpc).value, Planck15.Om0, -1), '-k')

    Hs = Hz(zs[newaxis,:], c.posterior['H0'].values.flatten()[:,newaxis], c.posterior['Om'].values.flatten()[:,newaxis], c.posterior['w0'].values.flatten()[:,newaxis])

    m = median(Hs, axis=0)
    l = percentile(Hs, 16, axis=0)
    ll = percentile(Hs, 2.5, axis=0)
    h = percentile(Hs, 84, axis=0)
    hh = percentile(Hs, 97.5, axis=0)
    rel_err = (h-l)/(2*m)
    imin = argmin(rel_err)
    print('Redshift at which 1-sigma fractional H(z) interval min of {:.3f} is {:.2f}'.format(rel_err[imin], zs[imin]))

    plot(zs, m, color=color, label=label)
    fill_between(zs, h, l, color=color, alpha=0.25)
    fill_between(zs, hh, ll, color=color, alpha=0.25)

    if draw_tracks:
        hs = c.posterior['H0'].values.flatten()
        Oms = c.posterior['Om'].values.flatten()
        ws = c.posterior['w0'].values.flatten()
        for i in np.random.choice(len(hs), size=10, replace=False):
            h = hs[i]
            Om = Oms[i]
            w = ws[i]

            plot(zs, Hz(zs, h, Om, w), color=color, alpha=0.25)

    xlabel(r'$z$')
    ylabel(r'$H(z)$ ($\mathrm{km} \, \mathrm{s}^{-1} \, \mathrm{Mpc}^{-1}$)')

    return zs, Hs

def mass_correction_plot(c):
    errorbar(mean(c.posterior['zs'], axis=(0,1)), mean(c.posterior['m1s'],axis=(0,1)), yerr=std(c.posterior['m1s'], axis=(0,1)), xerr=std(c.posterior['zs'], axis=(0,1)), fmt='.')
    zs = linspace(0, np.max(c.posterior['zs']), 100)
    plot(zs, median(c.posterior['MMax'])*ones_like(zs), color=sns.color_palette()[0])
    fill_between(zs, percentile(c.posterior['MMax'], 84)*ones_like(zs), percentile(c.posterior['MMax'], 16)*ones_like(zs), color=sns.color_palette()[0], alpha=0.25)
    fill_between(zs, percentile(c.posterior['MMax'], 97.5)*ones_like(zs), percentile(c.posterior['MMax'], 2.5)*ones_like(zs), color=sns.color_palette()[0], alpha=0.25)

    xlabel(r'$z$')
    ylabel(r'$m_1$ ($M_\odot$)')

def post_process(f, select_subset=None):
    c = add_constrained_cosmo(az.from_netcdf(f))

    print('Minimum effective sample size is')
    es = az.ess(c).min()
    print(min([es[k] for k in es.keys()]))

    print('Number of constrained samples is {:.1f}'.format(np.sum(np.exp(c.posterior.constrained_cosmo_log_wts.values))))

    traceplot(c)

    figure()
    sampled_variables_scatterplot(c)

    figure()
    neff_det_check_plot(c)

    figure()
    Hz_plot(c)

    cosmo_corner_plot(c)
    pop_corner_plot(c)

    figure()
    H0_plot(c)

    figure()
    pure_DE_w_plot(c)

    figure()
    constrained_versus_w0_plot(c)

    figure()
    MMax_plot(c)
    title(interval_string(c.posterior['MMax'].values.flatten(), prefix=r'$M_\mathrm{max} = ', postfix=' \, M_\odot$'))

    figure()
    mass_correction_plot(c)

    return c

def constrained_cosmo_log_weights(H0s, Oms, ws):
    H0_wide_prior = norm(loc=70, scale=15)
    Om_wide_prior = norm(loc=0.3, scale=0.15)

    H0_Planck = Planck15.H0.to(u.km/u.s/u.Mpc).value
    H0_narrow_prior = norm(loc=H0_Planck, scale=0.01*H0_Planck)
    Omh2_narrow_prior = norm(loc=0.02225+0.1198, scale=sqrt(0.00016**2 + 0.0015**2))

    hs = H0s/100
    Omh2s = Oms*hs*hs

    return H0_narrow_prior.logpdf(H0s) + Omh2_narrow_prior.logpdf(Omh2s) + 2.0*log(hs) - H0_wide_prior.logpdf(H0s) - Om_wide_prior.logpdf(Oms)

def constrained_cosmo_samples(c, size=4000):
    H0 = c.posterior.H0.values.flatten()
    Om = c.posterior.Om.values.flatten()
    w0 = c.posterior.w0.values.flatten()

    kde = gaussian_kde(row_stack((H0, Om, w0)))

    cpts = zeros((0,3))
    while cpts.shape[0] < size:
        pts = kde.resample(size)
        H0s, Oms, w0s = pts

        log_wts = constrained_cosmo_log_weights(H0s, Oms, w0s)

        log_rs = log(rand(len(log_wts))) + np.max(log_wts)

        s = (log_rs < log_wts) & (w0s > -2) & (w0s < 0) & (Oms > 0) & (Oms < 1) & (H0s > 0)

        cpts = np.concatenate((cpts, pts[:,s].T), axis=0)

    return cpts

def add_constrained_cosmo(c):
    coords = [('chain', c.posterior.chain), ('draw', c.posterior.draw)]

    log_wts = constrained_cosmo_log_weights(c.posterior.H0.values, c.posterior.Om.values, c.posterior.w0.values)
    log_wts = log_wts - np.max(log_wts)

    c.posterior = c.posterior.assign({'constrained_cosmo_log_wts': xr.DataArray(log_wts, coords=coords)})

    return c
