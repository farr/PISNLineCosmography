from pylab import *

import arviz as az
import astropy.cosmology as cosmo
from astropy.cosmology import Planck15
import astropy.units as u
import corner
import h5py
from scipy.stats import gaussian_kde, norm
import seaborn as sns
from true_params import true_params

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
        mpl.rcParams.update({
            'figure.figsize': self._figsize,
            'text.usetex': True
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

def Hz(z, H0, Om, w, w_a):
    return H0*np.sqrt(Om*(1+z)**3 + (1.0-Om)*(1+z)**(3*(1+w+w_a))*exp(-3*w_a*z/(1+z)))

def load_chains(f, select_subset=None):
    names = ['H0', 'Om', 'w', 'w_p', 'w_a', 'R0_30', 'MMin', 'MMax', 'sigma_min', 'sigma_max', 'alpha', 'beta', 'gamma', 'neff_det', 'm1s', 'm2s', 'dls', 'zs']

    c = {}

    if select_subset is None:
        select_subset = slice(None)
    with h5py.File(f, 'r') as inp:
        nobs = inp.attrs['nobs']
        for n in names:
            arr = squeeze(array(inp[n]))

            if len(arr.shape) == 1:
                c[n] = reshape(arr, (4, -1))
            else:
                c[n] = reshape(arr, (4, -1) + arr.shape[1:])

            c[n] = c[n][select_subset, ...]

        c['nobs'] = nobs
    return c

def traceplot(c):
    fit = az.convert_to_inference_data(c)

    lines = (('H0', {}, true_params['H0']),
             ('Om', {}, true_params['Om']),
             ('w', {}, true_params['w']),
             ('w_p', {}, true_params['w']),
             ('w_a', {}, true_params['w_a']),
             ('R0_30', {}, true_params['R0_30']),
             ('MMin', {}, true_params['MMin']),
             ('MMax', {}, true_params['MMax']),
             ('sigma_max', {}, true_params['sigma_max']),
             ('sigma_min', {}, true_params['sigma_min']),
             ('alpha', {}, true_params['alpha']),
             ('beta', {}, true_params['beta']),
             ('gamma', {}, true_params['gamma']))

    az.plot_trace(fit, var_names=['H0', 'Om', 'w', 'w_p', 'w_a', 'R0_30', 'MMin', 'MMax', 'sigma_min', 'sigma_max', 'alpha', 'beta', 'gamma'], lines=lines)

def neff_det_check_plot(c):
    fit = az.convert_to_inference_data(c)

    az.plot_density(fit, var_names=['neff_det'], credible_interval=0.99)

    xlabel(r'$N_\mathrm{eff}$')
    ylabel(r'$p\left( N_\mathrm{eff} \right)$')

    nobs = c['m1s'].shape[2]
    axvline(4*nobs)

    nemin = percentile(c['neff_det'], 2.5)
    title(r'Two-sigma lower $N_\mathrm{{eff}}$ is factor {:.2f} above limit'.format(nemin/(4*nobs)))

def cosmo_corner_plot(c, *args, **kwargs):
    pts = column_stack([c[n].flatten() for n in ['H0', 'Om', 'w', 'w_p', 'w_a']])

    fig = figure()

    corner.corner(pts, labels=[r'$H_0$', r'$\Omega_M$', r'$w$', r'$w_p$', r'$w_a$'],
                  truths=[Planck15.H0.to(u.km/u.s/u.Mpc).value,
                          Planck15.Om0,
                          -1,
                          -1,
                          0],
                  quantiles=[0.16, 0.84],
                  show_titles=True)

def pop_corner_plot(c, *args, **kwargs):
    pts = column_stack([c[n].flatten() for n in ['R0_30', 'MMin', 'MMax', 'sigma_min', 'sigma_max', 'alpha', 'beta', 'gamma']])

    fig = figure()

    corner.corner(pts,
                  labels=[r'$R_{0,30}$',
                          r'$M_\mathrm{min}$',
                          r'$M_\mathrm{max}$',
                          r'$\sigma_\mathrm{min}$',
                          r'$\sigma_\mathrm{max}$',
                          r'$\alpha$',
                          r'$\beta$',
                          r'$\gamma$'],
                  truths=[true_params['R0_30'],
                          true_params['MMin'],
                          true_params['MMax'],
                          true_params['sigma_min'],
                          true_params['sigma_max'],
                          true_params['alpha'],
                          true_params['beta'],
                          true_params['gamma']],
                  quantiles=[0.16, 0.84],
                  show_titles=True)

def H0_plot(c, *args, **kwargs):
    H0 = c['H0'].flatten()

    sns.distplot(H0)

    l, h = spd_interval(H0, 0.68)
    axvline(l, ls='--')
    axvline(h, ls='--')
    xlabel(r'$H_0$ ($\mathrm{km} \, \mathrm{s}^{-1} \, \mathrm{Mpc}^{-1}$)')
    ylabel(r'$p\left( H_0 \mid d \right)$')
    axvline(Planck15.H0.to(u.km/u.s/u.Mpc).value, color='k')
    title(interval_string(H0, prefix=r'$H_0 = ', postfix=r' \, \mathrm{km} \, \mathrm{s}^{-1} \, \mathrm{Mpc}^{-1}$'))

def optimal_w_plot(c, *args, **kwargs):
    w_p = c['w_p'].flatten()
    w_a = c['w_a'].flatten()

    pts = column_stack((w_p, w_a))
    cm = cov(pts, rowvar=False)
    evals, evecs = np.linalg.eigh(cm)

    w_p_index = argmax(abs(evecs[0,:])) # Find the one with the largest w_p component

    # Ensure the appropriate component is positive!
    evecs[:,w_p_index] *= np.sign(evecs[0,w_p_index])
    evecs[:,1-w_p_index] *= np.sign(evecs[1,1-w_p_index])

    w_p_opts = np.dot(pts, evecs[:,w_p_index])
    w_a_opts = np.dot(pts, evecs[:,1-w_p_index])

    a_p_0 = 1.0/(1.0+0.7)
    a_p = a_p_0 - evecs[1,w_p_index]/evecs[0,w_p_index]
    z_p = 1.0/a_p - 1.0

    fom = 1.0/(std(w_p_opts)*std(w_a_opts))

    fig = figure()

    corner.corner(column_stack((w_p_opts, w_a_opts)),
                  labels=[r'$w_p$', r'$w_a$'],
                  quantiles=[0.16, 0.84],
                  show_titles=True,
                  truths=[-1.0, 0.0])

    print('FOM = {:.1f} (WARNING: prior FOM = 4)'.format(fom))
    print('Pivot redshift = {:.2f}'.format(z_p))

def MMax_plot(c, *args, **kwargs):
    MMax = c['MMax'].flatten()

    sns.distplot(MMax)
    xlabel(r'$M_\mathrm{max}$ ($M_\odot$)')
    ylabel(r'$p\left( M_\mathrm{max} \right)$')
    axvline(true_params['MMax'], color='k')

def Hz_plot(c, *args, color=None, draw_tracks=True, label=None, **kwargs):
    if color is None:
        color = sns.color_palette()[0]

    zs = linspace(0, 2, 1000)

    plot(zs, Hz(zs, Planck15.H0.to(u.km/u.s/u.Mpc).value, Planck15.Om0, -1, 0), '-k')

    Hs = Hz(zs[newaxis,:], c['H0'].flatten()[:,newaxis], c['Om'].flatten()[:,newaxis], c['w'].flatten()[:,newaxis], c['w_a'].flatten()[:,newaxis])

    m = median(Hs, axis=0)
    l = percentile(Hs, 16, axis=0)
    ll = percentile(Hs, 2.5, axis=0)
    h = percentile(Hs, 84, axis=0)
    hh = percentile(Hs, 97.5, axis=0)
    rel_err = (h-l)/(2*m)
    imin = argmin(rel_err)
    print('Redshift at which 1-sigma fractional H(z) interval min of {:.2f} is {:.2f}'.format(rel_err[imin], zs[imin]))

    plot(zs, m, color=color, label=label)
    fill_between(zs, h, l, color=color, alpha=0.25)
    fill_between(zs, hh, ll, color=color, alpha=0.25)

    if draw_tracks:
        hs = c['H0'].flatten()
        Oms = c['Om'].flatten()
        ws = c['w'].flatten()
        was = c['w_a'].flatten()
        for i in np.random.choice(len(hs), size=10, replace=False):
            h = hs[i]
            Om = Oms[i]
            w = ws[i]
            w_a = was[i]

            plot(zs, Hz(zs, h, Om, w, w_a), color=color, alpha=0.25)

    xlabel(r'$z$')
    ylabel(r'$H(z)$ ($\mathrm{km} \, \mathrm{s}^{-1} \, \mathrm{Mpc}^{-1}$)')

    return zs, Hs

def mass_correction_plot(c):
    errorbar(mean(c['zs'], axis=(0,1)), mean(c['m1s'],axis=(0,1)), yerr=std(c['m1s'], axis=(0,1)), xerr=std(c['zs'], axis=(0,1)), fmt='.')
    zs = linspace(0, np.max(c['zs']), 100)
    plot(zs, median(c['MMax'])*ones_like(zs), color=sns.color_palette()[0])
    fill_between(zs, percentile(c['MMax'], 84)*ones_like(zs), percentile(c['MMax'], 16)*ones_like(zs), color=sns.color_palette()[0], alpha=0.25)
    fill_between(zs, percentile(c['MMax'], 97.5)*ones_like(zs), percentile(c['MMax'], 2.5)*ones_like(zs), color=sns.color_palette()[0], alpha=0.25)

    xlabel(r'$z$')
    ylabel(r'$m_1$ ($M_\odot$)')

def post_process(f, select_subset=None):
    c = load_chains(f, select_subset=select_subset)

    traceplot(c)

    neff_det_check_plot(c)

    figure()
    Hz_plot(c)

    cosmo_corner_plot(c)
    pop_corner_plot(c)

    H0_plot(c)

    optimal_w_plot(c)

    MMax_plot(c)
    title(interval_string(c['MMax'].flatten(), prefix=r'$M_\mathrm{max} = ', postfix=' \, M_\odot$'))

    figure()
    mass_correction_plot(c)

    return c
