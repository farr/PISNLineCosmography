#!/usr/bin/env python

# Set the backend to a non-displaying one.
import matplotlib
matplotlib.use('PDF')

from pylab import *

from argparse import ArgumentParser
import arviz as az
import astropy.cosmology as cosmo
from astropy.cosmology import Planck15
import astropy.units as u
from corner import corner
import h5py
import pystan
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
import sys
from true_params import true_params
from tqdm import tqdm

p = ArgumentParser()

post = p.add_argument_group('Event Options')
post.add_argument('--sampfile', metavar='FILE.h5', default='observations.h5', help='posterior samples file (default: %(default)s)')
post.add_argument('--subset', metavar='DESIGNATOR', help='name of the attribute giving the number of detection to analyze (default: all)')
post.add_argument('--event-begin', metavar='N', type=int, help='beginning of range of event indices to analyze')
post.add_argument('--event-end', metavar='N', type=int, help='end of range of event indices to analyze (not inclusive)')
post.add_argument('--livetime', metavar='T', type=float, help='live time of event range')
post.add_argument('--nmix', metavar='N', default=7, type=int, help='number of Gaussians in GMM likelihood approx (default: %(default)s)')

sel = p.add_argument_group('Selection Function Options')
sel.add_argument('--selfile', metavar='FILE.h5', default='selected.h5', help='file containing records of successful injections for VT estimation (default: %(default)s)')
sel.add_argument('--nsel', metavar='N', type=int, help='number of selected systems to include (default: all)')

copt = p.add_argument_group('Cosmology Options')
copt.add_argument('--cosmo-prior', action='store_true', help='use a prior on H0 and Om*h^2 from SNe and CMB (default: broad priors)')

samp = p.add_argument_group('Sampling Options')
samp.add_argument('--iter', metavar='N', type=int, default=1000, help='number of sampling iterations (equal amount of tuning; default: %(default)s)')
samp.add_argument('--thin', metavar='N', type=int, default=1, help='thin samples (default: %(default)s)')

oop = p.add_argument_group('Output Options')
oop.add_argument('--chainfile', metavar='F', default='population.h5', help='output file (default: %(default)s)')
oop.add_argument('--tracefile', metavar='F', default='traceplot.pdf', help='traceplot file (default: %(default)s)')

args = p.parse_args()

# Check consistency among event selection options:
if args.subset is not None:
    if args.event_begin is not None or args.event_end is not None:
        raise ValueError('--subset and --event-begin or --event-end are mutually exclusive')
    if args.livetime is not None:
        raise ValueError('--subset and --livetime are mutually exclusive')
if args.event_begin is not None or args.event_end is not None:
    if args.livetime is None:
        raise ValueError('require --livetime with --event-begin or --event-end')

print('Called with the following command line:')
print(' '.join(sys.argv))

chain = {}
with h5py.File(args.sampfile, 'r') as inp:
    for n in ['m1det', 'm2det', 'dl']:
        chain[n] = array(inp['posteriors'][n])
    if args.subset is not None:
        nn, Tobs = inp.attrs[args.subset]
        nn = int(round(nn))
        for k in ['m1det', 'm2det', 'dl']:
            chain[k] = chain[k][:nn,:]
    else:
        if args.event_begin is not None:
            istart = args.event_begin
        else:
            istart = 0

        if args.event_end is not None:
            iend = args.event_end
        else:
            iend = chain['m1det'].shape[0]

        for k in ['m1det', 'm2det', 'dl']:
            chain[k] = chain[k][istart:iend, :]

        if args.livetime is not None:
            Tobs = args.livetime
        else:
            Tobs = inp.attrs['Tobs']

with h5py.File(args.selfile, 'r') as inp:
    N_gen = inp.attrs['N_gen']

    m1s_det = array(inp['m1det'])
    m2s_det = array(inp['m2det'])
    dls_det = array(inp['dl'])
    wts_det = array(inp['wt'])

if args.nsel is not None:
    f = float(args.nsel)/float(len(m1s_det))

    s = np.random.choice(m1s_det.shape[0], replace=False, size=args.nsel)

    N_gen = int(round(f*N_gen))
    m1s_det = m1s_det[s]
    m2s_det = m2s_det[s]
    dls_det = dls_det[s]
    wts_det = wts_det[s]

ndet = m1s_det.shape[0]

nobs = chain['m1det'].shape[0]

gmm_wts = []
gmm_means = []
gmm_covs = []

gmm = GaussianMixture(args.nmix)

for i in tqdm(range(nobs), desc='GMM Fitting'):
    pts = column_stack((chain['m1det'][i,:],
                        chain['m2det'][i,:],
                        chain['dl'][i,:]))
    gmm.fit(pts)

    gmm_wts.append(gmm.weights_)
    gmm_means.append(gmm.means_)
    gmm_covs.append(gmm.covariances_)

zi = expm1(linspace(log(1), log(11), 1000))
di = Planck15.luminosity_distance(zi).to(u.Gpc).value
z_of_d = interp1d(di, zi)

mu_samp = []
chol_cov_samp = []
for i in tqdm(range(nobs), desc='Mean/Sigma Samples'):
    q = chain['m2det'][i,:] / chain['m1det'][i,:]
    pts = column_stack((log(chain['m1det'][i,:]),
                        log(q) - log(1-q),
                        log(chain['dl'][i,:])))

    mu_samp.append(mean(pts, axis=0))
    chol_cov_samp.append(np.linalg.cholesky(cov(pts, rowvar=False)))

ninterp = 1000
zMax = 10
zinterp = expm1(linspace(log(1), log(zMax+1), ninterp))

m = pystan.StanModel(file='model.stan')
d = {
    'nobs': nobs,
    'nsel': ndet,
    'ngmm': args.nmix,

    'Tobs': Tobs,
    'N_gen': N_gen,

    'weights': gmm_wts,
    'means': gmm_means,
    'covs': gmm_covs,

    'mu_samp': mu_samp,
    'chol_cov_samp': chol_cov_samp,

    'm1sel': m1s_det,
    'm2sel': m2s_det,
    'dlsel': dls_det,
    'log_wtsel': np.log(wts_det),

    'ninterp': ninterp,
    'zinterp': zinterp,

    'cosmo_prior': 1 if args.cosmo_prior else 0,

    'z_p': true_params['z_p'],
    'd_p': Planck15.luminosity_distance(true_params['z_p']).to(u.Gpc).value
}

def init(chain_index=None):
    if args.cosmo_prior:
        Hp = Planck15.H0.to(u.km/u.s/u.Mpc).value*Planck15.efunc(true_params['z_p'])*(1+0.01*randn())
        Om = Planck15.Om0*(1+0.01*randn())
    else:
        Hp = Planck15.H0.to(u.km/u.s/u.Mpc).value*Planck15.efunc(true_params['z_p'])*(1+0.1*randn())
        Om = Planck15.Om0*(1+0.1*randn())
    w = -1 + 0.1*randn()

    c = cosmo.FlatwCDM(Hp/Planck15.efunc(true_params['z_p']), Om, w)

    MMax_d_p = (45 + 5*randn())*(1+cosmo.z_at_value(c.luminosity_distance, Planck15.luminosity_distance(true_params['z_p']).to(u.Gpc)))
    smooth_max = 0.1 + 0.01*randn()
    MMax_d_p_2sigma = exp(log(MMax_d_p) + 2*smooth_max)

    alpha = 0.7 + 0.1*randn()
    beta = 0.1*randn()
    gamma = 3 + 0.1*randn()

    return {
        'H_p': Hp,
        'Om': Om,
        'w0': w,
        'MMax_d_p': MMax_d_p,
        'MMax_d_p_2sigma': MMax_d_p_2sigma,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'xs': randn(nobs, 3)
    }

f = m.sampling(data=d, iter=2*args.iter, thin=args.thin, init=init)
fit = az.convert_to_inference_data(f)

print(f)

# Now that we're done with sampling, let's draw some pretty lines.
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
         ('gamma', {}, true_params['gamma']),
         ('neff_det', {}, 4*nobs))

az.plot_trace(fit, var_names=['H0', 'Om', 'w0', 'R0_30', 'MMax', 'smooth_max', 'alpha', 'beta', 'gamma', 'neff_det'], lines=lines)
savefig(args.tracefile)

az.to_netcdf(fit, args.chainfile)
