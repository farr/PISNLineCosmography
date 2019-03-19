#!/usr/bin/env python

# Set the backend to a non-displaying one.
import matplotlib
matplotlib.use('PDF')

from pylab import *

from argparse import ArgumentParser
import arviz as az
import astropy.cosmology as cosmo
import astropy.units as u
from corner import corner
import h5py
import pystan
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import sys
from tqdm import tqdm

p = ArgumentParser()

post = p.add_argument_group('Event Options')
post.add_argument('--sampfile', metavar='FILE.h5', default='observations.h5', help='posterior samples file (default: %(default)s)')
post.add_argument('--subset', metavar='DESIGNATOR', help='name of the attribute giving the number of detection to analyze (default: all)')
post.add_argument('--event-begin', metavar='N', type=int, help='beginning of range of event indices to analyze')
post.add_argument('--event-end', metavar='N', type=int, help='end of range of event indices to analyze (not inclusive)')
post.add_argument('--livetime', metavar='T', type=float, help='live time of event range')
post.add_argument('--nsamp', metavar='N', default=128, type=int, help='number of samples to use in KDE (default: %(default)s)')

sel = p.add_argument_group('Selection Function Options')
sel.add_argument('--selfile', metavar='FILE.h5', default='selected.h5', help='file containing records of successful injections for VT estimation (default: %(default)s)')
sel.add_argument('--nsel', metavar='N', type=int, help='number of selected systems to include (default: all)')

copt = p.add_argument_group('Cosmology Options')
copt.add_argument('--cosmo-prior', action='store_true', help='use a prior on H0 and Om*h^2 from SNe and CMB (default: broad priors)')

samp = p.add_argument_group('Sampling Options')
samp.add_argument('--iter', metavar='N', type=int, default=1000, help='number of sampling iterations (equal amount of tuning; default: %(default)s)')

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

m1 = []
m2 = []
dl = []
bws = []

for i in range(nobs):
    npost = chain['m1det'].shape[1]

    s = np.random.choice(chain['m1det'].shape[1], args.nsamp, replace=False)
    m1.append(chain['m1det'][i,s])
    m2.append(chain['m2det'][i,s])
    dl.append(chain['dl'][i,s])

    pts = column_stack((chain['m1det'][i,:], chain['m2det'][i,:], chain['dl'][i,:]))
    bws.append(cov(pts, rowvar=False) / args.nsamp**(2.0/7.0))

ninterp = 500
zMax = 10
zinterp = expm1(linspace(log(1), log(zMax+1), ninterp))

m = pystan.StanModel(file='model.stan')
d = {
    'nobs': nobs,
    'nsel': ndet,
    'nsamp': args.nsamp,

    'Tobs': Tobs,
    'N_gen': N_gen,

    'm1obs': m1,
    'm2obs': m2,
    'dlobs': dl,
    'bw': bws,

    'm1sel': m1s_det,
    'm2sel': m2s_det,
    'dlsel': dls_det,
    'log_wtsel': np.log(wts_det),

    'ninterp': ninterp,
    'zinterp': zinterp,

    'cosmo_prior': 1 if args.cosmo_prior else 0
}

def init(chain=None):
    H0 = 70 + 5*randn()
    Om = 0.3 + 0.05*randn()
    w = -1 + 0.1*randn()
    w_a = 0 + 0.1*randn()

    z = expm1(linspace(log(1), log(5+1), 1000))
    c = cosmo.w0waCDM(H0*u.km/u.s/u.Mpc, Om, 1-Om, w, w_a)
    d = c.luminosity_distance(z).to(u.Gpc).value

    z_of_d = interp1d(d, z)

    m1source = []
    m2source = []
    zsource = []
    dlsource = []

    for i in range(nobs):
        j = randint(m1[i].shape[0])
        dlsource.append(dl[i][j])
        zsource.append(z_of_d(dl[i][j]))
        m1source.append(m1[i][j]/(1+zsource[-1]))
        m2source.append(m2[i][j]/(1+zsource[-1]))

    m1source = array(m1source)
    m2source = array(m2source)
    zsource = array(zsource)
    dlsource = array(dlsource)

    # If any m2's come out less than 5, correct them.
    m2source[m2source < 5] = 5 + 0.1*rand(count_nonzero(m2source < 5))

    # Now that we've corrected m2sources, we might have m1 < m2, so fix that, too
    s = m1source < m2source
    m1source[s] = m2source[s] + 1*rand(count_nonzero(s))

    MMin = np.min(m2source) - 0.5*rand()
    MMax = np.max(m1source) + 2*rand()

    m2frac = (m2source - MMin)/(m1source - MMin)

    alpha = 0.75 + 0.2*randn()
    beta = 0.0 + 0.2*randn()
    gamma = 3 + 0.1*randn()

    return {
        'H0': H0,
        'Omh2': Om*(H0/100)**2,
        'w_p': w,
        'w_a': w_a,

        'MMin': MMin,
        'MMax': MMax,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,

        'm1s': m1source,
        'm2_frac': m2frac,
        'zs': zsource
    }

f = m.sampling(data=d, iter=2*args.iter, init=init)
fit = f.extract(permuted=True)

print(f)

# Now that we're done with sampling, let's draw some pretty lines.
from true_params import true_params
lines = (('H0', {}, true_params['H0']),
         ('Om', {}, true_params['Om']),
         ('w', {}, true_params['w']),
         ('w_p', {}, true_params['w']),
         ('w_a', {}, true_params['w_a']),
         ('R0_30', {}, true_params['R0_30']),
         ('MMin', {}, true_params['MMin']),
         ('MMax', {}, true_params['MMax']),
         ('sigma_min', {}, true_params['sigma_min']),
         ('sigma_max', {}, true_params['sigma_max']),
         ('alpha', {}, true_params['alpha']),
         ('beta', {}, true_params['beta']),
         ('gamma', {}, true_params['gamma']),
         ('neff_det', {}, 4*nobs))

az.plot_trace(f, var_names=['H0', 'Om', 'w', 'w_p', 'w_a', 'R0_30', 'MMin', 'MMax', 'sigma_min', 'sigma_max', 'alpha', 'beta', 'gamma', 'neff_det'], lines=lines)
savefig(args.tracefile)

with h5py.File(args.chainfile, 'w') as out:
    out.attrs['nobs'] = nobs
    out.attrs['nsel'] = ndet
    out.attrs['nsamp'] = args.nsamp

    for n in ['H0', 'Om', 'w', 'w_p', 'w_a', 'R0_30', 'MMin', 'MMax', 'sigma_min', 'sigma_max', 'alpha', 'beta', 'gamma', 'neff_det', 'm1s', 'm2s', 'dls', 'zs']:
        out.create_dataset(n, data=fit[n], compression='gzip', shuffle=True)
