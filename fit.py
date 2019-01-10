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
import h5py
import pystan
from sklearn.mixture import GaussianMixture
import sys
from tqdm import tqdm

p = ArgumentParser()

post = p.add_argument_group('Event Options')
post.add_argument('--sampfile', metavar='FILE.h5', default='observations.h5', help='posterior samples file (default: %(default)s)')
post.add_argument('--subset', metavar='DESIGNATOR', help='name of the attribute giving the number of detection to analyze (default: all)')
post.add_argument('--event-begin', metavar='N', type=int, help='beginning of range of event indices to analyze')
post.add_argument('--event-end', metavar='N', type=int, help='end of range of event indices to analyze (not inclusive)')
post.add_argument('--livetime', metavar='T', type=float, help='live time of event range')
post.add_argument('--ngmm', metavar='N', default=6, type=int, help='number of components in GMM likelihood (default: %(default)s)')

sel = p.add_argument_group('Selection Function Options')
sel.add_argument('--selfile', metavar='FILE.h5', default='selected.h5', help='file containing records of successful injections for VT estimation (default: %(default)s)')
sel.add_argument('--nsel', metavar='N', type=int, help='number of selected systems to include (default: all)')

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

print('Running on {:d} events with {:d} posterior samples per event'.format(chain['m1det'].shape[0], chain['m1det'].shape[1]))

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
nsamp = chain['m1det'].shape[1]
m1 = chain['m1det']
m2 = chain['m2det']
dl = chain['dl']

ninterp = 500
zMax = 10
zinterp = expm1(linspace(log(1), log(zMax+1), ninterp))

msnorm = exp(arange(log(1), log(300), 0.01))
nnorm = len(msnorm)

print('Fitting GMMs to likelihood samples.')
gmm_wts = []
gmm_means = []
gmm_covs = []
gmm = GaussianMixture(args.ngmm)
for i in tqdm(range(nobs)):
    pts = column_stack((m1[i,:], m2[i,:], dl[i,:]))
    gmm.fit(pts)

    gmm_wts.append(gmm.weights_)
    gmm_means.append(gmm.means_)
    gmm_covs.append(gmm.covariances_)

m = pystan.StanModel(file='model.stan')
d = {
    'nobs': nobs,
    'nsel': ndet,
    'ninterp': ninterp,
    'nnorm': nnorm,

    'nsamp': nsamp,

    'ngmm': args.ngmm,

    'Tobs': Tobs,
    'N_gen': N_gen,

    'm1obs': m1,
    'm2obs': m2,
    'dlobs': dl,

    'gmm_wts': gmm_wts,
    'gmm_means': gmm_means,
    'gmm_cov': gmm_covs,

    'm1sel': m1s_det,
    'm2sel': m2s_det,
    'dlsel': dls_det,
    'wtsel': wts_det,

    'zinterp': zinterp,

    'ms_norm': msnorm,

    'MLow': 1.0,
    'MHigh': 300.0,
    'dLmax': Planck15.luminosity_distance(4).to(u.Gpc).value
}

f = m.sampling(data=d, iter=2*args.iter)
fit = f.extract(permuted=True)

print(f)

az.plot_trace(f, var_names=['H0', 'Om', 'w', 'R0', 'MMin', 'MMax', 'sigma_low', 'sigma_high', 'alpha', 'beta', 'gamma'])
savefig(args.tracefile)

with h5py.File(args.chainfile, 'w') as out:
    out.attrs['nobs'] = nobs
    out.attrs['nsel'] = ndet
    out.attrs['nsamp'] = nsamp

    for n in ['H0', 'Om', 'w', 'R0', 'MMin', 'MMax', 'sigma_low', 'sigma_high', 'alpha', 'beta', 'gamma', 'mu_det', 'neff_det', 'm1', 'm2', 'dl', 'z']:
        out.create_dataset(n, data=fit[n], compression='gzip', shuffle=True)
