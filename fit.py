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

p = ArgumentParser()

post = p.add_argument_group('Event Options')
post.add_argument('--sampfile', metavar='FILE.h5', default='observations.h5', help='posterior samples file (default: %(default)s)')
post.add_argument('--subset', metavar='DESIGNATOR', help='name of the attribute giving the number of detection to analyze (default: all)')
post.add_argument('--ncomp', metavar='N', type=int, default=4, help='number of Gaussian components in mixture model for event likelihood (default: %(default)s)')

sel = p.add_argument_group('Selection Function Options')
sel.add_argument('--selfile', metavar='FILE.h5', default='selected.h5', help='file containing records of successful injections for VT estimation (default: %(default)s)')
sel.add_argument('--nsel', metavar='N', type=int, help='number of selected systems to include (default: all)')

cos = p.add_argument_group('Cosmology Prior Options')
cos.add_argument('--cosmo-constraints', action='store_true', help='implement constraints from BNS H0 and Planck Om*h^2')

samp = p.add_argument_group('Sampling Options')
samp.add_argument('--iter', metavar='N', type=int, default=1000, help='number of sampling iterations (equal amount of tuning; default: %(default)s)')

oop = p.add_argument_group('Output Options')
oop.add_argument('--chainfile', metavar='F', default='population.h5', help='output file (default: %(default)s)')
oop.add_argument('--tracefile', metavar='F', default='traceplot.pdf', help='traceplot file (default: %(default)s)')

args = p.parse_args()

print('Called with the following command line:')
print(' '.join(sys.argv))

chain = {}
with h5py.File(args.sampfile, 'r') as inp:
    for n in ['m1det', 'm2det', 'dl', 'log_m1m2dl_prior']:
        chain[n] = array(inp['posteriors'][n])
    nsamp = array(inp['posteriors']['nsamp'], dtype=np.int)
    if args.subset is not None:
        nn, Tobs = inp.attrs[args.subset]
        nn = int(round(nn))
        for k in chain.keys():
            chain[k] = chain[k][:nn,:]
        nsamp = nsamp[:nn]
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

dl_max = 1.25*np.max(dls_det)

mus = []
covs = []
wts = []
nmin = np.inf
for i in range(nobs):
    # First, re-sample to likelihood
    w = exp(-(chain['log_m1m2dl_prior'][i,:] - np.min(chain['log_m1m2dl_prior'][i,:])))
    r = np.random.uniform(size=len(w))
    s = r < w
    nmin = min(nmin, count_nonzero(s))
    pts = column_stack((chain['m1det'][i,s], chain['m2det'][i,s], chain['dl'][i,s]))
    gmm = GaussianMixture(args.ncomp)
    gmm.fit(pts)

    mus.append(gmm.means_)
    covs.append(gmm.covariances_)
    wts.append(gmm.weights_)

print('Minimum number of samples used to fit GMM is {:d}'.format(nmin))

model = pystan.StanModel(file='model.stan')

ninterp = 500
zinterp = expm1(linspace(log(1), log(11), ninterp))

mnorm = exp(arange(log(1), log(200), 0.01))
nnorm = len(mnorm)

if args.cosmo_constraints:
    cosmo_flag = 1
else:
    cosmo_flag = 0

data = {
    'nobs': nobs,
    'nsel': ndet,
    'ninterp': ninterp,
    'nnorm': nnorm,
    'ncomp': args.ncomp,

    'Tobs': Tobs,
    'N_gen': N_gen,

    'mu': mus,
    'cov': covs,
    'wts': wts,

    'm1sel': m1s_det,
    'm2sel': m2s_det,
    'dlsel': dls_det,
    'wtsel': wts_det,

    'zinterp': zinterp,

    'ms_norm': mnorm,
    'dl_max': dl_max,

    'use_cosmo_prior': cosmo_flag,
    'mu_H0': Planck15.H0.to(u.km/u.s/u.Mpc).value,
    'sigma_H0': 0.01*Planck15.H0.to(u.km/u.s/u.Mpc).value,
    'mu_Omh2': 0.02225+0.1198,
    'sigma_Omh2': sqrt(0.00016**2 + 0.0015**2)
}

fit_obj = model.sampling(data=data, iter=2*args.iter)

print(fit_obj)

fit = fit_obj.extract(permuted=True)

print('Just completed sampling.')
print('  Fraction of D(ln(pi)) due to selection Monte-Carlo is {:.2f}'.format(std(nobs**2/(2*fit['neff_det'])) / (nobs*std(log(fit['Nex'])-log(fit['R0'])))))
print('  Mean fractional bias in R is {:.2f}'.format(mean(nobs/fit['neff_det'])))
print('  Mean fractional increase in sigma_R is {:.2f}'.format(mean((1 - 4*nobs + 3*nobs**2)/(2*fit['neff_det']*(nobs-1)))))

with h5py.File(args.chainfile, 'w') as out:
    out.attrs['nobs'] = nobs
    out.attrs['nsel'] = ndet

    out.create_dataset('nsamp', data=nsamp, compression='gzip', shuffle=True)
    for n in ['H0', 'Om', 'w', 'R0', 'MMin', 'MMax', 'sigma_low', 'sigma_high', 'alpha', 'beta', 'gamma', 'Nex', 'neff_det', 'm1', 'm2', 'dl', 'z']:
        out.create_dataset(n, data=fit[n], compression='gzip', shuffle=True)

az.plot_trace(fit_obj, var_names=['H0', 'Om', 'w', 'R0', 'alpha', 'beta', 'gamma', 'MMin', 'MMax', 'sigma_low', 'sigma_high'])
savefig(args.tracefile)
