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
import bz2
import h5py
import pickle
import pystan
import sys

p = ArgumentParser()

post = p.add_argument_group('Event Options')
post.add_argument('--sampfile', metavar='FILE.h5', default='observations.h5', help='posterior samples file (default: %(default)s)')
post.add_argument('--subset', metavar='DESIGNATOR', help='name of the attribute giving the number of detection to analyze (default: all)')

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
oop.add_argument('--fitfile', metavar='F', default='fit.pkl.bz2', help='pickled model and fit object (default: %(default)s)')

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

m1 = np.zeros((0,))
m2 = np.zeros((0,))
dl = np.zeros((0,))
logwt = np.zeros((0,))

for i in range(nobs):
    inds = np.random.choice(chain['m1det'].shape[1], replace=False, size=nsamp[i])
    m1 = np.concatenate((m1, chain['m1det'][i,inds]))
    m2 = np.concatenate((m2, chain['m2det'][i,inds]))
    dl = np.concatenate((dl, chain['dl'][i,inds]))
    logwt = np.concatenate((logwt, chain['log_m1m2dl_prior'][i,inds]))

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

    'nsamp': nsamp,
    'nsamp_total': np.sum(nsamp),

    'Tobs': Tobs,
    'N_gen': N_gen,

    'm1obs': m1,
    'm2obs': m2,
    'dlobs': dl,
    'log_samp_wts': logwt,

    'm1sel': m1s_det,
    'm2sel': m2s_det,
    'dlsel': dls_det,
    'wtsel': wts_det,

    'zinterp': zinterp,

    'ms_norm': mnorm,

    'use_cosmo_prior': cosmo_flag,
    'mu_H0': Planck15.H0.to(u.km/u.s/u.Mpc).value,
    'sigma_H0': 0.01*Planck15.H0.to(u.km/u.s/u.Mpc).value,
    'mu_Omh2': 0.02225+0.1198,
    'sigma_Omh2': sqrt(0.00016**2 + 0.0015**2)
}

fit_obj = model.sampling(data=data, iter=2*args.iter)

with bz2.BZ2File(args.fitfile, 'w') as f:
    pickle.dump(model, f)
    pickle.dump(fit_obj, f)

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
    for n in ['H0', 'Om', 'w', 'R0', 'MMin', 'MMax', 'sigma_low', 'sigma_high', 'alpha', 'beta', 'gamma', 'Nex', 'neff_det', 'neff', 'm1_source', 'm2_source', 'dl_source', 'z_source']:
        out.create_dataset(n, data=fit[n], compression='gzip', shuffle=True)

az.plot_trace(fit_obj, var_names=['H0', 'Om', 'w', 'R0', 'alpha', 'beta', 'gamma', 'MMin', 'MMax', 'sigma_low', 'sigma_high'])
savefig(args.tracefile)
