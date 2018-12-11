#!/usr/bin/env python

# Set the backend to a non-displaying one.
import matplotlib
matplotlib.use('PDF')

from pylab import *

from argparse import ArgumentParser
import astropy.cosmology as cosmo
from astropy.cosmology import Planck15
import astropy.units as u
import h5py
import model
import pandas as pd
import pymc3 as pm
import theano
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
samp.add_argument('--njobs', metavar='N', type=int, default=4, help='number of chains/jobs to run (default: %(default)s)')

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
    chain['nsamp'] = array(inp['posteriors']['nsamp'], dtype=int)
    if args.subset is not None:
        nn, Tobs = inp.attrs[args.subset]
        nn = int(round(nn))
        for k in ['m1det', 'm2det', 'dl', 'log_m1m2dl_prior']:
            chain[k] = chain[k][:nn,:]
        chain['nsamp'] = chain['nsamp'][:nn]
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
log_prior = []

for i in range(nobs):
    inds = np.random.choice(chain['m1det'].shape[1], replace=False, size=chain['nsamp'][i])
    m1.append(chain['m1det'][i,inds])
    m2.append(chain['m2det'][i,inds])
    dl.append(chain['dl'][i,inds])
    log_prior.append(chain['log_m1m2dl_prior'][i,inds])
m1 = concatenate(m1)
m2 = concatenate(m2)
dl = concatenate(dl)
log_prior = concatenate(log_prior)

m = model.make_model(m1, m2, dl, log_prior, chain['nsamp'], m1s_det, m2s_det, dls_det, wts_det, N_gen, Tobs, cosmo_constraints=args.cosmo_constraints)

with m:
    fit = model.sample(m, args.iter, args.iter, args.njobs)

with pd.option_context('display.max_rows', 999, 'display.max_columns', 999):
    print(pm.summary(fit))

print('Just completed sampling.')
print('  Fraction of D(ln(pi)) due to selection Monte-Carlo is {:.2f}'.format(std(nobs**2/(2*fit['neff_det'])) / (nobs*std(log(fit['Nex'])-log(fit['R0'])))))
print('  Mean fractional bias in R is {:.2f}'.format(mean(nobs/fit['neff_det'])))
print('  Mean fractional increase in sigma_R is {:.2f}'.format(mean((1 - 4*nobs + 3*nobs**2)/(2*fit['neff_det']*(nobs-1)))))

pm.traceplot(fit)
savefig(args.tracefile)

with h5py.File(args.chainfile, 'w') as out:
    out.attrs['nobs'] = nobs
    out.attrs['nsel'] = ndet

    for n in ['H0', 'Om', 'w', 'R0', 'MMax', 'sigma_high', 'alpha', 'beta', 'gamma', 'Nex', 'neff_det']:
        out.create_dataset(n, data=fit[n], compression='gzip', shuffle=True)
