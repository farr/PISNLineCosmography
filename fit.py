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
import pymc3 as pm
import sys

p = ArgumentParser()

post = p.add_argument_group('Event Options')
post.add_argument('--sampfile', metavar='FILE.h5', default='observations.h5', help='posterior samples file (default: %(default)s)')
post.add_argument('--subset', metavar='DESIGNATOR', help='name of the attribute giving the number of detection to analyze (default: all)')
post.add_argument('--samp', metavar='N', type=int, default=100, help='number of posterior samples used for each event (default: %(default)s)')
post.add_argument('--zero-uncert', action='store_true', help='treat the observations as having no uncertainty (default: %(default)s)')

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
    for n in ['m1det', 'm2det', 'dl']:
        chain[n] = array(inp['posteriors'][n])
    if args.subset is not None:
        nn, Tobs = inp.attrs[args.subset]
        nn = int(round(nn))
        for k in chain.keys():
            chain[k] = chain[k][:nn,:]
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
nsamp = args.samp

m1 = zeros((nobs, nsamp))
m2 = zeros((nobs, nsamp))
dl = zeros((nobs, nsamp))

for i in range(nobs):
    inds = np.random.choice(chain['m1det'].shape[1], replace=False, size=nsamp)
    m1[i,:] = chain['m1det'][i,inds]
    m2[i,:] = chain['m2det'][i,inds]
    dl[i,:] = chain['dl'][i,inds]

if args.zero_uncert:
    # If we have zero uncertainty, then we throw away all the samples, and just use the true values
    with h5py.File(args.sampfile, 'r') as f:
        m1 = reshape(array(f['m1s']), (-1, 1))
        m2 = reshape(array(f['m2s']), (-1, 1))
        z = reshape(array(f['zs']), (-1, 1))

        m1 = m1*(1+z)
        m2 = m2*(1+z)

        dl = Planck15.luminosity_distance(z).to(u.Gpc).value

        m1 = m1[:nobs, :]
        m2 = m2[:nobs, :]
        dl = dl[:nobs, :]

m = model.make_model(m1, m2, dl, m1s_det, m2s_det, dls_det, wts_det, N_gen, Tobs, cosmo_constraints=args.cosmo_constraints)

fit = model.sample(m, args.iter, args.iter, args.njobs)

print(pm.summary(fit))

print('Just completed sampling.')
print('  Fraction of D(ln(pi)) due to selection Monte-Carlo is {:.2f}'.format(std(nobs**2/(2*fit['neff_det'])) / (nobs*std(log(fit['Nex'])-log(fit['R0'])))))
print('  Mean fractional bias in R is {:.2f}'.format(mean(nobs/fit['neff_det'])))
print('  Mean fractional increase in sigma_R is {:.2f}'.format(mean((1 - 4*nobs + 3*nobs**2)/(2*fit['neff_det']*(nobs-1)))))

pm.traceplot(fit)
savefig(args.tracefile)

with h5py.File(args.chainfile, 'w') as out:
    out.attrs['nobs'] = nobs
    out.attrs['nsamp'] = nsamp
    out.attrs['nsel'] = ndet

    for n in ['H0', 'Om', 'w', 'R0', 'MMin', 'MMax', 'sigma_low', 'sigma_high', 'alpha', 'beta', 'gamma', 'Nex', 'neff_det']:
        out.create_dataset(n, data=fit[n], compression='gzip', shuffle=True)
