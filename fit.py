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
import pystan
import sys

p = ArgumentParser()

post = p.add_argument_group('Event Options')
post.add_argument('--sampfile', metavar='FILE.h5', default='parameters.h5', help='posterior samples file (default: %(default)s)')
post.add_argument('--samp', metavar='N', type=int, default=100, help='number of posterior samples used (default: %(default)s)')

sel = p.add_argument_group('Selection Function Options')
sel.add_argument('--selfile', metavar='FILE.h5', default='selected.h5', help='file containing records of successful injections for VT estimation (default: %(default)s)')
sel.add_argument('--frac', metavar='F', type=float, default=1.0, help='fraction of database to use for selection (default: %(default)s)')
sel.add_argument('--smooth-low', metavar='dM', type=float, default=0.6, help='smoothing mass scale at low-mass cutoff (default: %(default)s)')
sel.add_argument('--smooth-high', metavar='dM', type=float, default=0.4, help='smoothing mass scale at high-mass cutoff (default: %(default)s)')

samp = p.add_argument_group('Sampling Options')
samp.add_argument('--stanfile', metavar='FILE.stan', default='PISNLineCosmography.stan', help='stan file (default: %(default)s)')
samp.add_argument('--iter', metavar='N', type=int, default=1000, help='number of sampling iterations (equal amount of tuning; default: %(default)s)')
samp.add_argument('--thin', metavar='N', type=int, default=1, help='steps between recorded samples (default: %(default)s)')

oop = p.add_argument_group('Output Options')
oop.add_argument('--chainfile', metavar='F', default='population.h5', help='output file (default: %(default)s)')
oop.add_argument('--tracefile', metavar='F', default='traceplot.pdf', help='traceplot file (default: %(default)s)')

args = p.parse_args()

print('Called with the following command line:')
print(' '.join(sys.argv))

MMin = 5
MMax = 40

chain = {}
with h5py.File(args.sampfile, 'r') as inp:
    for n in ['m1s', 'm2s', 'dLs']:
        chain[n] = array(inp[n])
    Tobs = inp.attrs['Tobs']

with h5py.File(args.selfile, 'r') as inp:
    MObsMin = inp.attrs['MObsMin']
    MObsMax = inp.attrs['MObsMax']
    dLmax = inp.attrs['dLMax']
    N_gen = inp.attrs['NGen']

    m1s_det = array(inp['m1'])
    m2s_det = array(inp['m2'])
    dls_det = array(inp['dl'])
    wts_det = array(inp['wt'])

n = int(round(args.frac*len(m1s_det)))
N_gen = int(round(args.frac*N_gen))

m1s_det = m1s_det[:n]
m2s_det = m2s_det[:n]
dls_det = dls_det[:n]
wts_det = wts_det[:n]

ndet = m1s_det.shape[0]

nobs = chain['m1s'].shape[0]
nsamp = args.samp

m1 = zeros((nobs, nsamp))
m2 = zeros((nobs, nsamp))
dl = zeros((nobs, nsamp))

for i in range(nobs):
    inds = np.random.choice(chain['m1s'].shape[1], replace=False, size=nsamp)
    m1[i,:] = chain['m1s'][i,inds]
    m2[i,:] = chain['m2s'][i,inds]
    dl[i,:] = chain['dLs'][i,inds]

model = pystan.StanModel(file=args.stanfile)

ndet = m1s_det.shape[0]
data = {
    'nobs': nobs,
    'nsamp': nsamp,
    'ndet': ndet,
    'ninterp': 500,
    'm1obs': m1,
    'm2obs': m2,
    'dlobs': dl,
    'm1obs_det': m1s_det,
    'm2obs_det': m2s_det,
    'dlobs_det': dls_det,
    'wts_det': wts_det,
    'Tobs': Tobs,
    'dLMax': dLmax,
    'Ngen': N_gen,
    'smooth_low': args.smooth_low,
    'smooth_high': args.smooth_high
}

fit = model.sampling(data=data, iter=2*args.iter, thin=args.thin, chains=4, n_jobs=4)

print(fit)

fit.plot(['H0', 'R0', 'MMin', 'MMax', 'alpha', 'beta', 'gamma', 'Nex'])
savefig(args.tracefile)

t = fit.extract(permuted=True)

with h5py.File(args.chainfile, 'w') as out:
    out.attrs['nsamp'] = nsamp

    for n in ['H0', 'R0', 'MMax', 'MMin', 'alpha', 'beta', 'gamma', 'm1_true', 'm2_true', 'dl_true', 'z_true', 'Nex', 'sigma_rel_Nex']:
        out.create_dataset(n, data=t[n], compression='gzip', shuffle=True)
