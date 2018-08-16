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

p = ArgumentParser()

post = p.add_argument_group('Event Options')
post.add_argument('--samp', metavar='N', type=int, default=100, help='number of posterior samples used (default: %(default)s)')
post.add_argument('--five-years', action='store_true', help='analyse five years of data (default is 1)')

sel = p.add_argument_group('Selection Function Options')
sel.add_argument('--frac', metavar='F', type=float, default=1.0, help='fraction of database to use for selection (default: %(default)s)')

alg = p.add_argument_group('Algorithm Options')
alg.add_argument('--ninterp', metavar='N', type=int, default=500, help='number of interpolated points for cosmology functions (default: %(default)s)')
alg.add_argument('--smooth-low', metavar='dM', type=float, default=0.07, help='low-mass smoothing scale for selection f\'cn (default: %(default)s)')
alg.add_argument('--smooth-high', metavar='dM', type=float, default=0.16, help='high-mass smoothing scale for selection f\'cn (default: %(default)s)')

samp = p.add_argument_group('Sampling Options')
samp.add_argument('--iter', metavar='N', type=int, default=2000, help='number of iterations, half to tuning (default: %(default)s)')
samp.add_argument('--thin', metavar='N', type=int, default=1, help='steps between recorded iterations (default: %(default)s)')

oop = p.add_argument_group('Output Options')
oop.add_argument('--chainfile', metavar='F', help='output file (default: population_{1yr,5yr}_NNNN.h5)')
oop.add_argument('--tracefile', metavar='F', help='traceplot file (default: traceplot_{1yr,5yr}_NNNN.pdf)')

args = p.parse_args()

MMin = 5
MMax = 40

with h5py.File('observations.h5', 'r') as inp:
    N_1yr = inp.attrs['N_1yr']

    m1s = array(inp['truth']['m1'])
    m2s = array(inp['truth']['m2'])
    zs = array(inp['truth']['z'])
    dls = array(inp['truth']['dl'])
    thetas = array(inp['truth']['theta'])

    mc_obs = array(inp['observations']['mc'])
    eta_obs = array(inp['observations']['eta'])
    A_obs = array(inp['observations']['A'])
    theta_obs = array(inp['observations']['theta'])
    sigma_mc = array(inp['observations']['sigma_mc'])
    sigma_eta = array(inp['observations']['sigma_eta'])
    sigma_theta = array(inp['observations']['sigma_theta'])

if args.five_years:
    N_evt = len(m1s)
else:
    N_evt = N_1yr

chain = {}
with h5py.File('parameters.h5', 'r') as inp:
    for n in ['m1s', 'm2s', 'mcs', 'etas', 'dLs', 'opt_snrs', 'thetas']:
        chain[n] = array(inp[n])[:N_evt, :]

with h5py.File('selected.h5', 'r') as inp:
    MObsMin = inp.attrs['MObsMin']
    MObsMax = inp.attrs['MObsMax']
    dLmax = inp.attrs['dLMax']
    N_gen = inp.attrs['NGen']

    m1s_det = array(inp['m1'])
    m2s_det = array(inp['m2'])
    dls_det = array(inp['dl'])

# Area of the triangle from [MObsMin,MObsMin] to [MObsMax,MObsMax]
# Times dLmax for dL volume.
Vgen = 0.5*(MObsMax-MObsMin)*(MObsMax-MObsMin)*dLmax

n = int(round(args.frac*len(m1s_det)))
N_gen = int(round(args.frac*N_gen))

m1s_det = m1s_det[:n]
m2s_det = m2s_det[:n]
dls_det = dls_det[:n]

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

m = pystan.StanModel(file='PISNLineCosmography.stan')

bws = []
for i in range(m1.shape[0]):
    bws.append(cov(row_stack((m1[i,:], m2[i,:], dl[i,:])))/nsamp**(2.0/7.0))

data = {
    'nobs': m1.shape[0],
    'nsamp': m1.shape[1],
    'nsel': m1s_det.shape[0],

    'm1s_m2s_dls': concatenate((m1[:,:,newaxis], m2[:,:,newaxis], dl[:,:,newaxis]), 2),
    'bws': bws,

    'm1s_sel': m1s_det,
    'm2s_sel': m2s_det,
    'dls_sel': dls_det,

    'Vgen': Vgen,
    'ngen': N_gen,

    'ninterp': args.ninterp,

    'smooth_low': args.smooth_low,
    'smooth_high': args.smooth_high,

    'dl_max': dLmax
}

f = m.sampling(data=data, iter=args.iter, thin=args.thin)
t = f.extract(permuted=True)

print(f) # Summary of sampling.

if args.tracefile is not None:
    fname = args.tracefile
elif args.five_years:
    fname = 'traceplot_5yr_{:04d}.pdf'.format(nsamp)
else:
    fname = 'traceplot_1yr_{:04d}.pdf'.format(nsamp)
f.plot(['H0', 'R0', 'MMax', 'MMin', 'alpha', 'beta', 'gamma'])
savefig(fname)

if args.chainfile is not None:
    fname = args.chainfile
elif args.five_years:
    fname = 'population_5yr_{:04d}.h5'.format(nsamp)
else:
    fname = 'population_1yr_{:04d}.h5'.format(nsamp)

with h5py.File(fname, 'w') as out:
    out.attrs['nsamp'] = nsamp

    for n in ['H0', 'R0', 'MMax', 'MMin', 'alpha', 'beta', 'gamma', 'm1s_true', 'm2s_true', 'dls_true', 'zs_true']:
        out.create_dataset(n, data=t[n], compression='gzip', shuffle=True)
