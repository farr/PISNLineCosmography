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

pr = p.add_argument_group('Prior Options')
pr.add_argument('--H0-mean', metavar='H0', default=70.0, type=float, help='Prior mean for H0 (default: %(default)s)')
pr.add_argument('--H0-sd', metavar='dH0', default=15.0, type=float, help='Prior s.d. for H0 (default %(default)s)')

alg = p.add_argument_group('Algorithm Options')
alg.add_argument('--ninterp', metavar='N', type=int, default=500, help='number of interpolated points for cosmology functions (default: %(default)s)')

samp = p.add_argument_group('Sampling Options')
samp.add_argument('--stanfile', metavar='FILE.stan', default='PISNLineCosmography.stan', help='file containing STAN code (default: %(default)s)')
samp.add_argument('--iter', metavar='N', type=int, default=2000, help='number of iterations, half to tuning (default: %(default)s)')
samp.add_argument('--thin', metavar='N', type=int, default=1, help='steps between recorded iterations (default: %(default)s)')

oop = p.add_argument_group('Output Options')
oop.add_argument('--chainfile', metavar='F', default='population.h5', help='output file (default: %(default)s)')
oop.add_argument('--tracefile', metavar='F', default='traceplot.pdf', help='traceplot file (default: %(default)s)')

args = p.parse_args()

def initializer(m1s, m2s, dls):
    def init():
        inds = randint(m1s.shape[1], size=m1s.shape[0])

        # Choose random observed masses and distances
        m1_obs = array([m1s[i,inds[i]] for i in range(m1s.shape[0])])
        m2_obs = array([m2s[i,inds[i]] for i in range(m2s.shape[0])])
        dl_obs = array([dls[i,inds[i]] for i in range(dls.shape[0])])

        # Choose a random H0
        H0 = 70.0 + 5.0*randn()

        c = cosmo.FlatLambdaCDM(H0, Planck15.Om0)

        zs = array([cosmo.z_at_value(c.luminosity_distance, d*u.Gpc) for d in dl_obs])
        m1 = m1_obs / (1+zs)
        m2 = m2_obs / (1+zs)

        MMax = np.max(m1) + rand()
        MMin = np.min(m2) - rand()

        if MMax < 30 or MMax > 100:
            return init() # Try again

        if MMin < 3 or MMin > 10:
            return init() # Try again

        R = 100.0 + 10*randn()

        alpha = -1.0 + randn()
        beta = 0.0 + randn()
        gamma = 3.0 + randn()

        return {
            'r': R/100.0,
            'h': H0/100.0,
            'MMin': MMin,
            'MMax': MMax,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'm1s_true': m1,
            'm2s_frac': (m2-MMin)/(m1-MMin),
            'dls_true': dl_obs
        }

    return init

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

m = pystan.StanModel(file=args.stanfile)

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
    'Tobs': Tobs,

    'ninterp': args.ninterp,

    'smooth_low': 0.05, # This should be physically irrelevant scale
    'smooth_high': 0.4, # Likewise, physically irrelevant scale

    'dl_max': dLmax,

    'H0_mean': args.H0_mean,
    'H0_sd': args.H0_sd
}

f = m.sampling(data=data, init=initializer(m1, m2, dl), iter=args.iter, thin=args.thin)
t = f.extract(permuted=True)

print(f) # Summary of sampling.

f.plot(['H0', 'R0', 'MMax', 'MMin', 'alpha', 'beta', 'gamma'])
savefig(args.tracefile)


with h5py.File(args.chainfile, 'w') as out:
    out.attrs['nsamp'] = nsamp

    for n in ['H0', 'R0', 'MMax', 'MMin', 'alpha', 'beta', 'gamma', 'm1s_true', 'm2s_true', 'dls_true', 'zs_true']:
        out.create_dataset(n, data=t[n], compression='gzip', shuffle=True)
