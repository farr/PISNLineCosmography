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

c = p.add_argument_group('Cosmology Options')
c.add_argument('--prior', choices=['free', 'Planck-Om-w', 'H0', 'H0-Planck-Om'], default='free', help='cosmology priors: free; Om, w from Planck; 1%% H0; 1%% H0 plus Om from Planck (default: %(default)s)')

sel = p.add_argument_group('Selection Function Options')
sel.add_argument('--selfile', metavar='FILE.h5', default='selected.h5', help='file containing records of successful injections for VT estimation (default: %(default)s)')
sel.add_argument('--nsel', metavar='N', type=int, help='number of selected systems to include (default: all)')
sel.add_argument('--smooth-low', metavar='dM', type=float, default=0.1, help='smoothing mass scale at low-mass cutoff (default: %(default)s)')
sel.add_argument('--smooth-high', metavar='dM', type=float, default=0.5, help='smoothing mass scale at high-mass cutoff (default: %(default)s)')

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

# Set up the prior data
data_free = {
    'mu_H0': 70.0,
    'sigma_H0': 20.0,
    'mu_Om': 0.3,
    'sigma_Om': 0.1,
    'mu_wp': -1.0,
    'sigma_wp': 0.5,
    'mu_wa': 0.0,
    'sigma_wa': 2.0/3.0
}

data_Om_w_Planck = {
    'mu_H0': 70.0,
    'sigma_H0': 20.0,
    'mu_Om': 0.3089,
    'sigma_Om': 0.0062,
    'mu_wp': -1.019,
    'sigma_wp': 0.0775,
    'mu_wa': 0.0,
    'sigma_wa': 0.01
}

H0 = Planck15.H0.to(u.km/u.s/u.Mpc).value
data_H0_1pct = {
    'mu_H0': H0,
    'sigma_H0': 0.01*H0,
    'mu_Om': 0.3,
    'sigma_Om': 0.1,
    'mu_wp': -1.0,
    'sigma_wp': 0.5,
    'mu_wa': 0.0,
    'sigma_wa': 2.0/3.0
}

data_H0_1pct_Om = {
    'mu_H0': H0,
    'sigma_H0': 0.01*H0,
    'mu_Om': 0.3089,
    'sigma_Om': 0.0062,
    'mu_wp': -1.0,
    'sigma_wp': 0.5,
    'mu_wa': 0.0,
    'sigma_wa': 2.0/3.0
}

if args.prior == 'free':
    data_prior = data_free
elif args.prior == 'Planck-Om-w':
    data_prior = data_Om_w_Planck
elif args.prior == 'H0':
    data_prior = data_H0_1pct
elif args.prior == 'H0-Planck-Om':
    data_prior = data_H0_1pct_Om
else:
    raise ValueError('unrecognized cosmology prior option')

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

if args.nsel is not None:
    f = float(args.nsel)/float(len(m1s_det))

    s = np.random.choice(m1s_det.shape[0], replace=False, size=args.nsel)

    N_gen = int(round(f*N_gen))
    m1s_det = m1s_det[s]
    m2s_det = m2s_det[s]
    dls_det = dls_det[s]
    wts_det = wts_det[s]

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

bws = []
for i in range(nobs):
    bws.append(cov(column_stack((m1[i,:], m2[i,:], dl[i,:])), rowvar=False)/nsamp**(2.0/7.0))
m1_m2_dl = stack((m1, m2, dl), axis=2)

model = pystan.StanModel(file=args.stanfile)

ndet = m1s_det.shape[0]
data = {
    'nobs': nobs,
    'nsamp': nsamp,
    'ndet': ndet,
    'ninterp': 100,
    'm1obs_m2obs_dl': m1_m2_dl,
    'bw': bws,
    'm1obs_det': m1s_det,
    'm2obs_det': m2s_det,
    'dlobs_det': dls_det,
    'wts_det': wts_det,
    'Tobs': Tobs,
    'dLMax': dLmax,
    'Ngen': N_gen,
    'smooth_low': args.smooth_low,
    'smooth_high': args.smooth_high,
    'z_p': 0.5 # Pivot redshift from perfect mass fits
}
data.update(data_prior) # Add in the prior

def init(chain_id=0):
    m1_init = zeros(nobs)
    m2_init = zeros(nobs)
    m2_frac_init = zeros(nobs)
    dl_init = zeros(nobs)
    z_init = zeros(nobs)

    H0 = 70 + 5*randn()
    Om = 0.3 + 0.05*randn()
    w = -1.0 + 0.1*randn()
    wa = 0.1*randn()
    R0 = 100 + 10*randn()

    c = cosmo.FlatwCDM(H0=H0, Om0=Om, w0=w)

    j = randint(m1.shape[1])

    for i in range(nobs):
        dl_init[i] = dl[i,j]
        z_init[i] = cosmo.z_at_value(c.luminosity_distance, dl[i,j]*u.Gpc)
        m1_init[i] = m1[i,j]/(1+z_init[i])
        m2_init[i] = m2[i,j]/(1+z_init[i])

    MMin = np.min(m2_init) - 0.5*rand()
    MMax = np.max(m1_init) + 0.5*rand()

    for i in range(nobs):
        m2_frac_init[i] = (m2_init[i] - MMin)/(m1_init[i] - MMin)

    alpha = 0.75 + 0.1*randn()
    beta = 0.0 + 0.1*randn()
    gamma = 3 + 0.1*randn()

    return {
        'H0': H0,
        'Om': Om,
        'w_p': w,
        'w_a': wa,
        'R0': R0,
        'MMin': MMin,
        'MMax': MMax,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'm1_true': m1_init,
        'm2_frac': m2_frac_init,
        'dl_true': dl_init
    }

fit = model.sampling(data=data, iter=2*args.iter, thin=args.thin, chains=4, n_jobs=4, init=init)

print(fit)

fit.plot(['H0', 'Om', 'w_p', 'w_a', 'R0', 'MMin', 'MMax', 'alpha', 'beta', 'gamma'])
savefig(args.tracefile)

t = fit.extract(permuted=True)

with h5py.File(args.chainfile, 'w') as out:
    out.attrs['nsamp'] = nsamp

    for n in ['H0', 'Om', 'w0', 'w_p', 'w_a', 'R0', 'MMax', 'MMin', 'alpha', 'beta', 'gamma', 'dH', 'Nex', 'sigma_Nex', 'neff_det', 'm1_true', 'm2_true', 'dl_true', 'z_true']:
        out.create_dataset(n, data=t[n], compression='gzip', shuffle=True)
