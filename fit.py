#!/usr/bin/env python

from pylab import *

from argparse import ArgumentParser
import h5py
import pystan

p = ArgumentParser()

post = p.add_argument_group('Event Options')
post.add_argument('--samp', metavar='N', type=int, default=100, help='number of posterior samples used (default: %(default)s)')
post.add_argument('--five-years', action='store_true', help='analyse five years of data (default is 1)')

sel = p.add_argument_group('Selection Function Options')
sel.add_argument('--frac', metavar='F', type=float, default=1.0, help='fraction of database to use for selection (default: %(default)s)')

samp = p.add_argument_group('Sampling Options')
samp.add_argument('--iter', metavar='N', type=int, default=2000, help='number of iterations (default: %(default)s)')
samp.add_argument('--thin', metavar='N', type=int, default=1, help='steps between saved iterations (default: %(default)s)')

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
    for n in ['m1s', 'm2s', 'mcs', 'etas', 'qs', 'dLs', 'opt_snrs', 'thetas']:
        chain[n] = array(inp[n])[:N_evt, :]

with h5py.File('selected.h5', 'r') as inp:
    MObsMin = inp.attrs['MObsMin']
    MObsMax = inp.attrs['MObsMax']
    dLmin = inp.attrs['dLMin']
    dLmax = inp.attrs['dLMax']
    N_gen = inp.attrs['NGen']

    m1s_det = array(inp['m1'])
    qs_det = array(inp['q'])
    dls_det = array(inp['dl'])

n = int(round(args.frac*len(m1s_det)))
N_gen = int(round(args.frac*N_gen))

m1s_det = m1s_det[:n]
qs_det = qs_det[:n]
dls_det = dls_det[:n]

model_pop = pystan.StanModel(file='PISNLineCosmography.stan')

nobs = chain['m1s'].shape[0]
nsamp = args.samp

m1 = zeros((nobs, nsamp))
dl = zeros((nobs, nsamp))

for i in range(nobs):
    m1[i,:] = np.random.choice(chain['m1s'][i,:], replace=False, size=nsamp)
    dl[i,:] = np.random.choice(chain['dLs'][i,:], replace=False, size=nsamp)

# It is important for the error checking of the ODE integrator that all the dl are unique, so we dither by 1 pc
dl = dl + 1e-9*randn(*dl.shape)

data_pop = {
    'nobs': nobs,
    'nsamp': nsamp,
    'm1s': m1,
    'dls': dl,

    'ndet': len(m1s_det),
    'ngen': N_gen,
    'Vgen': (MObsMax-MObsMin)*(log(dLmax) - log(dLmin)),
    'm1s_det': m1s_det,
    'dls_det': dls_det
}

fit_pop = model_pop.sampling(data=data_pop, iter=args.iter, thin=args.thin)

chain_pop = fit_pop.extract(permuted=True)

if args.five_years:
    fname = 'population_5yr_{:03d}.h5'.format(nsamp)
else:
    fname = 'population_1yr_{:03d}.h5'.format(nsamp)

with h5py.File(fname, 'w') as out:
    out.attrs['nsamp'] = nsamp

    for n in ['H0', 'R0', 'MMax', 'MMin', 'alpha', 'gamma']:
        out.create_dataset(n, data=chain_pop[n], compression='gzip', shuffle=True)

print(fit_pop)
