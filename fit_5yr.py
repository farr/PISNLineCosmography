#!/usr/bin/env python

from pylab import *

import h5py
import pystan

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

chain = {}
with h5py.File('parameters.h5', 'r') as inp:
    for n in ['m1s', 'm2s', 'mcs', 'etas', 'qs', 'dLs', 'opt_snrs', 'thetas']:
        chain[n] = array(inp[n])

with h5py.File('selected.h5', 'r') as inp:
    MObsMin = inp.attrs['MObsMin']
    MObsMax = inp.attrs['MObsMax']
    dLmax = inp.attrs['dLMax']
    N_gen = inp.attrs['NGen']

    m1s_det = array(inp['m1'])
    qs_det = array(inp['q'])
    dls_det = array(inp['dl'])

model_pop = pystan.StanModel(file='PISNLineCosmography.stan')

nsamp = 100 # TODO: check convergence with this number of samples.
nobs = chain['m1s'].shape[0]

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
    'Vgen': (MObsMax-MObsMin)*dLmax,
    'm1s_det': m1s_det,
    'dls_det': dls_det
}

fit_pop = model_pop.sampling(data=data_pop)

chain_pop = fit_pop.extract(permuted=True)

with h5py.File('population.h5', 'w') as out:
    out.attrs['nsamp'] = nsamp

    for n in ['H0', 'R0', 'MMax', 'MMin', 'alpha', 'gamma']:
        out.create_dataset(n, data=chain_pop[n], compression='gzip', shuffle=True)

print(fit_pop)
