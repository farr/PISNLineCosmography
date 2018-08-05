#!/usr/bin/env python

# Set the backend to a non-displaying one.
import matplotlib
matplotlib.use('PDF')

from pylab import *

from argparse import ArgumentParser
import h5py
import PISNLineCosmography as plc
import pymc3 as pm

p = ArgumentParser()

post = p.add_argument_group('Event Options')
post.add_argument('--samp', metavar='N', type=int, default=100, help='number of posterior samples used (default: %(default)s)')
post.add_argument('--five-years', action='store_true', help='analyse five years of data (default is 1)')

sel = p.add_argument_group('Selection Function Options')
sel.add_argument('--frac', metavar='F', type=float, default=1.0, help='fraction of database to use for selection (default: %(default)s)')

samp = p.add_argument_group('Sampling Options')
samp.add_argument('--iter', metavar='N', type=int, default=1000, help='number of post-tune iterations (default: %(default)s)')

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
    for n in ['m1s', 'm2s', 'mcs', 'etas', 'qs', 'dLs', 'opt_snrs', 'thetas']:
        chain[n] = array(inp[n])[:N_evt, :]

with h5py.File('selected.h5', 'r') as inp:
    MObsMin = inp.attrs['MObsMin']
    MObsMax = inp.attrs['MObsMax']
    dLmax = inp.attrs['dLMax']
    N_gen = inp.attrs['NGen']

    m1s_det = array(inp['m1'])
    qs_det = array(inp['q'])
    dls_det = array(inp['dl'])

Vgen = (MObsMax - MObsMin)*dLmax

n = int(round(args.frac*len(m1s_det)))
N_gen = int(round(args.frac*N_gen))

m1s_det = m1s_det[:n]
qs_det = qs_det[:n]
dls_det = dls_det[:n]

ndet = m1s_det.shape[0]

nobs = chain['m1s'].shape[0]
nsamp = args.samp

m1 = zeros((nobs, nsamp))
dl = zeros((nobs, nsamp))

for i in range(nobs):
    m1[i,:] = np.random.choice(chain['m1s'][i,:], replace=False, size=nsamp)
    dl[i,:] = np.random.choice(chain['dLs'][i,:], replace=False, size=nsamp)

# Perturb the dls and dls_det by 1 kpc just so that they are unique
dl = dl + 1e-6*randn(*dl.shape)
dls_det = dls_det + 1e-6*randn(*dls_det.shape)

m = plc.make_model(m1, dl, m1s_det, dls_det, Vgen, N_gen)

with m:
    step_met = pm.Metropolis(vars=[m.MMin, m.MMax], S=array([[0.01, 0.0], [0.0, 0.16]]))
    step_hmc = pm.NUTS(vars=[m.R0, m.alpha, m.gamma, m.H0])

    t = pm.sample(draws=args.iter, tune=args.iter, step=[step_met, step_hmc], chains=4, cores=4)

print(pm.summary(t))

if args.tracefile is not None:
    fname = args.tracefile
elif args.five_years:
    fname = 'traceplot_5yr_{:04d}.pdf'.format(nsamp)
else:
    fname = 'traceplot_1yr_{:04d}.pdf'.format(nsamp)
pm.traceplot(t, varnames=['H0', 'R0', 'MMax', 'MMin', 'alpha', 'gamma'])
savefig(fname)

if args.chainfile is not None:
    fname = args.chainfile
elif args.five_years:
    fname = 'population_5yr_{:04d}.h5'.format(nsamp)
else:
    fname = 'population_1yr_{:04d}.h5'.format(nsamp)

with h5py.File(fname, 'w') as out:
    out.attrs['nsamp'] = nsamp

    for n in ['H0', 'R0', 'MMax', 'MMin', 'alpha', 'gamma']:
        out.create_dataset(n, data=t[n], compression='gzip', shuffle=True)
