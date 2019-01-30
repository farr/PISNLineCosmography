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
from corner import corner
import h5py
import pystan
from scipy.stats import gaussian_kde
import sys
from tqdm import tqdm

p = ArgumentParser()

post = p.add_argument_group('Event Options')
post.add_argument('--sampfile', metavar='FILE.h5', default='observations.h5', help='posterior samples file (default: %(default)s)')
post.add_argument('--subset', metavar='DESIGNATOR', help='name of the attribute giving the number of detection to analyze (default: all)')
post.add_argument('--event-begin', metavar='N', type=int, help='beginning of range of event indices to analyze')
post.add_argument('--event-end', metavar='N', type=int, help='end of range of event indices to analyze (not inclusive)')
post.add_argument('--livetime', metavar='T', type=float, help='live time of event range')
post.add_argument('--nsamp', metavar='FILE', default='nsamp.txt', help='file with number of samples for each event (default: %(default)s)')

sel = p.add_argument_group('Selection Function Options')
sel.add_argument('--selfile', metavar='FILE.h5', default='selected.h5', help='file containing records of successful injections for VT estimation (default: %(default)s)')
sel.add_argument('--nsel', metavar='N', type=int, help='number of selected systems to include (default: all)')

copt = p.add_argument_group('Cosmology Options')
copt.add_argument('--cosmo-prior', action='store_true', help='use a prior on H0 and Om*h^2 from SNe and CMB (default: broad priors)')

samp = p.add_argument_group('Sampling Options')
samp.add_argument('--iter', metavar='N', type=int, default=1000, help='number of sampling iterations (equal amount of tuning; default: %(default)s)')

oop = p.add_argument_group('Output Options')
oop.add_argument('--chainfile', metavar='F', default='population.h5', help='output file (default: %(default)s)')
oop.add_argument('--tracefile', metavar='F', default='traceplot.pdf', help='traceplot file (default: %(default)s)')

args = p.parse_args()

# Check consistency among event selection options:
if args.subset is not None:
    if args.event_begin is not None or args.event_end is not None:
        raise ValueError('--subset and --event-begin or --event-end are mutually exclusive')
    if args.livetime is not None:
        raise ValueError('--subset and --livetime are mutually exclusive')
if args.event_begin is not None or args.event_end is not None:
    if args.livetime is None:
        raise ValueError('require --livetime with --event-begin or --event-end')

print('Called with the following command line:')
print(' '.join(sys.argv))

chain = {}
with h5py.File(args.sampfile, 'r') as inp:
    for n in ['m1det', 'm2det', 'dl', 'log_m1m2dl_wt']:
        chain[n] = array(inp['posteriors'][n])
    if args.subset is not None:
        nn, Tobs = inp.attrs[args.subset]
        nn = int(round(nn))
        for k in ['m1det', 'm2det', 'dl', 'log_m1m2dl_wt']:
            chain[k] = chain[k][:nn,:]
    else:
        if args.event_begin is not None:
            istart = args.event_begin
        else:
            istart = 0

        if args.event_end is not None:
            iend = args.event_end
        else:
            iend = chain['m1det'].shape[0]

        for k in ['m1det', 'm2det', 'dl', 'log_m1m2dl_wt']:
            chain[k] = chain[k][istart:iend, :]

        if args.livetime is not None:
            Tobs = args.livetime
        else:
            Tobs = inp.attrs['Tobs']

nsamp = np.round(loadtxt(args.nsamp)).astype(np.int)
if args.subset is None:
    nsamp = nsamp[istart:iend]
nsamp_total = np.sum(nsamp)

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
log_m1m2dl_wt = []

for i in range(nobs):
    npost = chain['m1det'].shape[1]

    if nsamp[i] > npost:
        # Then we need to draw some extra points.  First we will weight by the
        # inverse prior so that we are sampling the likelihood; then we will
        # draw from a KDE
        logwt = -chain['log_m1m2dl_wt'][i,:]

        rs = (np.max(logwt) + log(rand(npost))) < logwt

        # Cut out any samples with m2_det < 3 (which will be outside our bounds anyway).
        rs = rs & (chain['m2det'][i,:] > 3)

        print('Drawing event {:d} from a KDE; using {:d} points after prior re-weighting'.format(i, count_nonzero(rs)))

        # We draw from a KDE, using a variable transformation to ensure that 0 <
        # m2 < m1 and 0 < dl.  Because we have re-weighted the points,
        # they are drawn from a flat prior in m1obs, m2obs, dl, so we can
        # transform them at will as long as we bring them back to this space in
        # the end.
        p = row_stack((log(chain['m1det'][i,rs]),
                       log(chain['m2det'][i,rs]) - log(chain['m1det'][i,rs]-chain['m2det'][i,rs]),
                       log(chain['dl'][i,rs])))

        kde = gaussian_kde(p)
        pts = kde.resample(nsamp[i])

        ms1 = exp(pts[0,:])
        m1.append(ms1)

        gz = pts[1,:] > 0
        m2.append(where(gz,
                        (ms1)/(exp(-pts[1,:]) + 1),
                        (exp(pts[1,:])*ms1)/(1 + exp(pts[1,:]))))
        dl.append(exp(pts[2,:]))
        log_m1m2dl_wt.append(zeros_like(pts[0,:]))
    else:
        s = np.random.choice(chain['m1det'].shape[1], nsamp[i], replace=False)
        m1.append(chain['m1det'][i,s])
        m2.append(chain['m2det'][i,s])
        dl.append(chain['dl'][i,s])
        log_m1m2dl_wt.append(chain['log_m1m2dl_wt'][i,s])

ninterp = 500
zMax = 10
zinterp = expm1(linspace(log(1), log(zMax+1), ninterp))

msnorm = exp(arange(log(1), log(300), 0.01))
nnorm = len(msnorm)

z_p = 0.75

m = pystan.StanModel(file='model.stan')
d = {
    'nobs': nobs,
    'nsel': ndet,
    'ninterp': ninterp,
    'nnorm': nnorm,

    'nsamp': nsamp,
    'nsamp_total': nsamp_total,

    'Tobs': Tobs,
    'N_gen': N_gen,

    'm1obs': np.concatenate(m1),
    'm2obs': np.concatenate(m2),
    'dlobs': np.concatenate(dl),
    'log_m1m2dl_wt': np.concatenate(log_m1m2dl_wt),

    'm1sel': m1s_det,
    'm2sel': m2s_det,
    'dlsel': dls_det,
    'wtsel': wts_det,

    'zinterp': zinterp,

    'ms_norm': msnorm,

    'cosmo_prior': 1 if args.cosmo_prior else 0
}

def init(chain=None):
    if args.cosmo_prior:
        H0 = Planck15.H0.to(u.km/u.s/u.Mpc).value*(1.0 + 0.01*randn())
        Om = Planck15.Om0*(1.0 + 0.01*randn())
    else:
        H0 = np.random.uniform(low=60, high=80)
        Om = np.random.uniform(low=0.2, high=0.4)
    w = np.random.uniform(low=-1.5, high=-0.5)

    MMin = np.random.uniform(low=4, high=6)
    MMax = np.random.uniform(low=35, high=45)

    alpha = np.random.uniform(low=0.5, high=1.0)
    beta = np.random.uniform(low=-0.5, high=0.5)
    gamma = np.random.uniform(low=2, high=4)

    MLow2Sigma = np.random.uniform(low=2, high=4)
    MHigh2Sigma = np.random.uniform(low=45, high=55)

    return {
        'H0': H0,
        'Om': Om,
        'w': w,
        'MMin': MMin,
        'MMax': MMax,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'MLow2Sigma': MLow2Sigma,
        'MHigh2Sigma': MHigh2Sigma
    }

f = m.sampling(data=d, iter=2*args.iter, init=init, control={'metric': 'dense_e'})
fit = f.extract(permuted=True)

print(f)

# Now that we're done with sampling, let's draw some pretty lines.
from true_params import true_params
lines = (('H0', {}, true_params['H0']),
         ('Om', {}, true_params['Om']),
         ('w', {}, true_params['w']),
         ('R0', {}, true_params['R0']),
         ('MMin', {}, true_params['MMin']),
         ('MMax', {}, true_params['MMax']),
         ('alpha', {}, true_params['alpha']),
         ('beta', {}, true_params['beta']),
         ('gamma', {}, true_params['gamma']),
         ('sigma_low', {}, true_params['sigma_low']),
         ('sigma_high', {}, true_params['sigma_high']),
         ('neff_det', {}, 4*nobs))

az.plot_trace(f, var_names=['H0', 'Om', 'w', 'R0', 'MMin', 'MMax', 'sigma_low', 'sigma_high', 'alpha', 'beta', 'gamma', 'neff_det'], lines=lines)
savefig(args.tracefile)

with h5py.File(args.chainfile, 'w') as out:
    out.attrs['nobs'] = nobs
    out.attrs['nsel'] = ndet
    out.attrs['nsamp'] = nsamp

    for n in ['H0', 'Om', 'w', 'R0', 'MMin', 'MMax', 'sigma_low', 'sigma_high', 'alpha', 'beta', 'gamma', 'mu_det', 'neff_det', 'm1', 'm2', 'dl', 'z', 'neff']:
        out.create_dataset(n, data=fit[n], compression='gzip', shuffle=True)
