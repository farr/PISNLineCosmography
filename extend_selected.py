#!/usr/bin/env python

from pylab import *

from argparse import ArgumentParser
import h5py
from scipy.interpolate import interp1d, interp2d
from tqdm import tqdm

parser = ArgumentParser()

parser.add_argument('--selfile', metavar='FILE', default='selected.h5', help='file to add to (default: %(default)s)')
parser.add_argument('--number', metavar='N', default=1024, type=int, help='number of selected systems to add (default: %(default)s)')

args = parser.parse_args()

with h5py.File('optimal_snr.h5', 'r') as inp:
    ms = array(inp['ms'])
    OSNRS = array(inp['SNR'])
M1S, M2S = meshgrid(ms, ms, indexing='ij')

with h5py.File(args.selfile, 'r') as inp:
    MObsMin = inp.attrs['MObsMin']
    MObsMax = inp.attrs['MObsMax']
    dLmax = inp.attrs['dLMax']
    N_gen_file = inp.attrs['NGen']

    m1s_det_file = array(inp['m1'])
    m2s_det_file = array(inp['m2'])
    dls_det_file = array(inp['dl'])
    wts_det_file = array(inp['wt'])

with h5py.File('thetas.h5', 'r') as inp:
    thetas = array(inp['Theta'])

m1s_det = []
m2s_det = []
dls_det = []
wts_det = []
N_gen = 0
N_det = 0

ts = concatenate(([0.0], sort(thetas), [1.0]))
tcum = interp1d(linspace(0, 1, ts.shape[0]), ts)
oi = interp2d(M1S, M2S, OSNRS)

def wt_fn(m1obs, m2obs, dlobs):
    m1wt = 1.0/(m1obs*(log(MObsMax) - log(MObsMin)))
    m2wt = 1.0/(m1obs - MObsMin)

    dlwt = 3.0*dlobs**2/dLmax**3

    return m1wt*m2wt*dlwt

with tqdm(total=args.number) as bar:
    while N_det < args.number:
        N_gen += 1
        m1obs = exp(np.random.uniform(low=log(MObsMin), high=log(MObsMax)))
        m2obs = np.random.uniform(low=MObsMin, high=m1obs)

        dl = dLmax*cbrt(rand())

        opt_snr = oi(m1obs, m2obs)/dl

        th = tcum(rand())

        A = th * opt_snr

        sa = 1.0
        st = 0.15*8.0/A

        Aobs = np.random.normal(loc=A, scale=sa)

        if Aobs > 8:
            N_det += 1
            bar.update(1)

            m1s_det.append(m1obs)
            m2s_det.append(m2obs)
            dls_det.append(dl)
            wts_det.append(wt_fn(m1obs, m2obs, dl))

m1s_det = concatenate((array(m1s_det), m1s_det_file))
m2s_det = concatenate((array(m2s_det), m2s_det_file))
dls_det = concatenate((array(dls_det), dls_det_file))
wts_det = concatenate((array(wts_det), wts_det_file))

N_gen += N_gen_file

with h5py.File('selected.h5', 'w') as out:
    out.attrs['MObsMin'] = MObsMin
    out.attrs['MObsMax'] = MObsMax
    out.attrs['dLMax'] = dLmax
    out.attrs['NGen'] = N_gen

    for n, d in (('m1', m1s_det), ('m2', m2s_det), ('dl', dls_det), ('wt', wts_det)):
        out.create_dataset(n, data=array(d), compression='gzip', shuffle=True)
