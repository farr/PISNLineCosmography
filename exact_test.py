from pylab import *

from argparse import ArgumentParser
import arviz as az
import astropy.cosmology as cosmo
from astropy.cosmology import Planck15
import astropy.units as u
import h5py
import pystan
from true_params import true_params

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--nsel', required=True, type=int, help='number of selection function samples')

    args = parser.parse_args()

    model = pystan.StanModel(file='model_exact.stan')

    with h5py.File('observations.h5', 'r') as f:
        nobs, Tobs = f.attrs['1yr']
        nobs = int(round(nobs))
        m1s = array(f['m1s'][:nobs])
        m2s = array(f['m2s'][:nobs])
        zs = array(f['zs'][:nobs])
        dls = Planck15.luminosity_distance(zs).to(u.Gpc).value

        m1obs = m1s*(1+zs)
        m2obs = m2s*(1+zs)

    nsel = args.nsel
    with h5py.File('selected.h5', 'r') as f:
        N_gen = f.attrs['N_gen']
        m1s_det = array(f['m1det'])
        m2s_det = array(f['m2det'])
        dls_det = array(f['dl'])
        wts_det = array(f['wt'])

    N = len(m1s_det)
    N_gen = int(round(N_gen * nsel / float(N)))
    m1s_det = m1s_det[:nsel]
    m2s_det = m2s_det[:nsel]
    dls_det = dls_det[:nsel]
    wts_det = wts_det[:nsel]

    ninterp = 512
    zinterp = expm1(linspace(log(1), log(11), ninterp))

    d = {
        'nobs': nobs,
        'nsel': nsel,

        'Tobs': Tobs,
        'N_gen': N_gen,

        'm1obs': m1obs,
        'm2obs': m2obs,
        'dlobs': dls,

        'm1sel': m1s_det,
        'm2sel': m2s_det,
        'dlsel': dls_det,
        'log_wtsel': log(wts_det),

        'ninterp': ninterp,
        'zinterp': zinterp,

        'cosmo_prior': 0,

        'd_p': Planck15.luminosity_distance(true_params['z_p']).to(u.Gpc).value,
        'z_p': true_params['z_p']
    }

    fit = model.sampling(data=d)

    az.to_netcdf(fit, 'exact-{:d}.nc'.format(args.nsel))
