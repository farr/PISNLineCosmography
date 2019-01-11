#!/usr/bin/env python

from pylab import *

import astropy.cosmology as cosmo
from astropy.cosmology import Planck15
import astropy.units as u
import h5py
import multiprocessing as multi
import pystan
from tqdm import tqdm

if __name__ == '__main__':

    with h5py.File('observations.h5', 'r') as f:
        m1s = array(f['m1s'])
        m2s = array(f['m2s'])
        zs = array(f['zs'])
        dls = Planck15.luminosity_distance(zs).to(u.Gpc).value
        thetas = array(f['thetas'])

        mcobs = array(f['mcobs'])
        etaobs = array(f['etaobs'])
        rhoobs = array(f['rhoobs'])
        thetaobs = array(f['thetaobs'])

        sigma_mc = array(f['sigma_mc'])
        sigma_eta = array(f['sigma_eta'])
        sigma_rho = array(f['sigma_rho'])
        sigma_theta = array(f['sigma_t'])

    with h5py.File('optimal_snr.h5', 'r') as f:
        ms_osnr = array(f['ms'])
        osnrs = array(f['SNR'])

    model = pystan.StanModel('infer.stan')

    def process(i):
        data = {
            'mc_obs': mcobs[i],
            'eta_obs': etaobs[i],
            'rho_obs': rhoobs[i],
            'theta_obs': thetaobs[i],

            'sigma_mc': sigma_mc[i],
            'sigma_eta': sigma_eta[i],
            'sigma_theta': sigma_theta[i],

            'nm': len(ms_osnr),
            'ms': ms_osnr,
            'opt_snrs': osnrs,

            'dL_max': Planck15.luminosity_distance(3).to(u.Gpc).value
        }

        init = {
            'm1': m1s[i]*(1+zs[i]),
            'm2': m2s[i]*(1+zs[i]),
            'dL': dls[i]
        }

        fit = model.sampling(data=data, n_jobs=1, control={'max_treedepth': 15, 'adapt_delta': 0.99, 'metric': 'dense_e'}, init=4*(init,))
        chain = fit.extract(permuted=True)

        return i, chain

    with h5py.File('observations.h5', 'a') as f:
        # Delete the group for posteriors:
        try:
            del f['posteriors']
        except:
            pass

        pg = f.create_group('posteriors')

        nobs = len(mcobs)
        nsamp = 4000
        pg.create_dataset('dl', data=zeros((nobs, nsamp)), compression='gzip', shuffle=True)
        pg.create_dataset('m1det', data=zeros((nobs, nsamp)), compression='gzip', shuffle=True)
        pg.create_dataset('m2det', data=zeros((nobs, nsamp)), compression='gzip', shuffle=True)
        pg.create_dataset('theta', data=zeros((nobs, nsamp)), compression='gzip', shuffle=True)
        pg.create_dataset('log_m1m2dl_wt', data=zeros((nobs, nsamp)), compression='gzip', shuffle=True)

        p = multi.Pool()
        try:
            for i, chain in tqdm(p.imap_unordered(process, range(nobs)), total=nobs):
                pg['dl'][i,:] = chain['dL']
                pg['m1det'][i,:] = chain['m1']
                pg['m2det'][i,:] = chain['m2']
                pg['theta'][i,:] = chain['theta']
                pg['log_m1m2dl_wt'][i,:] = chain['log_m1m2dl_wt']
        finally:
            p.close()
