import astropy.cosmology as cosmo
import astropy.units as u
import lal
import lalsimulation as ls
import multiprocessing as multi
from pylab import *

def draw_thetas(N, iota=None, cos_iota=None):
    """Draw `N` random angular factors for the SNR.

    Theta is as defined in [Finn & Chernoff
    (1993)](https://ui.adsabs.harvard.edu/#abs/1993PhRvD..47.2198F/abstract).

    If `iota` or `cos_iota` are given then the factor will be drawn at that
    fixed inclination. """

    cos_thetas = np.random.uniform(low=-1, high=1, size=N)
    phis = np.random.uniform(low=0, high=2*np.pi, size=N)
    zetas = np.random.uniform(low=0, high=2*np.pi, size=N)

    if iota is not None and cos_iota is not None:
        raise ValueError('cannot give both iota and cos_iota')
    elif iota is not None:
        cos_iota = np.cos(iota)
    elif cos_iota is not None:
        # Do nothing
        pass
    else:
        cos_iota = np.random.uniform(low=-1, high=1, size=N)

    Fps = 0.5*cos(2*zetas)*(1 + square(cos_thetas))*cos(2*phis) - sin(2*zetas)*cos_thetas*sin(2*phis)
    Fxs = 0.5*sin(2*zetas)*(1 + square(cos_thetas))*cos(2*phis) + cos(2*zetas)*cos_thetas*sin(2*phis)

    return np.sqrt(0.25*square(Fps)*square(1 + square(cos_iota)) + square(Fxs)*square(cos_iota))

thetas = draw_thetas(10000)

def next_pow_two(x):
    """Return the next (integer) power of two above `x`.

    """

    x2 = 1
    while x2 < x:
        x2 = x2 << 1
    return x2

def optimal_snr(m1_intrinsic, m2_intrinsic, z, psd_fn=ls.SimNoisePSDaLIGOEarlyHighSensitivityP1200087):
    """Return the optimal SNR of a signal.

    :param m1_intrinsic: The source-frame mass 1.

    :param m2_intrinsic: The source-frame mass 2.

    :param z: The redshift.

    :param psd_fn: A function that returns the detector PSD at a given
      frequency (default is early aLIGO high sensitivity, defined in
      [P1200087](https://dcc.ligo.org/LIGO-P1200087/public).

    :return: The SNR of a face-on, overhead source.

    """

    # Get dL, Gpc
    dL = cosmo.Planck15.luminosity_distance(z).to(u.Gpc).value

    # Basic setup: min frequency for w.f., PSD start freq, etc.
    fmin = 19.0
    fref = 40.0
    psdstart = 20.0

    # This is a conservative estimate of the chirp time + MR time (2 seconds)
    tmax = ls.SimInspiralChirpTimeBound(fmin, m1_intrinsic*(1+z)*lal.MSUN_SI, m2_intrinsic*(1+z)*lal.MSUN_SI, 0.0, 0.0) + 2

    df = 1.0/next_pow_two(tmax)
    fmax = 2048.0 # Hz --- based on max freq of 5-5 inspiral

    # Generate the waveform, redshifted as we would see it in the detector, but with zero angles (i.e. phase = 0, inclination = 0)
    hp, hc = ls.SimInspiralChooseFDWaveform((1+z)*m1_intrinsic*lal.MSUN_SI, (1+z)*m2_intrinsic*lal.MSUN_SI, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dL*1e9*lal.PC_SI, 0.0, 0.0, 0.0, 0.0, 0.0, df, fmin, fmax, fref, None, ls.IMRPhenomPv2)

    Nf = int(round(fmax/df)) + 1
    fs = linspace(0, fmax, Nf)
    sel = fs > psdstart

    # PSD
    sffs = lal.CreateREAL8FrequencySeries("psds", 0, 0.0, df, lal.DimensionlessUnit, fs.shape[0])
    psd_fn(sffs, psdstart)

    return ls.MeasureSNRFD(hp, sffs, psdstart, -1.0)

def fraction_above_threshold(m1_intrinsic, m2_intrinsic, z, snr_thresh, psd_fn=ls.SimNoisePSDaLIGOEarlyHighSensitivityP1200087):
    """Returns the fraction of sources above a given threshold.

    :param m1_intrinsic: Source-frame mass 1.

    :param m2_intrinsic: Source-frame mass 2.

    :param z: Redshift.

    :param snr_thresh: SNR threshold.

    :param psd_fn: Function computing the PSD (see :func:`optimal_snr`).

    :return: The fraction of sources that are above the given
      threshold.

    """
    if z == 0.0:
        return 1.0

    rho_max = optimal_snr(m1_intrinsic, m2_intrinsic, z, psd_fn=psd_fn)

    # From Finn & Chernoff, we have SNR ~ theta*integrand, assuming that the polarisations are
    # orthogonal
    theta_min = snr_thresh / rho_max

    if theta_min > 1:
        return 0.0
    else:
        return np.mean(thetas > theta_min)

def vt_from_mass(m1, m2, thresh, analysis_time, calfactor=1.0, psd_fn=ls.SimNoisePSDaLIGOEarlyHighSensitivityP1200087, opz_power=0):
    """Returns the sensitive time-volume for a given system.

    :param m1: Source-frame mass 1.

    :param m2: Source-frame mass 2.

    :param analysis_time: The total detector-frame searched time.

    :param calfactor: Fudge factor applied multiplicatively to the final result.

    :param psd_fn: Function giving the assumed single-detector PSD
      (see :func:`optimal_snr`).

    :param opz_power: Assume the volumetric rate in the source frame evolves as
      `(1+z)**opz_power`.

    :return: The sensitive time-volume in comoving Gpc^3-yr (assuming
      analysis_time is given in years).

    """
    def integrand(z):
        if z == 0.0:
            return 0.0
        else:
            return 4*np.pi*cosmo.Planck15.differential_comoving_volume(z).to(u.Gpc**3 / u.sr).value*(1+z)**(opz_power-1)*fraction_above_threshold(m1, m2, z, thresh)

    zmax = 1.0
    zmin = 0.001
    assert fraction_above_threshold(m1, m2, zmax, thresh) == 0.0
    assert fraction_above_threshold(m1, m2, zmin, thresh) > 0.0
    while zmax - zmin > 1e-3:
        zhalf = 0.5*(zmax+zmin)
        fhalf = fraction_above_threshold(m1, m2, zhalf, thresh)

        if fhalf > 0.0:
            zmin=zhalf
        else:
            zmax=zhalf

    zs = linspace(0.0, zmax, 20)
    ys = array([integrand(z) for z in zs])
    return calfactor*analysis_time*trapz(ys, zs)

class VTFromMassTuple(object):
    def __init__(self, thresh, analyt, calfactor, psd_fn, opz_power):
        self.thresh = thresh
        self.analyt = analyt
        self.calfactor = calfactor
        self.psd_fn = psd_fn
        self.opz_power = opz_power
    def __call__(self, m1m2):
        m1, m2 = m1m2
        return vt_from_mass(m1, m2, self.thresh, self.analyt, self.calfactor, self.psd_fn, self.opz_power)

def vts_from_masses(m1s, m2s, thresh, analysis_time, calfactor=1.0, psd_fn=ls.SimNoisePSDaLIGOEarlyHighSensitivityP1200087, opz_power=0):
    """Returns array of VTs corresponding to the given systems.

    Uses multiprocessing for more efficient computation.

    """

    vt_m_tuple = VTFromMassTuple(thresh, analysis_time, calfactor, psd_fn, opz_power)

    pool = multi.Pool()
    try:
        vts = array(pool.map(vt_m_tuple, zip(m1s, m2s)))
    finally:
        pool.close()

    return vts
