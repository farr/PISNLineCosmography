functions {
  int bisect_index(real x, real[] xs) {
    int n = size(xs);
    int i = 1;
    int j = n;
    real xi = xs[i];
    real xj = xs[j];

    if (x < xs[1] || x > xs[n]) reject("cannot interpolate out of bounds");

    while (j - i > 1) {
      int k = i + (j-i)/2;
      real xk = xs[k];

      if (x <= xk) {
        j = k;
        xj = xk;
      } else {
        i = k;
        xi = xk;
      }
    }

    return j;
  }

  real interp1d(real x, real[] xs, real[] ys) {
    int i = bisect_index(x, xs);

    real x0 = xs[i-1];
    real x1 = xs[i];
    real y0 = ys[i-1];
    real y1 = ys[i];

    real r = (x-x0)/(x1-x0);

    return r*y1 + (1.0-r)*y0;
  }

  real wz(real z, real z_p, real w_p, real w_a) {
    real a = 1.0/(1.0 + z);
    real a_p = 1.0/(1.0 + z_p);

    return w_p + w_a*(a_p - a);
  }

  real Ez(real z, real Om, real z_p, real w_p, real w_a) {
    real opz = 1.0 + z;
    real opz2 = opz*opz;
    real opz3 = opz2*opz;

    real w = wz(z, z_p, w_p, w_a);

    return sqrt(opz3*Om + (1.0-Om)*(1.0+z)^(3*(1+w)));
  }

  real dzddL(real dl, real z, real dH, real Om, real z_p, real w_p, real w_a) {
    return 1.0/(dl/(1+z) + (1+z)*dH/Ez(z,Om,z_p,w_p,w_a));
  }

  real [] dzddL_system(real dl, real[] state, real[] theta, real[] x_r, int[] x_i) {
    real z_p = x_r[1];
    real dH = theta[1];
    real Om = theta[2];
    real w_p = theta[3];
    real w_a = theta[4];
    real z = state[1];

    real dstatedDL[1];

    /* DL = (1+z) DC and d(DC)/dz = dH/E(z) => this equation */
    dstatedDL[1] = dzddL(dl, z, dH, Om, z_p, w_p, w_a);

    return dstatedDL;
  }

  real[] log_dNdm1dm2ddLdt(real[] m1, real[] m2, real[] dl, real[] z, real R0, real MMin, real MMax, real alpha, real beta, real gamma, real dH, real Om, real z_p, real w_p, real w_a, real[] smooth_low, real[] smooth_high) {
    int N = size(m1);
    real log_dNs[N];

    real log_m1norm = log((1.0-alpha)/(MMax^(1-alpha) - MMin^(1-alpha)));

    real log_dN_prefactor = log(R0) + log_m1norm;

    for (i in 1:N) {
      /* In the event that this function is called with m1 < MMin, then this can
         go negative (obv this is "out of bounds", but can happen in the selection
         function). */
      real m2norm_neg = (1.0+beta)/(m1[i]^(1+beta) - MMin^(1+beta));
      real log_m2norm = log((m2norm_neg < 0 ? -m2norm_neg : m2norm_neg));

      real log_dNdm1dm2dVdt = log_dN_prefactor + log_m2norm - alpha*log(m1[i]) + beta*log(m2[i]) + (gamma-1)*log1p(z[i]);

      real log_dVdz = log(4.0*pi()) + 2.0*log(dl[i]/(1+z[i])) + log(dH) - log(Ez(z[i],Om,z_p,w_p,w_a));
      real log_dzddL = log(dzddL(dl[i], z[i], dH, Om, z_p, w_p, w_a));

      real log_sl;
      real log_sh;

      if (m1[i] > MMax) {
        log_sh = -0.5*(m1[i]-MMax)^2/smooth_high[i]^2;
      } else {
        log_sh = 0.0;
      }

      if (m2[i] < MMin) {
        log_sl = -0.5*(m2[i]-MMin)^2/smooth_low[i]^2;
      } else {
        log_sl = 0.0;
      }

      log_dNs[i] = log_dNdm1dm2dVdt + log_dVdz + log_dzddL + log_sh + log_sl;
    }

    return log_dNs;
  }
}

data {
  int nobs;
  int nsamp;
  int ndet;

  int ninterp;

  real m1obs[nobs, nsamp];
  real m2obs[nobs, nsamp];
  real dlobs[nobs, nsamp];

  real m1obs_det[ndet];
  real m2obs_det[ndet];
  real dlobs_det[ndet];
  real wts_det[ndet];

  real Tobs;

  real dLMax;
  int Ngen;

  real smooth_low;
  real smooth_high;

  /* Gaussian priors on cosmography */
  real mu_H0;
  real sigma_H0;
  real mu_Om;
  real sigma_Om;
  real mu_Om_h2;
  real sigma_Om_h2;
  real mu_wp;
  real sigma_wp;
  real mu_wa;
  real sigma_wa;
  int use_Om_h2; /* 1 if you want to set the prior on Om_h2, so Om ignored; otherwise Om_h2 ignored */

  /* Pivot redshift */
  real z_p;
}

transformed data {
  real dlinterp[ninterp];

  real x_r[1];
  int x_i[0];

  real bw_high[nobs];
  real bw_low[nobs];

  x_r[1] = z_p;

  for (i in 1:ninterp) {
    dlinterp[i] = (i-1.0)/(ninterp-1.0)*dLMax;
  }

  for (i in 1:nobs) {
    bw_high[i] = sd(m1obs[i,:])/nsamp^0.2;
    bw_low[i] = sd(m2obs[i,:])/nsamp^0.2;
  }
}

parameters {
  real<lower=50,upper=100> H0;
  real<lower=0,upper=1> Om;
  real w_p;
  real w_a;

  real<lower=0> R0;

  real<lower=-3,upper=3> alpha;
  real<lower=-3,upper=3> beta;
  real<lower=-3,upper=5> gamma;

  real<lower=3,upper=10> MMin;
  real<lower=30,upper=100> MMax;
}

transformed parameters {
  real dH = 4.42563416002 * (67.74/H0);
  real Om_h2 = Om * (H0/100.0) * (H0/100.0); /* Planck measures this. */
  real w0;
  real Nex;
  real sigma_Nex;
  real neff_det;

  real zinterp[ninterp];

  w0 = wz(0.0, z_p, w_p, w_a);

  /* Interpolate over redshifts */
  {
    real state0[1];
    real theta[4];
    real states[ninterp-1,1];

    state0[1] = 0.0;

    theta[1] = dH;
    theta[2] = Om;
    theta[3] = w_p;
    theta[4] = w_a;

    states = integrate_ode_rk45(dzddL_system, state0, 0.0, dlinterp[2:], theta, x_r, x_i);
    zinterp[1] = 0.0;
    zinterp[2:] = states[:,1];
  }

  /* Poisson norm */
  {
    real fsum;
    real fsum2;
    real fs[ndet];
    real fs2[ndet];
    real zobs[ndet];
    real m1[ndet];
    real m2[ndet];
    real dNs[ndet];
    real sls[ndet];
    real shs[ndet];

    for (i in 1:ndet) {
      zobs[i] = interp1d(dlobs_det[i], dlinterp, zinterp);
      m1[i] = m1obs_det[i]/(1+zobs[i]);
      m2[i] = m2obs_det[i]/(1+zobs[i]);

      sls[i] = smooth_low;
      shs[i] = smooth_high;
    }

    dNs = log_dNdm1dm2ddLdt(m1, m2, dlobs_det, zobs, R0, MMin, MMax, alpha, beta, gamma, dH, Om, z_p, w_p, w_a, sls, shs);

    for (i in 1:ndet) {
      fs[i] = dNs[i] - 2.0*log1p(zobs[i]) - log(wts_det[i]);

      fs2[i] = 2.0*fs[i];
    }

    fsum = exp(log_sum_exp(fs));
    fsum2 = exp(log_sum_exp(fs2));

    Nex = Tobs/Ngen*fsum;
    sigma_Nex = sqrt(fsum2 - fsum*fsum/ndet)*Tobs/Ngen;
    neff_det = Nex*Nex/(sigma_Nex*sigma_Nex);
  }
}

model {
  R0 ~ lognormal(log(100), 1);

  H0 ~ normal(mu_H0, sigma_H0);

  if (use_Om_h2 == 0) {
    Om ~ normal(mu_Om, sigma_Om);
  } else {
    Om_h2 ~ normal(mu_Om_h2, sigma_Om_h2);
    /* We need a Jacobian: d(Om_h2)/d(Om) = (H0/100)^2 */
    target += 2.0*log(H0/100.0);
  }
  w_p ~ normal(mu_wp, sigma_wp);
  w_a ~ normal(mu_wa, sigma_wa);

  alpha ~ normal(1, 2);
  beta ~ normal(0, 2);
  gamma ~ normal(3,2);

  MMin ~ normal(5, 2);
  MMax ~ normal(40, 10);

  for (i in 1:nobs) {
    real log_fs[nsamp];
    real m1[nsamp];
    real m2[nsamp];
    real z[nsamp];
    real bl[nsamp];
    real bh[nsamp];

    for (j in 1:nsamp) {
      z[j] = interp1d(dlobs[i,j], dlinterp, zinterp);
      m1[j] = m1obs[i,j]/(1+z[j]);
      m2[j] = m2obs[i,j]/(1+z[j]);

      bl[j] = bw_low[i]/(1+z[j]);
      bh[j] = bw_high[i]/(1+z[j]);
    }

    log_fs = log_dNdm1dm2ddLdt(m1, m2, dlobs[i,:], z, R0, MMin, MMax, alpha, beta, gamma, dH, Om, z_p, w_p, w_a, bl, bh);

    target += log_sum_exp(log_fs) - log(nsamp);
  }

  /* Poisson norm. */
  target += -Nex;
}
