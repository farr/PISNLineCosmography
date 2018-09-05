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

  real Ez(real z, real Om) {
    real opz = 1.0 + z;
    real opz2 = opz*opz;
    real opz3 = opz2*opz;

    return sqrt(opz3*Om + (1.0-Om));
  }

  real dzddL(real dl, real z, real dH, real Om) {
    return 1.0/(dl/(1+z) + (1+z)*dH/Ez(z,Om));
  }

  real [] dzddL_system(real dl, real[] state, real[] theta, real[] x_r, int[] x_i) {
    real Om = x_r[1];
    real dH = theta[1];
    real z = state[1];

    real dstatedDL[1];

    /* DL = (1+z) DC and d(DC)/dz = dH/E(z) => this equation */
    dstatedDL[1] = dzddL(dl, z, dH, Om);

    return dstatedDL;
  }

  real log_dNdm1dm2ddLdt(real m1, real m2, real dl, real z, real R0, real MMin, real MMax, real alpha, real beta, real gamma, real dH, real Om) {
    real log_m1norm = log((1.0-alpha)/(MMax^(1-alpha) - MMin^(1-alpha)));
    real m2norm_neg = (1.0+beta)/(m1^(1+beta) - MMin^(1+beta));
    real log_m2norm = log((m2norm_neg < 0 ? -m2norm_neg : m2norm_neg));

    real log_dNdm1dm2dVdt = log(R0) - alpha*log(m1) + beta*log(m2) + log_m1norm + log_m2norm + (gamma-1)*log1p(z);

    real log_dVdz = log(4.0*pi()) + 2.0*log(dl/(1+z)) + log(dH) - log(Ez(z,Om));
    real log_dzddL = log(dzddL(dl, z, dH, Om));

    real log_sl;
    real log_sh;

    return log_dNdm1dm2dVdt + log_dVdz + log_dzddL;
  }
}

data {
  int nobs;
  int ndet;

  int ninterp;

  real m1obs[nobs];
  real m2obs[nobs];
  real dlobs[nobs];

  real m1obs_det[ndet];
  real m2obs_det[ndet];
  real dlobs_det[ndet];
  real wts_det[ndet];

  real Tobs;

  real dLMax;
  int Ngen;

  real smooth_low;
  real smooth_high;
}

transformed data {
  real dlinterp[ninterp];
  real Om = 0.3075;

  real x_r[1];
  int x_i[0];

  real MMin = 5.0;

  x_r[1] = Om;

  for (i in 1:ninterp) {
    dlinterp[i] = (i-1.0)/(ninterp-1.0)*dLMax;
  }
}

parameters {
  real<lower=50,upper=100> H0;
  real<lower=0> R0;

  real<lower=-3,upper=3> alpha;
  real<lower=-3,upper=3> beta;
  real<lower=-3,upper=5> gamma;

  real<lower=0,upper=100> dMMax;
}

transformed parameters {
  real dH = 4.42563416002 * (67.74/H0);

  real Nex;
  real sigma_log_Nex;
  real ndet_neff;

  real zinterp[ninterp];

  real m1_true[nobs];
  real m2_true[nobs];
  real z_true[nobs];

  real MMax;

  /* Interpolate over redshifts */
  {
    real state0[1];
    real theta[1];
    real states[ninterp-1,1];

    state0[1] = 0.0;
    theta[1] = dH;

    states = integrate_ode_rk45(dzddL_system, state0, 0.0, dlinterp[2:], theta, x_r, x_i);
    zinterp[1] = 0.0;
    zinterp[2:] = states[:,1];
  }

  for (i in 1:nobs) {
    z_true[i] = interp1d(dlobs[i], dlinterp, zinterp);
    m1_true[i] = m1obs[i] / (1+z_true[i]);
    m2_true[i] = m2obs[i] / (1+z_true[i]);
  }

  MMax = max(m1_true) + dMMax;

  /* Poisson norm */
  {
    real fsum;
    real fsum2;
    real fs[ndet];
    real fs2[ndet];

    for (i in 1:ndet) {
      real zobs;
      real m1;
      real m2;

      zobs = interp1d(dlobs_det[i], dlinterp, zinterp);
      m1 = m1obs_det[i] / (1+zobs);
      m2 = m2obs_det[i] / (1+zobs);

      fs[i] = log_dNdm1dm2ddLdt(m1, m2, dlobs_det[i], zobs, R0, MMin, MMax, alpha, beta, gamma, dH, Om);

      /* Re-weight */
      fs[i] = fs[i] - log(wts_det[i]);

      /* Jacobian d(m)/d(mobs) for m1 and m2 */
      fs[i] = fs[i] - 2.0*log1p(zobs);

      /* Now smooth if the samples are outside the mass limits. */
      if (m2 < MMin) {
        fs[i] = fs[i] - 0.5*(m2-MMin)^2/smooth_low^2;
      }

      if (m1 > MMax) {
        fs[i] = fs[i] - 0.5*(m1-MMax)^2/smooth_high^2;
      }
    }

    for (i in 1:ndet) {
      fs2[i] = 2.0*fs[i];
    }

    fsum = exp(log_sum_exp(fs));
    fsum2 = exp(log_sum_exp(fs2));

    Nex = Tobs/Ngen*fsum;
    sigma_log_Nex = sqrt(fsum2 - fsum*fsum/ndet)/fsum;
    ndet_neff = 1.0/sigma_log_Nex^2;
  }
}

model {
  R0 ~ lognormal(log(100), 1);
  H0 ~ lognormal(log(70), 15.0/70.0);

  alpha ~ normal(1, 2);
  beta ~ normal(0, 2);
  gamma ~ normal(3,2);

  /* No Jacobian needed because MMin = XX - dMMin */
  MMax ~ normal(40, 10);

  /* Population prior for each system. */
  for (i in 1:nobs) {
    target += log_dNdm1dm2ddLdt(m1_true[i], m2_true[i], dlobs[i], z_true[i], R0, MMin, MMax, alpha, beta, gamma, dH, Om);
    /* But we have data of m1obs, m2obs, dl, so need a Jacobian */
    target += -2.0*log1p(z_true[i]);
  }

  /* Poisson Norm */
  target += -Nex;
}
