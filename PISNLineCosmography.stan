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

    /* In the event that this function is called with m1 < MMin, then this can
       go negative (obv this is "out of bounds", but can happen in the selection
       function). */
    real m2norm_neg = (1.0+beta)/(m1^(1+beta) - MMin^(1+beta));
    real log_m2norm = log((m2norm_neg < 0 ? -m2norm_neg : m2norm_neg));

    real log_dNdm1dm2dVdt = log(R0) - alpha*log(m1) + beta*log(m2) + log_m1norm + log_m2norm + (gamma-1)*log1p(z);

    real log_dVdz = log(4.0*pi()) + 2.0*log(dl/(1+z)) + log(dH) - log(Ez(z,Om));
    real log_dzddL = log(dzddL(dl, z, dH, Om));

    return log_dNdm1dm2dVdt + log_dVdz + log_dzddL;
  }
}

data {
  int nobs;
  int nsamp;
  int ndet;

  int ninterp;

  vector[3] m1obs_m2obs_dl[nobs, nsamp];
  cov_matrix[3] bw[nobs];

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

  matrix[3,3] chol_bw[nobs];

  x_r[1] = Om;

  for (i in 1:ninterp) {
    dlinterp[i] = (i-1.0)/(ninterp-1.0)*dLMax;
  }

  for (i in 1:nobs) {
    chol_bw[i] = cholesky_decompose(bw[i]);
  }
}

parameters {
  real<lower=50,upper=100> H0;
  real<lower=0> R0;

  real<lower=-3,upper=3> alpha;
  real<lower=-3,upper=3> beta;
  real<lower=-3,upper=5> gamma;

  real<lower=3,upper=10> MMin;
  real<lower=30,upper=100> MMax;

  real<lower=MMin,upper=MMax> m1_true[nobs];
  real<lower=0,upper=1> m2_frac[nobs];
  real<lower=0,upper=dLMax> dl_true[nobs];
}

transformed parameters {
  real dH = 4.42563416002 * (67.74/H0);
  real Nex;
  real sigma_log_Nex;
  real neff_det;

  real m2_true[nobs];
  real z_true[nobs];

  real zinterp[ninterp];

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

  /* Compute m2_true, z_true */
  for (i in 1:nobs) {
    m2_true[i] = MMin + (m1_true[i]-MMin)*m2_frac[i];
    z_true[i] = interp1d(dl_true[i], dlinterp, zinterp);
  }

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

      m1 = m1obs_det[i]/(1+zobs);
      m2 = m2obs_det[i]/(1+zobs);

      fs[i] = log_dNdm1dm2ddLdt(m1, m2, dlobs_det[i], zobs, R0, MMin, MMax, alpha, beta, gamma, dH, Om);

      /* Re-weight */
      fs[i] = fs[i] - log(wts_det[i]);

      /* Two factors of dm/d(mobs) = 1/(1+z) for Jacobian */
      fs[i] = fs[i] - 2.0*log1p(zobs);

      /* Smooth. */
      if (m1 > MMax) {
        fs[i] = fs[i] - 0.5*(m1-MMax)^2/smooth_high^2;
      }

      if (m2 < MMin) {
        fs[i] = fs[i] - 0.5*(m2-MMin)^2/smooth_low^2;
      }

      fs2[i] = 2.0*fs[i];
    }

    fsum = exp(log_sum_exp(fs));
    fsum2 = exp(log_sum_exp(fs2));

    Nex = Tobs/Ngen*fsum;
    sigma_log_Nex = sqrt(fsum2 - fsum*fsum/ndet)/fsum;
    neff_det = 1.0/(sigma_log_Nex^2);
  }
}

model {
  R0 ~ lognormal(log(100), 1);
  H0 ~ lognormal(log(70), 15.0/70.0);

  alpha ~ normal(1, 2);
  beta ~ normal(0, 2);
  gamma ~ normal(3,2);

  MMin ~ normal(5, 2);
  MMax ~ normal(40, 10);

  /* Population prior */
  for (i in 1:nobs) {
    target += log_dNdm1dm2ddLdt(m1_true[i], m2_true[i], dl_true[i], z_true[i], R0, MMin, MMax, alpha, beta, gamma, dH, Om);
    /* Jacobian because we sample in m2_frac: dm2/d(m2_frac) = (m1-MMin). */
    target += log(m1_true[i]-MMin);
  }

  /* Implement KDE likelihood */
  for (i in 1:nobs) {
    real fs[nsamp];
    vector[3] x;

    x[1] = m1_true[i]*(1+z_true[i]);
    x[2] = m2_true[i]*(1+z_true[i]);
    x[3] = dl_true[i];

    for (j in 1:nsamp) {
      fs[j] = multi_normal_cholesky_lpdf(m1obs_m2obs_dl[i,j] | x, chol_bw[i]);
    }

    target += log_sum_exp(fs) - log(nsamp);
  }

  /* Poisson norm */
  target += -Nex;
}
