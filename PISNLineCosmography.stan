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

  real dNdm1dm2ddLdt(real m1, real m2, real dl, real z, real R0, real MMin, real MMax, real alpha, real beta, real gamma, real dH, real Om) {
    real m1norm = (1.0-alpha)/(MMax^(1-alpha) - MMin^(1-alpha));
    real m2norm = (1.0+beta)/(m1^(1+beta) - MMin^(1+beta));

    real dNdm1dm2dVdt = R0 * m1^(-alpha) * m2^beta * m1norm * m2norm * (1+z)^(gamma-1);

    real dVdz = 4.0*pi()*(dl/(1+z))^2*dH/Ez(z,Om);
    real dzddL_ = dzddL(dl, z, dH, Om);

    return dNdm1dm2dVdt * dVdz * dzddL_;
  }
}

data {
  int nobs;
  int nsamp;
  int ndet;

  int ninterp;

  vector[3] m1obs_m2obs_dL[nobs, nsamp];
  matrix[3,3] bw_chol[nobs];

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

  real<lower=3,upper=10> MMin;
  real<lower=30,upper=100> MMax;

  real<lower=MMin, upper=MMax> m1_true[nobs];
  real<lower=0, upper=1> m2_frac[nobs];
  real<lower=0, upper=dLMax> dl_true[nobs];
}

transformed parameters {
  real Nex;
  real sigma_rel_Nex;
  real m2_true[nobs];
  real z_true[nobs];
  real zinterp[ninterp];

  real dH = 4.42563416002 * (67.74/H0);

  for (i in 1:nobs) {
    m2_true[i] = MMin + m2_frac[i]*(m1_true[i] - MMin);
  }

  {
    real state0[1];
    real theta[1];
    real states[ninterp-1,1];

    state0[1] = 0.0;
    theta[1] = dH;

    states = integrate_ode_rk45(dzddL_system, state0, 0.0, dlinterp[2:], theta, x_r, x_i);
    zinterp[1] = 0.0;
    zinterp[2:] = states[:,1];

    for (i in 1:nobs) {
      z_true[i] = interp1d(dl_true[i], dlinterp, zinterp);
    }
  }

  {
    real fsum;
    real fs[ndet];

    for (i in 1:ndet) {
      real zobs;

      zobs = interp1d(dlobs_det[i], dlinterp, zinterp);

      fs[i] = dNdm1dm2ddLdt(m1obs_det[i]/(1+zobs), m2obs_det[i]/(1+zobs), dlobs_det[i], zobs, R0, MMin, MMax, alpha, beta, gamma, dH, Om);

      /* It can happen when *both* m1 and m2 are smaller than MMin that negative
      /* numbers come out.  Hopefully this is a small fraction of the total! */
      if (fs[i] < 0) fs[i] = 0.0;

      /* Transform to dm1obs dm2obs */
      fs[i] = fs[i]/(1+zobs)^2;

      /* Implement smoothing */
      fs[i] = fs[i] * normal_cdf(m2obs_det[i]/(1+zobs), MMin, smooth_low) * (1.0 - normal_cdf(m1obs_det[i]/(1+zobs), MMax, smooth_high));

      /* Re-weight */
      fs[i] = fs[i] / wts_det[i];
    }

    fsum = sum(fs);

    Nex = Tobs/Ngen*fsum;
    sigma_rel_Nex = sqrt(ndet)*sd(fs)/fsum;
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

  /* Impose distribution on m1, m2, dl */
  for (i in 1:nobs) {
    target += log(dNdm1dm2ddLdt(m1_true[i], m2_true[i], dl_true[i], z_true[i], R0, MMin, MMax, alpha, beta, gamma, dH, Om));
    target += log(m1_true[i] - MMin); /* Jacobian because we sample in m2_frac */
  }

  /* Likelihood for each event. */
  for (i in 1:nobs) {
    vector[3] mu;
    real f[nsamp];

    mu[1] = m1_true[i]*(1+z_true[i]);
    mu[2] = m2_true[i]*(1+z_true[i]);
    mu[3] = dl_true[i];

    for (j in 1:nsamp) {
      f[j] = multi_normal_cholesky_lpdf(m1obs_m2obs_dL[i,j] | mu, bw_chol[i]);
    }

    target += log_sum_exp(f) - log(nsamp);
  }

  /* Poisson Norm */
  target += -Nex;
}
