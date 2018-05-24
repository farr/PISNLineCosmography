functions {
  real Ez(real z, real Om) {
    real opz = 1.0 + z;
    real opz2 = opz*opz;
    real opz3 = opz2*opz;

    return sqrt(opz3*Om + (1.0-Om));
  }

  real [] dzdDL(real dl, real[] state, real[] theta, real[] x_r, int[] x_i) {
    real Om = x_r[1];
    real dH = theta[1];
    real z = state[1];

    real dstatedDL[1];

    /* DL = (1+z) DC and d(DC)/dz = dH/E(z) => this equation */
    dstatedDL[1] = 1.0/(dl/(1+z) + (1+z)*dH/Ez(z, Om));

    return dstatedDL;
  }

  real dNdm1obsdqddl(real m1obs, real dl, real z, real R0, real alpha, real MMin, real MMax, real gamma, real dH, real Om) {
    real dVdz;
    real dzddl;

    dVdz = 4.0*pi()*dH*(dl/(1+z))^2/Ez(z, Om);
    dzddl = 1.0/(dl/(1+z) + dH*(1+z)/Ez(z, Om));

    return R0*(1-alpha)/(MMax^(1-alpha) - MMin^(1-alpha))*(m1obs/(1+z))^(-alpha)/(1+z)*dVdz*dzddl*(1+z)^(gamma-1);
  }
}

data {
  int nobs;
  int nsamp;

  real m1s[nobs, nsamp];
  real dls[nobs, nsamp];

  int ndet;
  int ngen;
  real Vgen;
  real m1s_det[ndet];
  real dls_det[ndet];
}

transformed data {
  real dls_1d[nobs*nsamp];
  real dls_1d_sorted[nobs*nsamp];
  int dls_ind[nobs*nsamp];

  real sms[nobs];

  real dls_det_sorted[ndet];
  int dls_det_ind[ndet];

  real sm_det;

  real Om;

  real x_r[1];
  int x_i[0];

  Om = 0.3075;
  x_r[1] = Om;

  for (i in 1:nobs) {
    for (j in 1:nsamp) {
      dls_1d[(i-1)*nsamp + j] = dls[i,j];
    }
  }

  dls_ind = sort_indices_asc(dls_1d);
  for (i in 1:nobs*nsamp) {
    dls_1d_sorted[i] = dls_1d[dls_ind[i]];
  }

  for (i in 1:nobs) {
    sms[i] = sd(m1s[i,:])/nsamp^0.2; // Silverman/Scott rule for bandwidth estimation.
  }
  sm_det = sd(m1s_det)/ndet^0.2;

  dls_det_ind = sort_indices_asc(dls_det);
  for (i in 1:ndet) {
    dls_det_sorted[i] = dls_det[dls_det_ind[i]];
  }
}

parameters {
  real<lower=0> h;

  real<lower=0> R0;
  real<lower=1.0,upper=10.0> MMin;
  real<lower=30, upper=60.0> MMax;
  real<lower=-5, upper=2> alpha;
  real<lower=-1, upper=5> gamma;
}

transformed parameters {
  real H0 = 100.0*h;
  real dH = 4.42563416002 * (67.74/H0);
}

model {
  real fs_det[ndet];
  real zs[nobs, nsamp];

  real zs_det[ndet];

  {
    real zs_1d[nobs*nsamp];
    real theta[1];
    real state0[1];
    real states[nobs*nsamp, 1];

    theta[1] = dH;
    state0[1] = 0.0;

    states = integrate_ode_rk45(dzdDL, state0, 0.0, dls_1d_sorted, theta, x_r, x_i);

    for (i in 1:nobs*nsamp) {
      zs_1d[dls_ind[i]] = states[i,1];
    }

    for (i in 1:nobs) {
      for (j in 1:nsamp) {
        zs[i,j] = zs_1d[(i-1)*nsamp + j];
      }
    }
  }

  {
    real theta[1];
    real state0[1];
    real states[ndet,1];

    theta[1] = dH;
    state0[1] = 0.0;

    states = integrate_ode_rk45(dzdDL, state0, 0.0, dls_det_sorted, theta, x_r, x_i);

    for (i in 1:ndet) {
      zs_det[dls_det_ind[i]] = states[i,1];
    }
  }

  h ~ lognormal(log(0.7), 0.2);

  R0 ~ lognormal(log(100.0), 1.0);
  alpha ~ normal(1.0, 1.0);
  gamma ~ normal(3.0, 1.0);
  MMin ~ normal(5.0, 1.0);
  MMax ~ normal(40.0, 10.0);

  for (i in 1:nobs) {
    real fs[nsamp];

    for (j in 1:nsamp) {
      fs[j] = normal_cdf(MMax*(1+zs[i,j]), m1s[i,j], sms[i]) * (1.0 - normal_cdf(MMin*(1+zs[i,j]), m1s[i,j], sms[i])) * dNdm1obsdqddl(m1s[i,j], dls[i,j], zs[i,j], R0, alpha, MMin, MMax, gamma, dH, Om);
    }

    target += log(mean(fs));
  }

  // Poisson norm; we marginalise over the uncertainty in the Monte-Carlo integral.
  for (i in 1:ndet) {
    fs_det[i] = normal_cdf(MMax*(1+zs_det[i]), m1s_det[i], sm_det) * (1.0 - normal_cdf(MMin*(1+zs_det[i]), m1s_det[i], sm_det)) * dNdm1obsdqddl(m1s_det[i], dls_det[i], zs_det[i], R0, alpha, MMin, MMax, gamma, dH, Om);
  }
  target += -Vgen/ngen*sum(fs_det);
}
