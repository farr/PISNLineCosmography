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

  real dNdm1obsdm2obsddLdt(real m1obs, real m2obs, real dl, real z, real R0, real MMin, real MMax, real alpha, real beta, real gamma, real dH, real Om, real bw_low, real bw_high) {
    real m1 = m1obs / (1+z);
    real m2 = m2obs / (1+z);
    real m1norm = (1.0-alpha)/(MMax^(1-alpha) - MMin^(1-alpha));
    real m2norm = (1.0+beta)/(m1^(1+beta) - MMin^(1+beta));

    real dNdm1dm2dVdt = R0 * m1^(-alpha) * m2^beta * m1norm * m2norm * (1+z)^(gamma-1);

    real dVdz = 4.0*pi()*(dl/(1+z))^2*dH/Ez(z,Om);
    real dzddL_ = dzddL(dl, z, dH, Om);

    real sl;
    real sh;

    real dmdmobs = 1.0/(1.0+z);

    if (m2 < MMin) {
      sl = exp(-0.5*(m2-MMin)^2/bw_low^2);
    } else {
      sl = 1.0;
    }

    if (m1 > MMax) {
      sh = exp(-0.5*(m1-MMax)^2/bw_high^2);
    } else {
      sh = 1.0;
    }

    return dNdm1dm2dVdt * dmdmobs^2 * dVdz * dzddL_ * sl * sh;
  }
}

data {
  int nobs;
  int nsamp;
  int ndet;

  int ninterp;

  real m1obs[nobs,nsamp];
  real m2obs[nobs,nsamp];
  real dlobs[nobs,nsamp];

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

  real bw_low[nobs];
  real bw_high[nobs];

  for (i in 1:nobs) {
    bw_low[i] = sd(m2obs[i,:])/nsamp^0.2;
    bw_high[i] = sd(m1obs[i,:])/nsamp^0.2;
  }

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
}

transformed parameters {
  real dH = 4.42563416002 * (67.74/H0);
}

model {
  real Nex;
  real sigma_rel_Nex;
  real zinterp[ninterp];

  R0 ~ lognormal(log(100), 1);
  H0 ~ lognormal(log(70), 15.0/70.0);

  alpha ~ normal(1, 2);
  beta ~ normal(0, 2);
  gamma ~ normal(3,2);

  MMin ~ normal(5, 2);
  MMax ~ normal(40, 10);

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

  /* Poisson norm */
  {
    real fsum;
    real fs[ndet];

    for (i in 1:ndet) {
      real zobs;

      zobs = interp1d(dlobs_det[i], dlinterp, zinterp);

      fs[i] = dNdm1obsdm2obsddLdt(m1obs_det[i], m2obs_det[i], dlobs_det[i], zobs, R0, MMin, MMax, alpha, beta, gamma, dH, Om, smooth_low, smooth_high);

      /* It can happen when *both* m1 and m2 are smaller than MMin that negative
      /* numbers come out.  Hopefully this is a small fraction of the total! */
      if (fs[i] < 0) fs[i] = 0.0;

      /* Re-weight */
      fs[i] = fs[i] / wts_det[i];
    }

    fsum = sum(fs);

    Nex = Tobs/Ngen*fsum;
  }

  /* Marginalize over likelihood */
  for (i in 1:nobs) {
    real fs[nsamp];

    for (j in 1:nsamp) {
      real z;

      z = interp1d(dlobs[i,j], dlinterp, zinterp);

      fs[j] = dNdm1obsdm2obsddLdt(m1obs[i,j], m2obs[i,j], dlobs[i,j], z, R0, MMin, MMax, alpha, beta, gamma, dH, Om, bw_low[i], bw_high[i]);

      /* If m1 *and* m2 are smaller than MMin, then there is a negative number
         in the density, so get rid of it.  Hopefully this occurs rarely. */
      if (fs[j] < 0.0) fs[j] = 0.0;
    }

    target += log(mean(fs));
  }

  /* Poisson Norm */
  target += -Nex;
}
