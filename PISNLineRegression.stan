functions {
  real Ez(real z, real Om) {
    real opz = 1.0 + z;
    real opz2 = opz*opz;
    real opz3 = opz2*opz;

    return sqrt(opz3*Om + (1.0-Om));
  }

  real[] dDCudz(real z, real[] state, real[] theta, real[] x_r, int[] x_i) {
    // real Om = theta[1];
    real Om = x_r[1];
    real dstatedz[1];

    dstatedz[1] = 1.0/Ez(z, Om);

    return dstatedz;
  }

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
    real xl = xs[i-1];
    real xh = xs[i];
    real yl = ys[i-1];
    real yh = ys[i];
    real r = (x-xl)/(xh-xl);

    return r*yh + (1.0-r)*yl;
  }

  real interp2d(real x, real y, real[] xs, real[] ys, real[,] zs) {
    int i = bisect_index(x, xs);
    int j = bisect_index(y, ys);

    real xl = xs[i-1];
    real xh = xs[i];
    real yl = ys[j-1];
    real yh = ys[j];

    real r = (x-xl)/(xh-xl);
    real s = (y-yl)/(yh-yl);

    return (1.0-r)*(1.0-s)*zs[i-1,j-1] + (1.0-r)*s*zs[i-1,j] + r*(1.0-s)*zs[i,j-1] + r*s*zs[i,j];
  }
}

data {
  int nobs;
  vector[nobs] mc_obs;
  vector[nobs] eta_obs;
  vector[nobs] A_obs;
  vector[nobs] theta_obs;

  vector[nobs] sigma_mc;
  vector[nobs] sigma_eta;
  vector[nobs] sigma_theta;

  int nm;
  real ms[nm];
  real opt_snr[nm,nm];

  real dlinterp_max;
}

transformed data {
  int ninterp = 100;
  real zinterp[ninterp];
  real x_r[1];
  int x_i[0];
  real Om = 0.3;

  real minterp_min = min(ms);
  real minterp_max = max(ms);

  x_r[1] = Om;

  for (i in 1:ninterp) {
    zinterp[i] = (i-1.0)/(ninterp-1.0)*zinterp_max;
  }
}

parameters {
  real<lower=0> h;
  // real<lower=0,upper=1> Om;

  real<lower=0,upper=zinterp_max> zmax;
  real<lower=minterp_min,upper=10> MMin;
  real<lower=30, upper=minterp_max/(1.0+zmax)> MMax;

  vector<lower=MMin, upper=MMax>[nobs] m1s;
  vector<lower=0, upper=1>[nobs] qs;
  vector<lower=0, upper=zmax>[nobs] zs;
  vector<lower=0, upper=1>[nobs] thetas;
}

transformed parameters {
  real H0 = 100.0*h;
  real dH = 4.42563416002 * (67.74/H0);
  vector[nobs] m2s;
  vector[nobs] mcs;
  vector[nobs] etas;
  vector[nobs] dls;
  vector[nobs] opt_snrs;

  {
    real theta[0];
    real state0[1];
    real states[ninterp-1, 1];
    real dcs[ninterp];

    // theta[1] = Om;
    state0[1] = 0.0;

    states = integrate_ode_rk45(dDCudz, state0, 0.0, zinterp[2:], theta, x_r, x_i);

    dcs[1] = 0.0;
    for (i in 2:ninterp) {
      real dcu = states[i-1,1];
      dcs[i] = dH*dcu;
    }

    for (i in 1:nobs) {
      real dc = interp1d(zs[i], zinterp, dcs);
      real dl = dc*(1.0+zs[i]);

      dls[i] = dl;
    }
  }

  for (i in 1:nobs) {
    m2s[i] = MMin + qs[i]*(MMax-MMin);
  }

  for (i in 1:nobs) {
    real mt = m1s[i] + m2s[i];

    etas[i] = m1s[i]*m2s[i]/(mt*mt);
    mcs[i] = mt*etas[i]^(3.0/5.0);
  }

  for (i in 1:nobs) {
    opt_snrs[i] = interp2d(m1s[i]*(1.0+zs[i]), m2s[i]*(1.0+zs[i]), ms, ms, opt_snr)/dls[i];
  }
}

model {
  h ~ lognormal(log(0.7), 0.2); // H0 = 70 w/20% uncert
  // Flat prior in Om

  // Flat prior on MMin, MMax

  // Flat prior on m1 \in [MMin, MMax]
  target += -nobs*log(MMax - MMin);

  // Flat prior on q \in [0, 1]

  // Flat prior on z in [0, zmax].
  target += -nobs*log(zmax);

  // Flat prior on theta

  // Observations
  theta_obs ~ normal(thetas, sigma_theta);
  mc_obs ~ lognormal(log(mcs .* (1.0+zs)), sigma_mc);
  eta_obs ~ normal(etas, sigma_eta);
  A_obs ~ normal(thetas .* opt_snrs, 1.0);
  theta_obs ~ normal(thetas, sigma_theta);
}
