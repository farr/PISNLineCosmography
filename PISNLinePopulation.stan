functions {
  real dNdm1dm2dVdt(real m1, real m2, real z, real R0, real alpha, real gamma, real MMin, real MMax) {
    return R0*(1-alpha)*m1^(-alpha)/(m1 - MMin)/(MMax^(1-alpha) - MMin^(1-alpha))*(1+z)^(gamma-1);
  }

  real Ez(real z, real Om) {
    real opz = 1.0 + z;
    real opz2 = opz*opz;
    real opz3 = opz2*opz;

    return sqrt(opz3*Om + (1.0-Om));
  }

  real[] dDCdz(real z, real[] state, real[] theta, real[] x_r, int[] x_i) {
    real Om = theta[1];

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

    while (j-i > 1) {
      int k = i + (j-i)/2;
      real xk = xs[k];

      if (x > xk) {
        xi = xk;
        i = k;
      } else {
        xj = xk;
        j = k;
      }
    }

    return j;
  }

  real interp1d(real x, real[] xs, real[] ys) {
    int j = bisect_index(x, xs);

    real xi = xs[j-1];
    real xj = xs[j];
    real yi = ys[j-1];
    real yj = ys[j];

    real r = (x-xi)/(xj-xi);

    return r*yj + (1.0-r)*yi;
  }
}

data {
  int nobs;
  vector[nobs] mc_obs;
  vector[nobs] eta_obs;
  vector[nobs] dl_obs;
  vector[nobs] sigma_mc;
  vector[nobs] sigma_eta;
  vector[nobs] sigma_dl;

  real zmax;
  real MMin;

  int nms;
  real ms[nms];
  real opt_snrs[nms,nms];

  int nthetas;
  real thetas[nthetas];
}

transformed data {
  real x_r[0];
  int x_i[0];

  real fabove_thetas[nthetas];

  int ninterp = 100;
  real zinterp[ninterp];
  real zs_out[ninterp-1];

  for (i in 1:nthetas) {
    fabove_thetas[i] = 1.0 - (i-1.0)/(nthetas-1.0);
  }

  for (i in 1:ninterp) {
    zinterp[i] = (i-1.0)/(ninterp-1.0)*zmax;
  }
  zs_out = zinterp[2:];
}

parameters {
  real<lower=30, upper=300> H0;
  real<lower=0,upper=1> Om;

  real<lower=0> R0;
  real<lower=-3, upper=3> alpha;
  real<lower=-5, upper=5> gamma;
  real<lower=20> MMax;

  vector<lower=MMin, upper=MMax>[nobs] m1s;
  vector<lower=0, upper=1>[nobs] f_m2s;
  vector<lower=0, upper=zmax>[nobs] zs;
}

transformed parameters {
  real dlinterp[ninterp];
  real dcinterp[ninterp];
  real dVcdzinterp[ninterp];
  real dH = 4.42563416002 * (67.74/H0);

  vector[nobs] m2s;
  vector[nobs] m2s_jac;

  vector[nobs] dls;
  vector[nobs] mcs;
  vector[nobs] etas;

  real N;

  for (i in 1:nobs) {
    m2s[i] = MMin + f_m2s[i]*m1s[i];
    m2s_jac[i] = m1s[i];
  }

  {
    vector[nobs] mts;

    mts = m1s + m2s;
    etas = m1s .* m2s ./ (mts .* mts);
    for (i in 1:nobs) {
      mcs[i] = mts[i] * etas[i]^(3.0/5.0);
    }
  }

  {
    real state0[1];
    real states[ninterp-1,1];
    real theta[1];

    theta[1] = Om;
    state0[1] = 0.0;

    states = integrate_ode_rk45(dDCdz, state0, 0, zs_out, theta, x_r, x_i);
    dcinterp[1] = 0.0;
    dcinterp[2:] = states[:,1];

    for (i in 1:ninterp) {
      dcinterp[i] = dcinterp[i]*dH;
      dlinterp[i] = dcinterp[i]*(1+zinterp[i]);
      dVcdzinterp[i] = 4.0*pi()*dH*dcinterp[i]*dcinterp[i]/Ez(zinterp[i], Om);
    }
  }

  for (i in 1:nobs) {
    dls[i] = interp1d(zs[i], zinterp, dlinterp);
  }

  {
    real dN[nms, nms, ninterp] = rep_array(0.0, nms, nms, ninterp);

    for (i in 1:nms) {
      real m1_obs = ms[i];
      for (j in 1:i) {
        real m2_obs = ms[j];
        for (k in 1:ninterp) {
          real z = zinterp[k];
          real dl = dlinterp[k];

          real m1 = m1_obs / (1+z);
          real m2 = m2_obs / (1+z);

          real osnr = opt_snrs[i,j] / dl;
          real theta_thresh = 8.0/osnr;
          real f;

          if (theta_thresh > 1.0) {
            f = 0.0;
          } else {
            f = interp1d(theta_thresh, thetas, fabove_thetas);
          }

          if (m2 < MMin || m1 > MMax) {
            dN[i,j,k] = 0.0;
          } else {
            dN[i,j,k] = dNdm1dm2dVdt(m1, m2, z, R0, alpha, gamma, MMin, MMax) * dVcdzinterp[k] * f;
          }
        }
      }
    }

    N = 0.0;
    for (i in 1:nms-1) {
      real dm1 = ms[i+1] - ms[i];
      for (j in 1:i-1) {
        real dm2 = ms[j+1] - ms[j];
        for (k in 1:ninterp-1) {
          real dz = zinterp[k+1] - zinterp[k];

          N = N + 0.125*dm1*dm2*dz*(dN[i,j,k] + dN[i,j,k+1] + dN[i,j+1,k] + dN[i,j+1,k+1] + dN[i+1,j,k] + dN[i+1,j,k+1] + dN[i+1,j+1,k] + dN[i+1,j+1,k+1]);
        }
      }
    }
  }
}

model {
  H0 ~ lognormal(log(70), log(90.0/70.0));
  // Flat prior on Om

  R0 ~ lognormal(log(100), 3);
  // Flat prior on alpha
  // Flat prior on gamma
  // Flat prior on MMax

  // Likelihood for observations
  mc_obs ~ lognormal(log(mcs .* (1.0 + zs)), sigma_mc);
  for (i in 1:nobs) {
    eta_obs[i] ~ normal(etas[i], sigma_eta[i]) T[0,0.25];
  }
  dl_obs ~ lognormal(log(dls), sigma_dl);

  // Population for the observations
  for (i in 1:nobs) {
    real dVdz = interp1d(zs[i], zinterp, dVcdzinterp);
    target += log(dNdm1dm2dVdt(m1s[i], m2s[i], zs[i], R0, alpha, gamma, MMin, MMax)) + log(dVdz) + log(m2s_jac[i]);
  }

  // Poisson normalisation
  target += -N;
}
