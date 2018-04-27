functions {
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

  real trapz(real[] xs, real[] ys) {
    int n = size(xs);
    real s;

    if (size(ys) != n) reject("trapz: array sizes must match")

    s = 0.0;
    for (i in 1:n-1) {
      s = s + 0.5*(xs[i+1]-xs[i])*(ys[i] + ys[i+1]);
    }

    return s;
  }
}

data {
  int nobs;
  vector[nobs] mc_obs;
  vector[nobs] dl_obs;

  vector[nobs] sigma_mc;
  vector[nobs] sigma_dl;

  real zmax;
  int ninterp;
}

transformed data {
  real zinterp[ninterp];
  int npop = 10;
  real zpop[npop];
  real x_r[0];
  int x_i[0];

  for (i in 1:ninterp) {
    zinterp[i] = (i-1.0)*zmax/(ninterp-1.0);
  }

  for (i in 1:npop) {
    zpop[i] = (i-1.0)*zmax/(npop - 1.0);
  }
}

parameters {
  real<lower=0> H0;
  real<lower=0,upper=1> om;

  real<lower=0> mc_max;

  vector<lower=0,upper=mc_max>[nobs] mc;
  vector<lower=0,upper=zmax>[nobs] zs;

  real<lower=0> dNdz[npop];
}

transformed parameters {
  real dH = 4.42563416002 * (67.74/H0);
  vector[nobs] dl;
  vector[nobs] dc;

  {
    real dcinterp[ninterp];
    real states[ninterp-1,1];
    real istate[1];
    real theta[1];

    istate[1] = 0.0;
    theta[1] = om;

    states = integrate_ode_rk45(dDCdz, istate, 0.0, zinterp[2:], theta, x_r, x_i);
    dcinterp[1] = 0.0;
    dcinterp[2:] = states[:,1];

    for (i in 1:nobs) {
      dc[i] = dH*interp1d(zs[i], zinterp, dcinterp);
      dl[i] = (1.0+zs[i])*dc[i];
    }
  }
}

model {
  H0 ~ lognormal(log(70), log(80.0/70.0));
  om ~ normal(0.3, 0.1);

  mc_max ~ normal(80.0*(0.25^(3.0/5.0)), 15);

  /* Flat prior on mc */
  target += -nobs*log(mc_max);

  /* Volumetric prior on z */
  for (i in 1:nobs) {
    target += log(interp1d(zs[i], zpop, dNdz));
  }
  /* exp(-N) */
  target += -trapz(zpop, dNdz);

  mc_obs ~ lognormal(log(mc .* (1.0 + zs)), sigma_mc);
  dl_obs ~ lognormal(log(dl), sigma_dl);
}
