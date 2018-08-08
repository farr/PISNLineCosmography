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

  real interp_1d(real x, real[] xs, real[] ys) {
    int n = size(xs);

    int j = bisect_index(x, xs);

    if ((j <= 1) || (j >= n)) reject("cannot interpolate out of bounds");

    {
      real x0;
      real x1;
      real y0;
      real y1;
      real r;

      x0 = xs[j-1];
      x1 = xs[j];
      y0 = ys[j-1];
      y1 = ys[j];

      r = (x-x0)/(x1-x0);

      return (1.0-r)*y0 + r*y1;
    }
  }

  real Ez(real z, real Om) {
    real opz = 1.0 + z;
    real opz3 = opz*opz*opz;

    return sqrt(opz3*Om + (1.0-Om));
  }

  real dzddl(real dl, real z, real dH, real Om) {
    real dc = dl/(1.0+z);

    return 1.0/(dc + (1.0+z)*dH/Ez(z, Om));
  }

  real dNdm1obsddl(real m1_obs, real dl_obs, real z_obs, real R0, real MMin, real MMax, real alpha, real gamma, real dH, real Om) {
    real m1 = m1_obs / (1.0 + z_obs);
    real dc = dl_obs / (1.0 + z_obs);

    real dNdm1dV = R0 * m1^(-alpha) * (1.0 - alpha) / (MMax^(1.0-alpha) - MMin^(1.0-alpha));
    real dm1dm1_obs = 1.0/(1.0 + z_obs);
    real dVdz = 4.0*pi()*dc*dc*dH/Ez(z_obs, Om);
    real dzddl_ = dzddl(dl_obs, z_obs, dH, Om);

    return dNdm1dV*dm1dm1_obs*dVdz*dzddl_;
  }
}

data {
  int nobs; /* Number of observations */
  int nsamp; /* Number of samples per observation */
  int nsel; /* Number of detected injected objects. */

  vector[2] m1s_dls[nobs, nsamp]; /* Sample points for each event */
  cov_matrix[2] bws[nobs]; /* Bandwidth covariance matrices (pre-computed, presumably using something like Scott rule smoothing) */

  real m1s_sel[nsel]; /* Observed masses of injected systems */
  real dls_sel[nsel]; /* Luminosity distances of injected systems */

  real Vgen; /* Total injected volume (mass x luminosity distance) */
  int ngen; /* Total number of injections made. */

  int ninterp; /* Number of interpolation points for cosmography */
  real zs_interp[ninterp]; /* Redshifts */
  real dlu_interp[ninterp]; /* Unit-less luminosity distances (dL(z)/dH); will be re-scaled by dH(H0) before interpolation. */

  real smooth_low; /* Smoothing scale at low mass */
  real smooth_high;
}

transformed data {
  matrix[2,2] chol_bws[nobs];

  real Om = 0.3075; /* From Planck15 astropy.cosmology */

  real dlmax;
  real hmax;

  for (i in 1:nobs) {
    chol_bws[i] = cholesky_decompose(bws[i]);
  }

  {
    real dHmin;
    real dlselmax = max(dls_sel);
    real dls_obs[nobs, nsamp];

    for (i in 1:nobs) {
      for (j in 1:nsamp) {
        dls_obs[i,j] = m1s_dls[i,j][2];
      }
    }

    dlmax = max(to_array_1d(dls_obs));
    dlmax = (dlselmax > dlmax ? dlselmax : dlmax);
    dlmax = dlmax * 1.1; /* Make it a bit bigger. */

    dHmin = (dlmax / dlu_interp[ninterp]);

    hmax = 4.42563416002/dHmin * 0.6774;
  }
}

parameters {
  real<lower=0> r; /* R(z=0) / (100 Gpc^-3 yr^-1) */
  real<lower=0, upper=hmax> h; /* H0 / (100 km/s/Mpc) */
  real<lower=1,upper=10> MMin; /* MSun */
  real<lower=30,upper=60> MMax; /* MSun */
  real alpha; /* m1 power law slope is m1^(-alpha) */
  real gamma; /* (1+z) power law in redshift evolution: (1+z)^gamma */
  real<lower=MMin,upper=MMax> m1s_true[nobs];
  real<lower=0,upper=dlmax> dls_true[nobs];
}

transformed parameters {
  real H0 = 100.0*h;
  real R0 = 100.0*r;

  real dH = 4.42563416002 * (67.74/H0);

  real zs_true[nobs];

  {
    real dl_interp[ninterp];

    for (i in 1:ninterp) {
      dl_interp[i] = dH*dlu_interp[i];
    }

    for (i in 1:nobs) {
      zs_true[i] = interp_1d(dls_true[i], dl_interp, zs_interp);
    }
  }
}

model {
  /* Priors on population parameters */
  h ~ lognormal(log(0.7), 1);
  r ~ lognormal(log(1), 1);

  MMin ~ normal(5, 2);
  MMax ~ normal(40, 10);

  alpha ~ normal(1, 1);
  gamma ~ normal(3, 2);

  /* Population Prior on Masses Distances. */
  for (i in 1:nobs) {
    target += log(dNdm1obsddl(m1s_true[i]*(1.0+zs_true[i]), dls_true[i], zs_true[i], R0, MMin, MMax, alpha, gamma, dH, Om));
    /* Jacobian for sampling in m1s_true. */
    target += log1p(zs_true[i]);
  }

  /* Likelihood Terms */
  for (i in 1:nobs) {
    real logps[nsamp];
    vector[2] mu;

    mu[1] = m1s_true[i]*(1.0+zs_true[i]);
    mu[2] = dls_true[i];

    for (j in 1:nsamp) {
      logps[j] = multi_normal_cholesky_lpdf(m1s_dls[i,j] | mu, chol_bws[i]);
    }

    target += log_sum_exp(logps) - log(nsamp);
  }

  /* Poisson Normalisation */
  {
    real dNs[nsel];
    real dls_interp[ninterp];

    for (i in 1:ninterp) {
      dls_interp[i] = dH*dlu_interp[i];
    }

    for (i in 1:nsel) {
      real zsel;
      real m1;

      zsel = interp_1d(dls_sel[i], dls_interp, zs_interp);

      m1 = m1s_sel[i] / (1.0 + zsel);

      dNs[i] = dNdm1obsddl(m1s_sel[i], dls_sel[i], zsel, R0, MMin, MMax, alpha, gamma, dH, Om);
      dNs[i] = dNs[i]*normal_cdf(m1, MMin, smooth_low)*(1.0-normal_cdf(m1, MMax, smooth_high));
    }

    target += -Vgen/ngen*sum(dNs);
  }
}
