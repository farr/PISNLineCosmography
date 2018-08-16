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

  real[] dzddl_system(real dl, real[] state, real[] theta, real[] x_r, int[] x_i) {
    real z = state[1];
    real dzddl_[1];
    real dH = theta[1];
    real Om = x_r[1];

    dzddl_[1] = dzddl(dl, z, dH, Om);

    return dzddl_;
  }

  real dNdm1obsdm2obsddl(real m1_obs, real m2_obs, real dl_obs, real z_obs, real R0, real MMin, real MMax, real alpha, real beta, real gamma, real dH, real Om) {
    real m1 = m1_obs / (1.0 + z_obs);
    real m2 = m2_obs / (1.0 + z_obs);
    real dc = dl_obs / (1.0 + z_obs);

    real dNdm1dm2dV = R0 * m1^(-alpha) * m2^(beta) * (1.0 - alpha) * (1.0 + beta) / (MMax^(1.0-alpha) - MMin^(1.0-alpha)) / (m1^(beta+1.0) - MMin^(beta+1.0)) * (1 + z_obs)^(gamma-1);
    real dm1dm1_obs = 1.0/(1.0 + z_obs);
    real dm2dm2_obs = 1.0/(1.0 + z_obs);
    real dVdz = 4.0*pi()*dc*dc*dH/Ez(z_obs, Om);
    real dzddl_ = dzddl(dl_obs, z_obs, dH, Om);

    return dNdm1dm2dV*dm1dm1_obs*dm2dm2_obs*dVdz*dzddl_;
  }
}

data {
  int nobs; /* Number of observations */
  int nsamp; /* Number of samples per observation */
  int nsel; /* Number of detected injected objects. */

  vector[3] m1s_m2s_dls[nobs, nsamp]; /* Sample points for each event */
  cov_matrix[3] bws[nobs]; /* Bandwidth covariance matrices (pre-computed, presumably using something like Scott rule smoothing) */

  real m1s_sel[nsel]; /* Observed masses of injected & detected systems */
  real m2s_sel[nsel]; /* Observed masses of injected & detected systems */
  real dls_sel[nsel]; /* Luminosity distances of injected systems */

  real Vgen; /* Total injected volume (mass x luminosity distance) */
  int ngen; /* Total number of injections made. */

  int ninterp; /* Number of interpolation points for cosmography */

  real smooth_low; /* Smoothing scale at low mass */
  real smooth_high; /* ... at high mass. */

  real dl_max; /* Maximum dL. */
}

transformed data {
  matrix[3,3] chol_bws[nobs];
  real dl_interp[ninterp];

  real Om = 0.3075; /* From Planck15 astropy.cosmology */

  real x_r[1];
  int x_i[0];

  real hmax;

  x_r[1] = Om;

  for (i in 1:nobs) {
    chol_bws[i] = cholesky_decompose(bws[i]);
  }

  for (i in 1:ninterp) {
    dl_interp[i] = (i-1.0)/(ninterp-1.0)*dl_max;
  }
}

parameters {
  real<lower=0> r; /* R(z=0) / (100 Gpc^-3 yr^-1) */
  real<lower=0> h; /* H0 / (100 km/s/Mpc) */
  real<lower=1,upper=10> MMin; /* MSun */
  real<lower=30,upper=60> MMax; /* MSun */
  real alpha; /* m1 power law slope is m1^(-alpha) */
  real beta; /* m2 power law slope is p(m2 | m1) ~ m2^beta */
  real gamma; /* (1+z) power law in redshift evolution: (1+z)^gamma */
  real<lower=MMin,upper=MMax> m1s_true[nobs]; /* True masses */
  real<lower=0,upper=1> m2s_frac[nobs]; /* Fraction of m2 range. */
  real<lower=0,upper=dl_max> dls_true[nobs]; /* True lum. distances */
}

transformed parameters {
  real H0 = 100.0*h;
  real R0 = 100.0*r;

  real dH = 4.42563416002 * (67.74/H0);

  real m2s_true[nobs];
  real zs_true[nobs];
  real Nex;

  for (i in 1:nobs) {
    m2s_true[i] = MMin + m2s_frac[i]*(m1s_true[i]-MMin);
  }

  {
    real zs_interp[ninterp];
    real theta[1];
    real state0[1];
    real states[ninterp-1,1];

    theta[1] = dH;
    state0[1] = 0.0;

    states = integrate_ode_rk45(dzddl_system, state0, 0.0, dl_interp[2:], theta, x_r, x_i);

    zs_interp[1] = 0.0;
    zs_interp[2:] = states[:,1];

    for (i in 1:nobs) {
      zs_true[i] = interp_1d(dls_true[i], dl_interp, zs_interp);
    }

    {
      real dNs[nsel];
      real mu;
      real sigma;

      for (i in 1:nsel) {
        real zsel;
        real m1;
        real m2;

        zsel = interp_1d(dls_sel[i], dl_interp, zs_interp);
        m1 = m1s_sel[i] / (1+zsel);
        m2 = m2s_sel[i] / (1+zsel);

        dNs[i] = dNdm1obsdm2obsddl(m1s_sel[i], m2s_sel[i], dls_sel[i], zsel, R0, MMin, MMax, alpha, beta, gamma, dH, Om);
        dNs[i] = dNs[i]*normal_cdf(m2, MMin, smooth_low)*(1.0-normal_cdf(m1, MMax, smooth_high));
      }

      Nex = Vgen/ngen*sum(dNs);
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
  beta ~ normal(0, 2);
  gamma ~ normal(3, 2);

  /* Population Prior on Masses Distances. */
  for (i in 1:nobs) {
    target += log(dNdm1obsdm2obsddl(m1s_true[i]*(1.0+zs_true[i]), m2s_true[i]*(1.0+zs_true[i]), dls_true[i], zs_true[i], R0, MMin, MMax, alpha, beta, gamma, dH, Om));
    /* Jacobian for obs -> true: d(m_obs)/d(m_true) = 1+z. */
    target += 2.0*log1p(zs_true[i]);
    /* Jacobian from m2s_true to m2s_frac. */
    target += log(m1s_true[i] - MMin);
  }

  /* Likelihood Terms */
  for (i in 1:nobs) {
    real logps[nsamp];
    vector[3] mu;

    mu[1] = m1s_true[i]*(1.0+zs_true[i]);
    mu[2] = m2s_true[i]*(1.0+zs_true[i]);
    mu[3] = dls_true[i];

    for (j in 1:nsamp) {
      logps[j] = multi_normal_cholesky_lpdf(m1s_m2s_dls[i,j] | mu, chol_bws[i]);
    }

    target += log_sum_exp(logps) - log(nsamp);
  }

  /* Poisson Normalisation */
  target += -Nex;
}
