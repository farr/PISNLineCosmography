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
    real dNdm1obs;
    real dN;

    dVdz = 4.0*pi()*dH*(dl/(1+z))^2/Ez(z, Om);
    dzddl = 1.0/(dl/(1+z) + dH*(1+z)/Ez(z, Om));

    dNdm1obs = R0*(1-alpha)/(MMax^(1-alpha) - MMin^(1-alpha))*(m1obs/(1+z))^(-alpha)/(1+z);

    dN = dNdm1obs*dVdz*dzddl*(1+z)^(gamma-1);

    return dN;
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
}

data {
  int nobs;
  int nsamp;

  real dMax;
  int ninterp;

  vector[2] m1s_dls[nobs, nsamp];
  cov_matrix[2] kde_bws[nobs];

  int ndet;
  int ngen;
  real Vgen;
  real m1s_det[ndet];
  real dls_det[ndet];

  real smooth_high;
  real smooth_low;
}

transformed data {
  matrix[2,2] chol_kde_bws[nobs];

  real dls_det_sorted[ndet];
  int dls_det_ind[ndet];

  real dls_interp[ninterp];

  real Om;

  real x_r[1];
  int x_i[0];

  Om = 0.3075;
  x_r[1] = Om;

  {
    dls_det_ind = sort_indices_asc(dls_det);
    for (i in 1:ndet) {
      dls_det_sorted[i] = dls_det[dls_det_ind[i]];
    }
  }

  for (i in 1:nobs) {
    chol_kde_bws[i] = cholesky_decompose(kde_bws[i]);
  }

  for (i in 1:ninterp) {
    dls_interp[i] = (i-1.0)/(ninterp-1.0)*dMax;
  }
}

parameters {
  real<lower=0> h;

  real<lower=0> r100;
  real<lower=1.0,upper=10.0> MMin;
  real<lower=30.0, upper=60.0> MMax;
  real<lower=-3, upper=3> alpha;
  real<lower=-5, upper=5> gamma;

  real<lower=MMin, upper=MMax> m1s_obs[nobs];
  real<lower=0, upper=dMax> dls_obs[nobs];
}

transformed parameters {
  real H0 = 100.0*h;
  real dH = 4.42563416002 * (67.74/H0);
  real R0 = 100.0*r100;

  real zs_obs[nobs];

  {
    real zs_interp[ninterp];
    real theta[1];
    real state0[1];
    real states[ninterp-1, 1];

    state0[1] = 0.0;
    theta[1] = dH;

    states = integrate_ode_rk45(dzdDL, state0, 0.0, dls_interp[2:ninterp], theta, x_r, x_i);
    zs_interp[1] = 0.0;
    zs_interp[2:ninterp] = states[:,1];

    for (i in 1:nobs) {
      zs_obs[i] = interp1d(dls_obs[i], dls_interp, zs_interp);
    }
  }
}

model {
  real fs_det[ndet];
  real zs_det[ndet];

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

  h ~ lognormal(log(0.7), 0.5);

  r100 ~ lognormal(0.0, 1.0);
  alpha ~ normal(1.0, 1.0);
  gamma ~ normal(3.0, 1.0);
  MMin ~ normal(5.0, 1.0);
  MMax ~ normal(40.0, 10.0);

  /* Population prior for the observed masses and distances. */
  for (i in 1:nobs) {
    target += log(dNdm1obsdqddl(m1s_obs[i], dls_obs[i], zs_obs[i], R0, alpha, MMin, MMax, gamma, dH, Om));
  }

  /* KDE Approximation to the likelihood. */
  for (i in 1:nobs) {
    vector[2] x;
    vector[2] pts[nsamp] = m1s_dls[i,:];
    matrix[2,2] c = chol_kde_bws[i];
    real log_wts[nsamp];

    x[1] = m1s_obs[i];
    x[2] = dls_obs[i];

    for (j in 1:nsamp) {
        log_wts[j] = multi_normal_cholesky_lpdf(pts[j] | x, c);
    }
    target += log_sum_exp(log_wts) - log(nsamp); /* KDE(x) = mean(N(pt | x, c)) */
  }

  // Poisson norm; we marginalise over the uncertainty in the Monte-Carlo
  // integral.  Here we smooth the sharp cutoff of the distribution
  // exponentially at both ends with the smoothing scale given in the data block.
  for (i in 1:ndet) {
    real m1 = m1s_det[i]/(1+zs_det[i]);
    real smoothing = normal_cdf(m1, MMin, smooth_low)*(1-normal_cdf(m1, MMax, smooth_high));
    fs_det[i] = smoothing*dNdm1obsdqddl(m1s_det[i], dls_det[i], zs_det[i], R0, alpha, MMin, MMax, gamma, dH, Om);
  }

  {
    real mu;
    real sigma;

    mu = Vgen/ngen*sum(fs_det);
    sigma = Vgen/ngen*sqrt(ndet)*sd(fs_det);

    if (sigma/mu > 1.0/sqrt(nobs)) reject("cannot estimate selection integral reliably");

    target += -mu;
  }
}
