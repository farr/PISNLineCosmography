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

  real m1s[nobs, nsamp];
  real dls[nobs, nsamp];

  int ndet;
  int ngen;
  real Vgen;
  real m1s_det[ndet];
  real dls_det[ndet];

  int ninterp;

  real mass_smoothing_scale_low;
  real mass_smoothing_scale_high;
}

transformed data {
  real dls_det_sorted[ndet];
  int dls_det_ind[ndet];

  vector[2] kde_pts[nobs, nsamp];
  matrix[2,2] chol_kde_cov[nobs];

  real dl_max;
  real dls_interp[ninterp];

  real Om;

  real x_r[1];
  int x_i[0];

  Om = 0.3075;
  x_r[1] = Om;

  {
    dl_max = 2.0*max({max(dls_det), max(to_array_1d(dls))}); // Maximum dL is double largest sample

    for (i in 1:ninterp) {
      dls_interp[i] = (i-1.0)/(ninterp-1.0)*dl_max;
    }
  }

  {
    dls_det_ind = sort_indices_asc(dls_det);
    for (i in 1:ndet) {
      dls_det_sorted[i] = dls_det[dls_det_ind[i]];
    }
  }

  {
    for (i in 1:nobs) {
      real mu_dl = 0.0;
      real mu_m = 0.0;
      matrix[2,2] c;

      for (j in 1:nsamp) {
        mu_dl = mu_dl + dls[i,j];
        mu_m = mu_m + m1s[i,j];

        kde_pts[i,j][1] = m1s[i,j];
        kde_pts[i,j][2] = dls[i,j];
      }
      mu_dl = mu_dl / nsamp;
      mu_m = mu_m / nsamp;

      c = rep_matrix(0.0, 2, 2);
      for (j in 1:nsamp) {
        c[1,1] = c[1,1] + square(m1s[i,j] - mu_m);
        c[2,2] = c[2,2] + square(dls[i,j] - mu_dl);
        c[1,2] = c[1,2] + (m1s[i,j] - mu_m) * (dls[i,j] - mu_dl);
        c[2,1] = c[1,2];
      }
      c = c / nsamp^(4.0/3.0); // 1.0 + 2/(4 + d) = 4/3; 1.0 is to compute the mean, the 2/(4+d) is Silverman bandwidth rule
      chol_kde_cov[i] = cholesky_decompose(c);
    }
  }
}

parameters {
  real<lower=0> h;

  real<lower=0> R0;
  real<lower=1.0,upper=10.0> MMin;
  real<lower=30.0, upper=60.0> MMax;
  real<lower=-3, upper=3> alpha;
  real<lower=-5, upper=5> gamma;

  real<lower=MMin, upper=MMax> m1s_true[nobs];
  real<lower=0, upper=dl_max> dls_true[nobs];
}

transformed parameters {
  real H0 = 100.0*h;
  real dH = 4.42563416002 * (67.74/H0);
}

model {
  real fs_det[ndet];
  real zs_true[nobs];

  real zs_det[ndet];

  {
    real zs_interp[ninterp];
    real theta[1];
    real state0[1];
    real states[ninterp-1, 1];

    theta[1] = dH;
    state0[1] = 0.0;

    states = integrate_ode_rk45(dzdDL, state0, 0.0, dls_interp[2:], theta, x_r, x_i);
    zs_interp[1] = 0.0;
    zs_interp[2:] = states[:,1];

    for (i in 1:nobs) {
        zs_true[i] = interp1d(dls_true[i], dls_interp, zs_interp);
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

  /* Population Distribution */
  for (i in 1:nobs) {
    target += log(dNdm1obsdqddl(m1s_true[i], dls_true[i], zs_true[i], R0, alpha, MMin, MMax, gamma, dH, Om));
  }

  /* GW data likelihood */
  for (i in 1:nobs) {
    real fs[nsamp];
    vector[2] p;

    p[1] = m1s_true[i];
    p[2] = dls_true[i];

    for (j in 1:nsamp) {
      fs[j] = multi_normal_cholesky_lpdf(kde_pts[i,j] | p, chol_kde_cov[i]);
    }
    target += log_sum_exp(fs) - log(nsamp); /* Mean of the KDE kernels. */
  }

  // Poisson norm; we marginalise over the uncertainty in the Monte-Carlo
  // integral.  Here we smooth the sharp cutoff of the distribution
  // exponentially at both ends with the smoothing scale given in the data block.
  for (i in 1:ndet) {
    fs_det[i] = dNdm1obsdqddl(m1s_det[i], dls_det[i], zs_det[i], R0, alpha, MMin, MMax, gamma, dH, Om);

    fs_det[i] = fs_det[i] * normal_cdf(m1s_det[i], MMin, mass_smoothing_scale_low) * (1.0 - normal_cdf(m1s_det[i], MMax, mass_smoothing_scale_high));
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
