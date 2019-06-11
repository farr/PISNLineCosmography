functions {
  int searchsorted(real x, real[] xs) {
    int n = size(xs);
    int l;
    int h;

    if ((x < xs[1]) || (x > xs[n])) reject("cannot search outside bounds");

    l = 1;
    h = n;
    while (h-l > 1) {
      int m = l + (h-l)/2;

      if (x > xs[m]) {
        l = m;
      } else {
        h = m;
      }
    }

    return h;
  }

  real interp1d(real x, real[] xs, real[] ys) {
    int i = searchsorted(x, xs);
    real xl = xs[i-1];
    real xh = xs[i];
    real yl = ys[i-1];
    real yh = ys[i];

    real r = (x-xl)/(xh-xl);

    return r*yh + (1.0-r)*yl;
  }

  real[] cumtrapz(real[] xs, real[] ys) {
    int n = size(xs);
    real term[n-1];
    real zs[n];

    for (i in 1:n-1) {
      term[i] = 0.5*(xs[i+1]-xs[i])*(ys[i+1]+ys[i]);
    }

    zs[1] = 0.0;
    zs[2:] = cumulative_sum(term);

    return zs;
  }

  real trapz(real[] xs, real[] ys) {
    int n = size(xs);
    vector[n] vxs = to_vector(xs);
    vector[n] vys = to_vector(ys);

    return sum(0.5 * (vxs[2:] - vxs[:n-1]) .* (vys[2:] + vys[:n-1]));
  }

  real Ez(real z, real Om, real z_p, real w0) {
    real opz = 1.0 + z;
    real opz2 = opz*opz;
    real opz3 = opz2*opz;

    return sqrt(Om*opz3 + (1.0-Om)*opz^(3.0*(1.0 + w0)));
  }

  real dzddL(real dl, real z, real dH, real Om, real z_p, real w0) {
    real opz = 1.0 + z;
    return 1.0/(dl/opz + opz*dH/Ez(z, Om, z_p, w0));
  }

  real[] log_dNdm1dm2dzdt_norm(real[] m1s, real[] m2s, real[] dls, real[] zs, real MMin, real MMax, real smin, real smax, real alpha, real beta, real gamma, real dH, real Om, real z_p, real w0) {
    int n = size(m1s);
    real log_4pi_dH = log(4.0*pi()*dH);
    real log_dNs[n];
    real m0 = 30.0;
    real logm0 = log(m0);
    real logMMin = log(MMin);
    real logMMax = log(MMax);

    for (i in 1:n) {
      real sl1;
      real sh1;
      real sl2;
      real sh2;
      real logm1 = log(m1s[i]);
      real logm2 = log(m2s[i]);

      if (logm1 < logMMin - 7*smin) {
        sl1 = negative_infinity();
      } else {
        sl1 = normal_lcdf(logm1 | logMMin, smin);
      }

      if (logm1 > logMMax + 7*smax) {
        sh1 = negative_infinity();
      } else {
        sh1 = normal_lccdf(logm1 | logMMax, smax);
      }

      if (logm2 < logMMin - 7*smin) {
        sl2 = negative_infinity();
      } else {
        sl2 = normal_lcdf(logm2 | logMMin, smin);
      }

      if (logm2 > logMMax + 7*smax) {
        sh2 = negative_infinity();
      } else {
        sh2 = normal_lccdf(logm2 | logMMax, smax);
      }

      log_dNs[i] = -alpha*log(m1s[i]/m0) + beta*log(m2s[i]/m0) - 2.0*log(m0) + (gamma-1)*log1p(zs[i]) + log_4pi_dH + 2.0*log(dls[i]/(1+zs[i])) - log(Ez(zs[i], Om, z_p, w0)) + sl1 + sh1 + sl2 + sh2;
    }

    return log_dNs;
  }

  real[] dls_of_zs(real[] zs, real dH, real Om, real z_p, real w0) {
    int n = size(zs);
    real ddcdz[n];
    real dcs[n];
    real dls[n];

    for (i in 1:n) {
      ddcdz[i] = dH/Ez(zs[i], Om, z_p, w0);
    }
    dcs = cumtrapz(zs, ddcdz);

    for (i in 1:n) {
      dls[i] = dcs[i]*(1+zs[i]);
    }

    return dls;
  }
}

data {
  int nobs;
  int nsel;

  int ngmm;

  real Tobs;
  int N_gen;

  real weights[nobs, ngmm];
  vector[3] means[nobs, ngmm];
  matrix[3,3] covs[nobs, ngmm];

  vector[3] mu_samp[nobs];
  matrix[3,3] chol_cov_samp[nobs];

  real m1sel[nsel];
  real m2sel[nsel];
  real dlsel[nsel];
  real log_wtsel[nsel];

  int ninterp;
  real zinterp[ninterp];

  int cosmo_prior;

  real d_p;
  real z_p;
}

transformed data {
  real smooth_min = 0.1;
  real MMin = 5.0;

  real a_p = 1.0/(1+z_p);
  real zmax = zinterp[ninterp];
  matrix[3,3] chol_covs[nobs, ngmm];
  real log_weights[nobs, ngmm];

  for (i in 1:nobs) {
    for (j in 1:ngmm) {
      chol_covs[i,j] = cholesky_decompose(covs[i,j]);
      log_weights[i,j] = log(weights[i,j]);
    }
  }
}

parameters {
  real<lower=50, upper=200> H_p;
  real<lower=0,upper=1> Om;
  real<lower=-2,upper=0> w0;

  real<lower=50, upper=250> MMax_d_p;
  real<lower=MMax_d_p*1.02, upper=2*MMax_d_p> MMax_d_p_2sigma;
  real<lower=-5, upper=5> alpha;
  real<lower=-3, upper=3> beta;
  real<lower=-1, upper=7> gamma;

  vector[3] xs[nobs];
}

transformed parameters {
  real MMax;
  real logdMMaxdMMax_d_p;
  real smooth_max = (log(MMax_d_p_2sigma)-log(MMax_d_p))/2.0;
  real H0 = H_p / Ez(z_p, Om, z_p, w0);
  real Omh2 = Om*(H0/100)^2;
  real dH = 4.42563416002 * (67.74/H0);
  real mu_det;
  real neff_det;
  real m1s[nobs];
  real m2s[nobs];
  real zs[nobs];
  real dls[nobs];

  {
    real dlinterp[ninterp];
    real zsel[nsel];
    real log_dN_m_unwt[nsel];
    real log_dN[nsel];
    real log_dN2[nsel];
    real m1sel_source[nsel];
    real m2sel_source[nsel];

    real Nsum;
    real N2sum;

    real sigma_rel2;
    real sigma_rel;

    real mm_factor;

    dlinterp = dls_of_zs(zinterp, dH, Om, z_p, w0);

    mm_factor = interp1d(d_p, dlinterp, zinterp);
    MMax = MMax_d_p / (1 + mm_factor);
    logdMMaxdMMax_d_p = -log1p(mm_factor);

    for (i in 1:nobs) {
      vector[3] y;
      real m1det;
      real q;

      y = mu_samp[i] + chol_cov_samp[i] * xs[i];

      m1det = exp(y[1]);
      q = inv_logit(y[2]);
      dls[i] = exp(y[3]);
      zs[i] = interp1d(dls[i], dlinterp, zinterp);
      m1s[i] = m1det/(1+zs[i]);
      m2s[i] = q*m1s[i];
    }

    for (i in 1:nsel) {
      zsel[i] = interp1d(dlsel[i], dlinterp, zinterp);
      m1sel_source[i] = m1sel[i]/(1+zsel[i]);
      m2sel_source[i] = m2sel[i]/(1+zsel[i]);
    }

    log_dN_m_unwt = log_dNdm1dm2dzdt_norm(m1sel_source, m2sel_source, dlsel, zsel, MMin, MMax, smooth_min, smooth_max, alpha, beta, gamma, dH, Om, z_p, w0);
    for (i in 1:nsel) {
      log_dN[i] = log_dN_m_unwt[i] - 2.0*log1p(zsel[i]) + log(dzddL(dlsel[i], zsel[i], dH, Om, z_p, w0)) - log_wtsel[i];
    }

    for (i in 1:nsel) {
      log_dN2[i] = log_dN[i]*2.0;
    }

    Nsum = exp(log_sum_exp(log_dN));
    N2sum = exp(log_sum_exp(log_dN2));

    mu_det = Tobs/N_gen*Nsum;

    sigma_rel2 = N2sum/(Nsum*Nsum) - 1.0/N_gen;
    sigma_rel = sqrt(sigma_rel2);

    neff_det = 1.0/sigma_rel2;
  }
}

model {
  real log_pop_nojac[nobs];
  real log_pop_jac[nobs];

  /* The code below leaves the posterior un-modified as long as neff_det >
     5*nobs.  For 4*nobs < neff_det < 5*nobs, it introduces a factor f in
     [0,1] that tapers the posterior smoothly to zero at neff_det = 4*nobs, where the
     analytic marginalization over the selection integral breaks down. */
  if (neff_det > 5*nobs) {
    /* Totally safe */
  } else if (neff_det > 4*nobs) {
    /* Smoothly cut off the log-density: scale of cutoff is 0.1*nobs starting at
    /* 5*nobs down to 4*nobs */
    real dx = log(neff_det - 4*nobs) - log(nobs);
    target += -0.5*dx*dx/0.01;
  } else {
    reject("need more samples for selection integral");
  }

  /* Since we sample in MMax(d_p), but want a prior on MMax, we need a Jacobian
  /* factor, as below. */
  MMax ~ normal(50, 15);
  target += logdMMaxdMMax_d_p;

  /* We sample in the 2-sigma mass upper limit, and so need
  /* d(smooth_max)/d(M2Sigma) = 1/M2Sigma Jacobian. */
  smooth_max ~ lognormal(log(0.1), 1);
  target += -log(MMax_d_p_2sigma);

  if (cosmo_prior == 0) {
    /* Prior on H0, sample in Hp => Jacobian d(H0)/d(Hp) = 1/E(z) */
    H0 ~ normal(70, 15);
    target += -log(Ez(z_p, Om, z_p, w0));

    Om ~ normal(0.3, 0.15);
  } else {
    /* See note above on Jacobian. */
    H0 ~ normal(67.74, 0.6774);
    target += -log(Ez(z_p, Om, z_p, w0));

    /* Prior on Om*h^2 from CMB, sample in Om => need Jacobian: d(Om*h^2)/d(Om) = h^2. */
    Omh2 ~ normal(0.02225+0.1198, sqrt(0.00016^2 + 0.0015^2));
    target += 2.0*log(H0/100.0);
  }
  w0 ~ normal(-1, 0.5);

  alpha ~ normal(1, 2);
  beta ~ normal(0, 2);
  gamma ~ normal(1.5, 1.5);

  /* Population prior on m1,m2,dl,z; no smoothing required, since we enforce MMin < m2 < m1 < MMax */
  log_pop_nojac = log_dNdm1dm2dzdt_norm(m1s, m2s, dls, zs, MMin, MMax, smooth_min, smooth_max, alpha, beta, gamma, dH, Om, z_p, w0);
  /* We have dN/d(m1)d(m2)d(z)d(t_det) but sample in m2_fracs, not m2, so need d(m2)/d(m2_frac) = m1 - MMin */
  for (i in 1:nobs) {
    real q = m2s[i]/m1s[i];
    log_pop_jac[i] = log_pop_nojac[i] + 2.0*log(m1s[i]) + log(q) + log1p(-q) + log(dzddL(dls[i], zs[i], dH, Om, z_p, w0)) + log(dls[i]) + sum(log(diagonal(chol_cov_samp[i])));
  }
  target += sum(log_pop_jac);

  /* Likelihood */
  for (i in 1:nobs) {
    real logps[ngmm];
    vector[3] x = to_vector({m1s[i]*(1+zs[i]),
                             m2s[i]*(1+zs[i]),
                             dls[i]});

    for (j in 1:ngmm) {
      logps[j] = multi_normal_cholesky_lpdf(means[i,j] | x, chol_covs[i,j]) + log_weights[i,j];
    }

    target += log_sum_exp(logps);
  }

  // Normalization term
  target += -(nobs+1)*log(mu_det) + nobs*(3 + nobs)/(2*neff_det);
}

generated quantities {
  real R0_30;

  {
    real mu_R0 = nobs/mu_det*(1.0 + nobs/neff_det);
    real sigma_R0 = sqrt(nobs)/mu_det*(1.0 + 1.5*nobs/neff_det);

    R0_30 = normal_rng(mu_R0, sigma_R0);
  }
}
