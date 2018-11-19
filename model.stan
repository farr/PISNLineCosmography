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
    real terms[n-1];

    for (i in 1:n-1) {
      terms[i] = 0.5*(xs[i+1]-xs[i])*(ys[i+1]+ys[i]);
    }

    return sum(terms);
  }

  real softened_power_law_logpdf_unnorm(real x, real alpha, real xmin, real xmax, real sigma_min, real sigma_max) {
    real logx = log(x);
    real pl = alpha*logx;

    if (x < xmin) {
      return pl - 0.5*((logx - log(xmin))/sigma_min)^2;
    } else if (x > xmax) {
      return pl - 0.5*((logx - log(xmax))/sigma_max)^2;
    } else {
      return pl;
    }
  }

  real Ez(real z, real Om, real w) {
    real opz = 1.0 + z;
    real opz2 = opz*opz;
    real opz3 = opz2*opz;

    return sqrt(Om*opz3 + (1.0-Om)*opz^(3*(1+w)));
  }

  real dzddL(real dl, real z, real dH, real Om, real w) {
    real opz = 1.0 + z;
    return 1.0/(dl/opz + opz*dH/Ez(z, Om, w));
  }

  real[] log_dNdm1dm2ddldt(real[] m1s, real[] m2s, real[] dls, real[] zs, real R0, real MMin, real MMax, real alpha, real beta, real gamma, real dH, real Om, real w, real smooth_low, real smooth_high, real[] ms_norm) {
    int nm = size(ms_norm);
    int n = size(m1s);
    real pms_alpha[nm];
    real pms_beta[nm];
    real cum_beta[nm];
    real log_norm_alpha;
    real log_dN[n];

    for (i in 1:nm) {
      pms_alpha[i] = exp(softened_power_law_logpdf_unnorm(ms_norm[i], -alpha, MMin, MMax, smooth_low, smooth_high));
      pms_beta[i] = exp(softened_power_law_logpdf_unnorm(ms_norm[i], beta, MMin, MMax, smooth_low, smooth_high));
    }

    cum_beta = cumtrapz(ms_norm, pms_beta);

    log_norm_alpha = log(trapz(ms_norm, pms_alpha));

    for (i in 1:n) {
      real log_norm_beta = log(interp1d(m1s[i], ms_norm, cum_beta));
      real log_dNdm1dm2dVdt = softened_power_law_logpdf_unnorm(m1s[i], -alpha, MMin, MMax, smooth_low, smooth_high) + softened_power_law_logpdf_unnorm(m2s[i], beta, MMin, MMax, smooth_low, smooth_high) - log_norm_alpha - log_norm_beta + (gamma-1)*log1p(zs[i]);
      real log_dVdz = log(4.0*pi()) + 2.0*log(dls[i]/(1+zs[i])) + log(dH) - log(Ez(zs[i], Om, w));
      real log_dzddl = log(dzddL(dls[i], zs[i], dH, Om, w));

      log_dN[i] = log_dNdm1dm2dVdt + log_dVdz + log_dzddl;
    }

    return log_dN;
  }

  real[] dls_of_zs(real[] zs, real dH, real Om, real w) {
    int n = size(zs);
    real ddcdz[n];
    real dcs[n];
    real dls[n];

    for (i in 1:n) {
      ddcdz[i] = dH/Ez(zs[i], Om, w);
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
  int nsamp;
  int nsel;
  int ninterp;
  int nnorm;

  real Tobs;
  int N_gen;

  real m1obs[nobs, nsamp];
  real m2obs[nobs, nsamp];
  real dlobs[nobs, nsamp];

  matrix[3,3] bws[nobs];

  real m1sel[nsel];
  real m2sel[nsel];
  real dlsel[nsel];
  real wtsel[nsel];

  real zinterp[ninterp];

  real dl_max;

  real ms_norm[nnorm];
}

transformed data {
  real absolute_MMin = 3.0;
  real absolute_MMax = 100.0;
  vector[3] pts[nobs, nsamp];
  matrix[3,3] chol_bws[nobs];
  real log_wtsel[nsel];

  for (i in 1:nobs) {
    for (j in 1:nsamp) {
      pts[i,j][1] = m1obs[i,j];
      pts[i,j][2] = m2obs[i,j];
      pts[i,j][3] = dlobs[i,j];
    }

    chol_bws[i] = cholesky_decompose(bws[i]);
  }

  for (i in 1:nsel) {
    log_wtsel[i] = log(wtsel[i]);
  }
}

parameters {
  real<lower=0> H0;
  real<lower=0,upper=1> Om;
  real w;

  real<lower=0> R0;
  real<lower=3, upper=10> MMin;
  real<lower=30, upper=100> MMax;
  real alpha;
  real beta;
  real gamma;
  real<lower=0> sigma_low;
  real<lower=0> sigma_high;

  real<lower=absolute_MMin, upper=absolute_MMax> m1s[nobs];
  real<lower=0, upper=1> m2_frac[nobs];
  real<lower=0, upper=dl_max> dls[nobs];
}

transformed parameters {
  real dH = 4.42563416002 * (67.74/H0);
  real Nex;
  real neff_det;
  real m2s[nobs];
  real zs[nobs];

  {
    real dlinterp[ninterp] = dls_of_zs(zinterp, dH, Om, w);

    for (i in 1:nobs) {
      m2s[i] = absolute_MMin + m2_frac[i]*(m1s[i] - absolute_MMin);
      zs[i] = interp1d(dls[i], dlinterp, zinterp);
    }

    {
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

      for (i in 1:nsel) {
        zsel[i] = interp1d(dlsel[i], dlinterp, zinterp);
        m1sel_source[i] = m1sel[i]/(1+zsel[i]);
        m2sel_source[i] = m2sel[i]/(1+zsel[i]);
      }

      // Will put R0 back in by hand later.
      log_dN_m_unwt = log_dNdm1dm2ddldt(m1sel_source, m2sel_source, dlsel, zsel, 1.0, MMin, MMax, alpha, beta, gamma, dH, Om, w, sigma_low, sigma_high, ms_norm);
      for (i in 1:nsel) {
        log_dN[i] = log_dN_m_unwt[i] - 2.0*log1p(zsel[i]) - log_wtsel[i];
      }

      for (i in 1:nsel) {
        log_dN2[i] = log_dN[i]*2.0;
      }

      Nsum = exp(log_sum_exp(log_dN));
      N2sum = exp(log_sum_exp(log_dN2));

      Nex = R0*Tobs/N_gen*Nsum;

      sigma_rel2 = N2sum/(Nsum*Nsum) - 1.0/N_gen;
      sigma_rel = sqrt(sigma_rel2);

      neff_det = 1.0/sigma_rel2;
    }
  }
}

model {
  real log_dN[nobs];

  sigma_low ~ lognormal(log(0.1), 1);
  sigma_high ~ lognormal(log(0.1), 1);

  R0 ~ lognormal(log(100), 1);

  MMin ~ normal(5, 2);
  MMax ~ normal(40, 15);

  H0 ~ normal(70, 15);
  Om ~ normal(0.3, 0.15);
  w ~ normal(-1, 0.5);

  alpha ~ normal(-1, 2);
  beta ~ normal(0, 2);
  gamma ~ normal(3, 2);

  log_dN = log_dNdm1dm2ddldt(m1s, m2s, dls, zs, R0, MMin, MMax, alpha, beta, gamma, dH, Om, w, sigma_low, sigma_high, ms_norm);
  for (i in 1:nobs) {
    target += log_dN[i];
    target += log(m1s[i]-absolute_MMin); // Jacobian: dm2/dm2_frac
  }

  for (i in 1:nobs) {
    real lp[nsamp];
    vector[3] mu;

    mu[1] = m1s[i]*(1+zs[i]);
    mu[2] = m2s[i]*(1+zs[i]);
    mu[3] = dls[i];
    for (j in 1:nsamp) {
      lp[j] = multi_normal_cholesky_lpdf(pts[i,j] | mu, chol_bws[i]);
    }

    target += log_sum_exp(lp) - log(nsamp);
  }

  // Poisson norm
  target += -Nex;
}
