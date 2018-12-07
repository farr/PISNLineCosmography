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

  real[] log_dNdm1dm2ddldt_norm(real[] m1s, real[] m2s, real[] dls, real[] zs, real MMin, real MMax, real alpha, real beta, real gamma, real dH, real Om, real w, real smooth_low, real smooth_high, real[] ms_norm) {
    int nm = size(ms_norm);
    int n = size(m1s);
    real pms_alpha[nm];
    real pms_beta[nm];
    real cum_beta[nm];
    real log_norm_alpha;
    real log_dN[n];
    real log_4pi_dH = log(4.0*pi()*dH);

    for (i in 1:nm) {
      pms_alpha[i] = exp(softened_power_law_logpdf_unnorm(ms_norm[i], -alpha, MMin, MMax, smooth_low, smooth_high));
      pms_beta[i] = exp(softened_power_law_logpdf_unnorm(ms_norm[i], beta, MMin, MMax, smooth_low, smooth_high));
    }

    cum_beta = cumtrapz(ms_norm, pms_beta);

    log_norm_alpha = log(trapz(ms_norm, pms_alpha));

    for (i in 1:n) {
      real log_norm_beta = log(interp1d(m1s[i], ms_norm, cum_beta));
      real log_dNdm1dm2dVdt = softened_power_law_logpdf_unnorm(m1s[i], -alpha, MMin, MMax, smooth_low, smooth_high) + softened_power_law_logpdf_unnorm(m2s[i], beta, MMin, MMax, smooth_low, smooth_high) - log_norm_alpha - log_norm_beta + (gamma-1)*log1p(zs[i]);
      real log_dVdz = log_4pi_dH + log(dls[i]*dls[i]/((1+zs[i])*(1+zs[i]))/Ez(zs[i], Om, w));
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
  int nsel;
  int ninterp;
  int nnorm;

  int nsamp[nobs]; /* Number of samples assigned to each observation */
  int nsamp_total; /* Total number of samples to all observations */

  real Tobs;
  int N_gen;

  real m1obs[nsamp_total];
  real m2obs[nsamp_total];
  real dlobs[nsamp_total];
  real log_samp_wts[nsamp_total];

  real m1sel[nsel];
  real m2sel[nsel];
  real dlsel[nsel];
  real wtsel[nsel];

  real zinterp[ninterp];

  real ms_norm[nnorm];

  int use_cosmo_prior; /* Override hard-coded priors */
  real mu_H0;
  real sigma_H0;
  real mu_Omh2;
  real sigma_Omh2;
}

transformed data {
  real log_wtsel[nsel];

  for (i in 1:nsel) {
    log_wtsel[i] = log(wtsel[i]);
  }
}

parameters {
  real<lower=35, upper=140> H0;
  real<lower=0,upper=1> Om;
  real<lower=-2, upper=0> w;

  real<lower=0> R0;
  real<lower=3, upper=10> MMin;
  real<lower=30, upper=100> MMax;
  real<lower=-5, upper=3> alpha;
  real<lower=-3, upper=3> beta;
  real<lower=-1, upper=7> gamma;
  real<lower=0.01, upper=1> sigma_low;
  real<lower=0.01, upper=1> sigma_high;
}

transformed parameters {
  real dH = 4.42563416002 * (67.74/H0);
  real Nex;
  real neff_det;

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

    dlinterp = dls_of_zs(zinterp, dH, Om, w);

    for (i in 1:nsel) {
      zsel[i] = interp1d(dlsel[i], dlinterp, zinterp);
      m1sel_source[i] = m1sel[i]/(1+zsel[i]);
      m2sel_source[i] = m2sel[i]/(1+zsel[i]);
    }

    log_dN_m_unwt = log_dNdm1dm2ddldt_norm(m1sel_source, m2sel_source, dlsel, zsel, MMin, MMax, alpha, beta, gamma, dH, Om, w, sigma_low, sigma_high, ms_norm);
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

model {
  real log_dN_nojac[nsamp_total];
  real log_dN[nsamp_total];
  real m1s_source[nsamp_total];
  real m2s_source[nsamp_total];
  real zs[nsamp_total];
  real dlinterp[ninterp];

  sigma_low ~ lognormal(log(0.1), 1);
  sigma_high ~ lognormal(log(0.1), 1);

  R0 ~ lognormal(log(100), 1);

  MMin ~ normal(5, 2);
  MMax ~ normal(40, 15);

  if (use_cosmo_prior == 0) {
    H0 ~ normal(70, 15);
    Om ~ normal(0.3, 0.15);
  } else {
    real Omh2 = Om*(H0/100)^2;

    H0 ~ normal(mu_H0, sigma_H0);
    target += normal_lpdf(Omh2 | mu_Omh2, sigma_Omh2);
    target += 2.0*log(H0/100.0); /* Jacobian d(Om*h^2)/d(Om) */
  }

  w ~ normal(-1, 0.5);

  alpha ~ normal(-1, 2);
  beta ~ normal(0, 2);
  gamma ~ normal(3, 2);

  dlinterp = dls_of_zs(zinterp, dH, Om, w);

  for (i in 1:nsamp_total) {
    zs[i] = interp1d(dlobs[i], dlinterp, zinterp);
    m1s_source[i] = m1obs[i]/(1+zs[i]);
    m2s_source[i] = m2obs[i]/(1+zs[i]);
  }

  log_dN_nojac = log_dNdm1dm2ddldt_norm(m1s_source, m2s_source, dlobs, zs, MMin, MMax, alpha, beta, gamma, dH, Om, w, sigma_low, sigma_high, ms_norm);

  /* Here we compute dN/d(m1det)d(m2det)d(dL) / p(m1det, m2det, dL) */
  for (i in 1:nsamp_total) {
    log_dN[i] = log_dN_nojac[i] - 2.0*log1p(zs[i]) - log_samp_wts[i];
  }

  /* Now we marginalize over the samples for each event */
  {
    int istart = 1;
    for (i in 1:nobs) {
      target += log_sum_exp(log_dN[istart:istart+nsamp[i]-1]) - log(nsamp[i]);
      istart = istart + nsamp[i];
    }
  }

  /* Include the R0 term */
  target += nobs*log(R0);

  // Poisson norm
  target += -Nex;
}

generated quantities {
  real neff[nobs];
  real m1_source[nobs];
  real m2_source[nobs];
  real dl_source[nobs];
  real z_source[nobs];

  {
    real dlinterp[ninterp];
    real log_dN_nojac[nsamp_total];
    real log_dN[nsamp_total];
    real m1s_source[nsamp_total];
    real m2s_source[nsamp_total];
    real zs[nsamp_total];

    int istart;

    dlinterp = dls_of_zs(zinterp, dH, Om, w);

    for (i in 1:nsamp_total) {
      zs[i] = interp1d(dlobs[i], dlinterp, zinterp);
      m1s_source[i] = m1obs[i]/(1+zs[i]);
      m2s_source[i] = m2obs[i]/(1+zs[i]);
    }

    log_dN_nojac = log_dNdm1dm2ddldt_norm(m1s_source, m2s_source, dlobs, zs, MMin, MMax, alpha, beta, gamma, dH, Om, w, sigma_low, sigma_high, ms_norm);

    for (i in 1:nsamp_total) {
      log_dN[i] = log_dN_nojac[i] - 2.0*log1p(zs[i]);
    }

    istart = 1;
    for (i in 1:nobs) {
      real wts[nsamp[i]];
      real max_log_dN;
      real r;

      max_log_dN = max(log_dN[istart:istart+nsamp[i]-1]);
      for (j in 1:nsamp[i]) {
        wts[j] = exp(log_dN[istart+j-1]-max_log_dN);
      }

      neff[i] = sum(wts);

      r = uniform_rng(0,1);
      for (j in 1:nsamp[i]) {
        if (r < wts[j]) {
          m1_source[i] = m1s_source[istart + j - 1];
          m2_source[i] = m2s_source[istart + j - 1];
          dl_source[i] = dlobs[istart + j - 1];
          z_source[i] = zs[istart + j - 1];
          break;
        } else {
          r = r - wts[j];
        }
      }

      istart = istart + nsamp[i];
    }
  }
}
