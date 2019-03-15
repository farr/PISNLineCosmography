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

  real log_power_law_norm(real a, real l, real h) {
    if (a > -1) {
      real opa = 1.0 + a;
      return opa*log(h) + log1p(-(l/h)^opa) - log1p(a);
    } else {
      real opa = 1.0 + a;
      return opa*log(l) + log1p(-(h/l)^(opa)) - log(-opa);
    }
  }

  real Ez(real z, real Om, real z_p, real w_p, real w_a) {
    real a = 1.0/(1.0+z);
    real a_p = 1.0/(1.0+z_p);
    real opz = 1.0 + z;
    real opz2 = opz*opz;
    real opz3 = opz2*opz;

    return sqrt(Om*opz3 + (1.0-Om)*opz^(3*(1 + w_p + w_a*a_p))*exp(-3*w_a*(1-a)));
  }

  real dzddL(real dl, real z, real dH, real Om, real z_p, real w_p, real w_a) {
    real opz = 1.0 + z;
    return 1.0/(dl/opz + opz*dH/Ez(z, Om, z_p, w_p, w_a));
  }

  /* For x between A and B, this factor is 1 when x == A, 0 when x == B, and
  /* smooth at both ends. */
  real log_smoothing_factor(real x, real A, real B) {
    real Bmx = B - x;
    real AmB = A - B;
    return log((3.0*A - B - 2.0*x)*Bmx*Bmx/(AmB*AmB*AmB));
  }

  real[] log_dNdm1dm2dzdt_norm(real[] m1s, real[] m2s, real[] dls, real[] zs, real MMin, real MMax, real alpha, real beta, real gamma, real dH, real Om, real z_p, real w_p, real w_a) {
    int n = size(m1s);
    real log_norm_alpha = log_power_law_norm(-alpha, MMin, MMax);
    real log_4pi_dH = log(4.0*pi()*dH);
    real log_dNs[n];
    real smin = 0.1;
    real smax = 0.5;

    for (i in 1:n) {
      if ((MMin < m2s[i]) && (m2s[i] < m1s[i]) && (m1s[i] < MMax)) {
        real log_norm_beta = log_power_law_norm(beta, MMin, m1s[i]);
        real s1;
        real s2;

        /* Smooth over m1 when it gets close to either MMin or MMax */
        if (m1s[i] < MMin + smin) {
          s1 = log_smoothing_factor(m1s[i], MMin + smin, MMin);
        } else if (m1s[i] > MMax - smax) {
          s1 = log_smoothing_factor(m1s[i], MMax - smax, MMax);
        } else {
          s1 = 0.0;
        }

        if (m2s[i] < MMin + smin) {
          s2 = log_smoothing_factor(m2s[i], MMin + smin, MMin);
        } else if (m2s[i] > MMax - smax) {
          s2 = log_smoothing_factor(m2s[i], MMax - smax, MMax);
        } else {
          s2 = 0.0;
        }

        log_dNs[i] = -alpha*log(m1s[i]) + beta*log(m2s[i]) + (gamma-1)*log1p(zs[i]) - log_norm_alpha - log_norm_beta + log_4pi_dH + 2.0*log(dls[i]/(1+zs[i])) - log(Ez(zs[i], Om, z_p, w_p, w_a)) + s1 + s2;
      } else {
        log_dNs[i] = negative_infinity();
      }
    }

    return log_dNs;
  }

  real[] dls_of_zs(real[] zs, real dH, real Om, real z_p, real w_p, real w_a) {
    int n = size(zs);
    real ddcdz[n];
    real dcs[n];
    real dls[n];

    for (i in 1:n) {
      ddcdz[i] = dH/Ez(zs[i], Om, z_p, w_p, w_a);
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

  int nsamp;

  real Tobs;
  int N_gen;

  real m1obs[nobs, nsamp];
  real m2obs[nobs, nsamp];
  real dlobs[nobs, nsamp];
  matrix[3,3] bw[nobs];

  real m1sel[nsel];
  real m2sel[nsel];
  real dlsel[nsel];
  real log_wtsel[nsel];

  int ninterp;
  real zinterp[ninterp];

  int cosmo_prior;
}

transformed data {
  real z_p = 0.7;
  real a_p = 1.0/(1+z_p);
  matrix[3,3] chol_bw[nobs];
  real zmax = zinterp[ninterp];

  for (i in 1:nobs) {
    chol_bw[i] = cholesky_decompose(bw[i]);
  }
}

parameters {
  real<lower=35, upper=140> H0;
  real<lower=0,upper=(H0/100)^2> Omh2;
  real<lower=-2,upper=0> w_p;
  real<lower=-1, upper=1> w_a;

  real<lower=3, upper=10> MMin;
  real<lower=30, upper=100> MMax;
  real<lower=-5, upper=3> alpha;
  real<lower=-3, upper=3> beta;
  real<lower=-1, upper=7> gamma;

  real<lower=MMin, upper=MMax> m1s[nobs];
  real<lower=0, upper=1> m2_fracs[nobs];
  real<lower=0, upper=zmax> zs[nobs];
}

transformed parameters {
  real m2s[nobs];
  real dls[nobs];
  real w = w_p + (a_p - 1.0)*w_a;
  real dH = 4.42563416002 * (67.74/H0);
  real Om = Omh2/(H0/100)^2;
  real mu_det;
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

    dlinterp = dls_of_zs(zinterp, dH, Om, z_p, w_p, w_a);

    for (i in 1:nobs) {
      dls[i] = interp1d(zs[i], zinterp, dlinterp);
      m2s[i] = MMin + (m1s[i] - MMin)*m2_fracs[i];
    }

    for (i in 1:nsel) {
      zsel[i] = interp1d(dlsel[i], dlinterp, zinterp);
      m1sel_source[i] = m1sel[i]/(1+zsel[i]);
      m2sel_source[i] = m2sel[i]/(1+zsel[i]);
    }

    log_dN_m_unwt = log_dNdm1dm2dzdt_norm(m1sel_source, m2sel_source, dlsel, zsel, MMin, MMax, alpha, beta, gamma, dH, Om, z_p, w_p, w_a);
    for (i in 1:nsel) {
      log_dN[i] = log_dN_m_unwt[i] - 2.0*log1p(zsel[i]) + log(dzddL(dlsel[i], zsel[i], dH, Om, z_p, w_p, w_a)) - log_wtsel[i];
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

    for (i in 1:nobs) {
      dls[i] = interp1d(zs[i], zinterp, dlinterp);
    }
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
    /* Smoothly cut off the log-density: log(0) at 4*nobs, log(1) at 5*nobs */
    target += log_smoothing_factor(neff_det, 5*nobs, 4*nobs);
  } else {
    reject("need more samples for selection integral");
  }

  MMin ~ normal(5, 2); /* Imposes a prior on dMMin, since these are linearly related. */
  MMax ~ normal(50, 15); /* Imposes a prior on dMMax, since these are linearly related. */

  if (cosmo_prior == 0) {
    H0 ~ normal(70, 15);

    /* For H0 = 70, this peaks at Om ~ 0.3, with s.d. 0.15 */
    Omh2 ~ normal(0.15, 0.15/2.0);
  } else {
    H0 ~ normal(67.74, 0.6774);
    Omh2 ~ normal(0.02225+0.1198, sqrt(0.00016^2 + 0.0015^2));
  }
  w_p ~ normal(-1, 0.5);
  w_a ~ normal(0, 0.5);

  alpha ~ normal(1, 2);
  beta ~ normal(0, 2);
  gamma ~ normal(3, 2);

  /* Population */
  {
    real log_dNs_nojac[nobs] = log_dNdm1dm2dzdt_norm(m1s, m2s, dls, zs, MMin, MMax, alpha, beta, gamma, dH, Om, z_p, w_p, w_a);
    real log_dNs_jac[nobs];

    for (i in 1:nobs) {
      /* The density is in m1-m2-z space, but we sample in m1-m2_fracs-z, so
      /* need d(m2)/d(m2_frac) as Jacobian. */
      log_dNs_jac[i] = log_dNs_nojac[i] + log(m1s[i]-MMin);
    }

    target += sum(log_dNs_jac);
  }

  /* Likelihood */
  for (i in 1:nobs) {
    real log_ps[nsamp];
    vector[3] pt = to_vector({m1s[i]*(1+zs[i]), m2s[i]*(1+zs[i]), dls[i]});

    for (j in 1:nsamp) {
      vector[3] x = to_vector({m1obs[i,j], m2obs[i,j], dlobs[i,j]});

      log_ps[j] = multi_normal_cholesky_lpdf(x | pt, chol_bw[i]);
    }

    target += log_sum_exp(log_ps) - log(nsamp);
  }

  // Normalization term
  target += -(nobs+1)*log(mu_det) + nobs*(3 + nobs)/(2*neff_det);
}

generated quantities {
  real R0;

  {
    real mu_R0 = nobs/mu_det*(1.0 + nobs/neff_det);
    real sigma_R0 = sqrt(nobs)/mu_det*(1.0 + 1.5*nobs/neff_det);

    R0 = normal_rng(mu_R0, sigma_R0);
  }
}
