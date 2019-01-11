functions {
  real distance_lpdf(real d) {
    return 2.0*log(d) - log(1.06012 + d*(0.34262 + d*(-0.000814998 + d*7.18688e-05)));
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

  real interp2d(real x, real y, real[] xs, real[] ys, real[,] zs) {
    int i = bisect_index(x, xs);
    int j = bisect_index(y, ys);

    real xl = xs[i-1];
    real xh = xs[i];
    real yl = ys[j-1];
    real yh = ys[j];

    real r = (x-xl)/(xh-xl);
    real s = (y-yl)/(yh-yl);

    return (1.0-r)*(1.0-s)*zs[i-1,j-1] + (1.0-r)*s*zs[i-1,j] + r*(1.0-s)*zs[i,j-1] + r*s*zs[i,j];
  }
}

data {
  real mc_obs;
  real eta_obs;
  real rho_obs;
  real theta_obs;

  real sigma_mc;
  real sigma_eta;
  real sigma_theta;

  int nm;
  real ms[nm];
  real opt_snrs[nm,nm];

  real dL_max;
}

transformed data {
  real minterp_min = min(ms);
  real minterp_max = max(ms);
}

parameters {
  real<lower=minterp_min, upper=minterp_max> m1;
  real<lower=minterp_min, upper=m1> m2;
  real<lower=0, upper=dL_max> dL;

  /* These are the vectors that go into Theta.  We employ a trick for the
     azimuthal angles: we generate a unit-vector uniformly on the circle, and
     then the angle is just atan2(...) of the components.  Also, we restrict the
     range of both cos_theta and cos_iota to avoid double-covering the space of
     polar angles.  (We still double-cover the asizmuthal space, since we only
     depend on double-angles, but the sampler doesn't choke as much on that.) */
  unit_vector[2] azimuth_v;
  real<lower=0,upper=1> cos_theta;
  unit_vector[2] pol_v;
  real<lower=0,upper=1> cos_iota;
}

transformed parameters {
  real mc;
  real eta;
  real opt_snr;

  real theta;

  /* Flat in log(m1) prior, flat between minterp_min and m1 for m2, and our
  /* approximate distance prior. */
  real log_m1m2dl_wt = distance_lpdf(dL) - log(m1) - log(m1-minterp_min);

  {
    real mt = m1+m2;

    eta = m1*m2/(mt*mt);
    mc = mt*eta^(3.0/5.0);
  }

  opt_snr = interp2d(m1, m2, ms, ms, opt_snrs)/dL;

  {
    real az = atan2(azimuth_v[2], azimuth_v[1]);
    real pol = atan2(pol_v[2], pol_v[1]);

    real c2p = cos(2*pol);
    real s2p = sin(2*pol);

    real c2a = cos(2*az);
    real s2a = sin(2*az);

    real Fp = 0.5*c2p*(1+cos_theta*cos_theta)*c2a - s2p*cos_theta*s2a;
    real Fc = 0.5*s2p*(1+cos_theta*cos_theta)*c2a + c2p*cos_theta*s2a;

    /* theta \in [0,1], uniform angles above give the right prior. */
    theta = sqrt(Fp*Fp*(1+cos_iota*cos_iota)*(1+cos_iota*cos_iota) + 4.0*Fc*Fc*cos_iota*cos_iota)/2.0;
  }
}

model {
  /* Apply the m1-m2-dL prior */
  target += log_m1m2dl_wt;

  // Observations; for some reason the ``T[a,b]`` truncation statements cause trouble.
  mc_obs ~ lognormal(log(mc), sigma_mc);
  eta_obs ~ normal(eta, sigma_eta);
  target += -log(normal_cdf(0.25, eta, sigma_eta) - normal_cdf(0.0, eta, sigma_eta));
  rho_obs ~ normal(theta .* opt_snr, 1.0);
  theta_obs ~ normal(theta, sigma_theta);
  target += -log(normal_cdf(1.0, theta, sigma_theta) - normal_cdf(0.0, theta, sigma_theta));
}
