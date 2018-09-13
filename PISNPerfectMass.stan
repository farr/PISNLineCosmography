functions {
  real Ez(real z, real Om, real wDE) {
    real opz = 1.0 + z;
    real opz2 = opz*opz;
    real opz3 = opz2*opz;

    return sqrt(opz3*Om + (1.0-Om)*(1.0+z)^(3*(1+wDE)));
  }

  real [] dzdDL(real dl, real[] state, real[] theta, real[] x_r, int[] x_i) {
    real dH = theta[1];
    real Om = theta[2];
    real wDE = theta[3];
    real z = state[1];

    real dstatedDL[1];

    /* DL = (1+z) DC and d(DC)/dz = dH/E(z) => this equation */
    dstatedDL[1] = 1.0/(dl/(1+z) + (1+z)*dH/Ez(z, Om, wDE));

    return dstatedDL;
  }
}

data {
  int nobs;
  real m1_obs[nobs];
  real dls[nobs];
}

transformed data {
  real x_r[0];
  int x_i[0];

  int dlinds[nobs] = sort_indices_asc(dls);

  real sorted_dls[nobs];

  for (i in 1:nobs) {
    sorted_dls[i] = dls[dlinds[i]];
  }

}

parameters {
  real<lower=0> H0;
  real<lower=0,upper=1> Om;
  real<lower=-1,upper=1.0/3.0> wDE;

  real<lower=0> dMMax;
}

transformed parameters {
  real dH = 4.42563416002 * (67.74/H0);
  real zs[nobs];
  real m1s[nobs];
  real MMax;

  {
    real state0[1];
    real states[nobs, 1];
    real theta[3];

    state0[1] = 0.0;

    theta[1] = dH;
    theta[2] = Om;
    theta[3] = wDE;

    states = integrate_ode_rk45(dzdDL, state0, 0, sorted_dls, theta, x_r, x_i);

    for (i in 1:nobs) {
      zs[dlinds[i]] = states[i,1];
    }
  }

  for (i in 1:nobs) {
    m1s[i] = m1_obs[i]/(1+zs[i]);
  }

  MMax = max(m1s) + dMMax;
}

model {
  H0 ~ normal(70.0, 20.0);
  Om ~ normal(0.3, 0.1);

  /* We put a prior on wDE that peaks at -1 with a width of 1/3. */
  wDE + 1.0 ~ exponential(3.0);

  /* Flat prior on MMax => flat prior on dMMax. */
  /* Flat prior on zMax => flat prior on dzMax. */

  /* Population distribution: flat in m1, flat in dL. */
  target += -nobs*log(MMax);

  /* Observational likelihood just picks out the corresponding dL and m.  This
     is a bit subtle.  The likelihood is

     delta(m1_obs - m1*(1+z))

     but when we integrate out the m1 degree of freedom, we pick up a factor of
     1/(1+z) from the Jacobian, so we need to include that in our distribution.
     */

  for (i in 1:nobs) {
    target += -log1p(zs[i]);
  }
}
