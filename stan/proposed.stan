// Proposed model: a many-facet Rasch model with hierarchical shrinkage priors on facet-specific step parameters.
//   tau_{irm} = beta_i + beta_r + d_m + u_{im} + v_{rm}   (m = 2, ..., K)
//   beta_i is free; sum(beta_r) = 0; sum(d) = 0; u and v are double-centered (row and column sums are zero)
//   and scaled by sigma_u and sigma_v with the variance correction; sigma_u, sigma_v ~ half-normal(0, 0.5^2).
functions {
  // Double centering: subtract the row means and the column means and add the grand mean.
  matrix double_center(matrix M) {
    int nr = rows(M);
    int nc = cols(M);
    vector[nr] rm;
    vector[nc] cm;
    real gm = mean(M);
    matrix[nr, nc] out;
    for (i in 1:nr) rm[i] = mean(M[i, ]);
    for (j in 1:nc) cm[j] = mean(M[, j]);
    for (i in 1:nr) for (j in 1:nc) out[i, j] = M[i, j] - rm[i] - cm[j] + gm;
    return out;
  }
}
data {
  int<lower=1> J;  // number of examinees
  int<lower=2> I;  // number of items
  int<lower=2> R;  // number of raters
  int<lower=3> K;  // number of score categories
  int<lower=1> N;  // number of observed scores
  array[N] int<lower=1, upper=J> examinee;
  array[N] int<lower=1, upper=I> item;
  array[N] int<lower=1, upper=R> rater;
  array[N] int<lower=1, upper=K> score;
}
transformed data {
  vector[K] c = cumulative_sum(rep_vector(1, K)) - 1;  // 0, 1, ..., K - 1
  real corr_u = sqrt(I * (K - 1) * 1.0 / ((I - 1) * (K - 2)));  // variance correction for double centering
  real corr_v = sqrt(R * (K - 1) * 1.0 / ((R - 1) * (K - 2)));
}
parameters {
  vector[J] theta;
  vector[I] beta_i;
  vector[R - 1] beta_r_free;
  vector[K - 2] d_free;
  real<lower=0> sigma_u;
  real<lower=0> sigma_v;
  matrix[I, K - 1] w_u;  // auxiliary parameters (non-centered parameterization)
  matrix[R, K - 1] w_v;
}
transformed parameters {
  vector[R] beta_r;
  vector[K - 1] d;         // d_2, ..., d_K
  matrix[I, K - 1] u;      // u_{i2}, ..., u_{iK}
  matrix[R, K - 1] v;      // v_{r2}, ..., v_{rK}
  array[I, R] vector[K - 1] cum_tau;
  beta_r[1:(R - 1)] = beta_r_free;
  beta_r[R] = -sum(beta_r_free);
  d[1:(K - 2)] = d_free;
  d[K - 1] = -sum(d_free);
  u = sigma_u * corr_u * double_center(w_u);
  v = sigma_v * corr_v * double_center(w_v);
  for (i in 1:I) for (r in 1:R) cum_tau[i, r] = cumulative_sum(beta_i[i] + beta_r[r] + d + u[i]' + v[r]');
}
model {
  theta ~ normal(0, 1);
  beta_i ~ normal(0, 1);
  beta_r ~ normal(0, 1);  // prior on every element of the constrained vector
  d ~ normal(0, 1);
  sigma_u ~ normal(0, 0.5);
  sigma_v ~ normal(0, 0.5);
  to_vector(w_u) ~ std_normal();
  to_vector(w_v) ~ std_normal();
  for (n in 1:N)
    score[n] ~ categorical_logit(c * theta[examinee[n]] - append_row(0, cum_tau[item[n], rater[n]]));
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = categorical_logit_lpmf(score[n] | c * theta[examinee[n]] - append_row(0, cum_tau[item[n], rater[n]]));
}
