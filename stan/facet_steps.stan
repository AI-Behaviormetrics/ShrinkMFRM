// Conventional models whose step parameters vary across one facet without the decomposition:
//   by_rater = 0 : tau_{irm} = beta_i + beta_r + d_{im}   (M_u; step parameters vary across items)
//   by_rater = 1 : tau_{irm} = beta_i + beta_r + d_{rm}   (M_v; step parameters vary across raters)
// Each step vector (d_{i2}, ..., d_{iK}) or (d_{r2}, ..., d_{rK}) sums to zero and has an N(0, 1) prior on every element.
data {
  int<lower=1> J;
  int<lower=1> I;
  int<lower=1> R;
  int<lower=3> K;
  int<lower=1> N;
  array[N] int<lower=1, upper=J> examinee;
  array[N] int<lower=1, upper=I> item;
  array[N] int<lower=1, upper=R> rater;
  array[N] int<lower=1, upper=K> score;
  int<lower=0, upper=1> by_rater;
}
transformed data {
  vector[K] c = cumulative_sum(rep_vector(1, K)) - 1;
  int F = by_rater ? R : I;  // number of elements of the facet on which the step parameters depend
}
parameters {
  vector[J] theta;
  vector[I] beta_i;
  vector[R - 1] beta_r_free;
  matrix[F, K - 2] d_free;
}
transformed parameters {
  vector[R] beta_r;
  matrix[F, K - 1] d;  // step parameters of each item (or rater), m = 2, ..., K
  array[I, R] vector[K - 1] cum_tau;
  beta_r[1:(R - 1)] = beta_r_free;
  beta_r[R] = -sum(beta_r_free);
  for (f in 1:F) {
    d[f, 1:(K - 2)] = d_free[f];
    d[f, K - 1] = -sum(d_free[f]);
  }
  for (i in 1:I) for (r in 1:R) cum_tau[i, r] = cumulative_sum(beta_i[i] + beta_r[r] + (by_rater ? d[r]' : d[i]'));
}
model {
  theta ~ normal(0, 1);
  beta_i ~ normal(0, 1);
  beta_r ~ normal(0, 1);
  to_vector(d) ~ normal(0, 1);
  for (n in 1:N)
    score[n] ~ categorical_logit(c * theta[examinee[n]] - append_row(0, cum_tau[item[n], rater[n]]));
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = categorical_logit_lpmf(score[n] | c * theta[examinee[n]] - append_row(0, cum_tau[item[n], rater[n]]));
}
