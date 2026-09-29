// Comparison models that share the parameterization of the proposed model (tau_{irm} = beta_i + beta_r + d_m + u_{im} + v_{rm}).
//   use_u, use_v   : 1 to include the item-specific (u) or rater-specific (v) step parameters, 0 to omit them
//   hier_u, hier_v : 1 for the hierarchical shrinkage prior (sigma estimated), 0 for the fixed scale (sigma = 1)
//   M_0   : use_u = 0, use_v = 0
//   M_uv  : use_u = 1, use_v = 1, hier_u = 0, hier_v = 0
//   M_u^h : use_u = 1, use_v = 0, hier_u = 1
//   M_v^h : use_u = 0, use_v = 1, hier_v = 1
functions {
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
  int<lower=1> J;
  int<lower=2> I;
  int<lower=2> R;
  int<lower=3> K;
  int<lower=1> N;
  array[N] int<lower=1, upper=J> examinee;
  array[N] int<lower=1, upper=I> item;
  array[N] int<lower=1, upper=R> rater;
  array[N] int<lower=1, upper=K> score;
  int<lower=0, upper=1> use_u;
  int<lower=0, upper=1> use_v;
  int<lower=0, upper=1> hier_u;
  int<lower=0, upper=1> hier_v;
}
transformed data {
  vector[K] c = cumulative_sum(rep_vector(1, K)) - 1;
  real corr_u = sqrt(I * (K - 1) * 1.0 / ((I - 1) * (K - 2)));
  real corr_v = sqrt(R * (K - 1) * 1.0 / ((R - 1) * (K - 2)));
  int nu = use_u ? I : 0;
  int nv = use_v ? R : 0;
  int hu = use_u * hier_u;
  int hv = use_v * hier_v;
}
parameters {
  vector[J] theta;
  vector[I] beta_i;
  vector[R - 1] beta_r_free;
  vector[K - 2] d_free;
  vector<lower=0>[hu] sigma_u;  // empty unless the hierarchical prior is used
  vector<lower=0>[hv] sigma_v;
  matrix[nu, K - 1] w_u;        // empty unless u is included
  matrix[nv, K - 1] w_v;
}
transformed parameters {
  vector[R] beta_r;
  vector[K - 1] d;
  matrix[I, K - 1] u = rep_matrix(0, I, K - 1);
  matrix[R, K - 1] v = rep_matrix(0, R, K - 1);
  array[I, R] vector[K - 1] cum_tau;
  beta_r[1:(R - 1)] = beta_r_free;
  beta_r[R] = -sum(beta_r_free);
  d[1:(K - 2)] = d_free;
  d[K - 1] = -sum(d_free);
  if (use_u) u = (hu ? sigma_u[1] : 1.0) * corr_u * double_center(w_u);
  if (use_v) v = (hv ? sigma_v[1] : 1.0) * corr_v * double_center(w_v);
  for (i in 1:I) for (r in 1:R) cum_tau[i, r] = cumulative_sum(beta_i[i] + beta_r[r] + d + u[i]' + v[r]');
}
model {
  theta ~ normal(0, 1);
  beta_i ~ normal(0, 1);
  beta_r ~ normal(0, 1);
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
