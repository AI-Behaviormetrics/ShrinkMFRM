# Usage: Rscript main.R <dataset> <model> [iter] [warmup] [chains]
#   dataset: ratings1 | ratings2 | immer01b | japanese_essay
#   model:   proposed | m0 | mu | mv | muv | mu_h | mv_h
#   The defaults (iter = 6000, warmup = 3000, chains = 4) are the settings used in the paper.

library(rstan)
library(loo)
rstan_options(auto_write = TRUE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript main.R <dataset> <model> [iter] [warmup] [chains]")
dataset <- args[1]
model <- args[2]
iter <- if (length(args) >= 3) as.integer(args[3]) else 6000
warmup <- if (length(args) >= 4) as.integer(args[4]) else 3000
chains <- if (length(args) >= 5) as.integer(args[5]) else 4
options(mc.cores = chains)

# ---- data (long format: one row per rating, scores recoded to 1, ..., K) ----

load_data <- function(name) {
  if (name == "japanese_essay") {
    d <- read.csv("data/japanese_essay.csv")
  } else {
    src <- switch(name, ratings1 = c("sirt", "data.ratings1"), ratings2 = c("sirt", "data.ratings2"),
                  immer01b = c("immer", "data.immer01b"), stop("unknown dataset: ", name))
    e <- new.env()
    data(list = src[2], package = src[1], envir = e)
    raw <- e[[src[2]]]
    # the five rubric criteria k1, ..., k5 are treated as the items; scores 0-3 are recoded to 1-4
    d <- data.frame(examinee = rep(raw$idstud, 5), item = rep(1:5, each = nrow(raw)),
                    rater = rep(as.character(raw$rater), 5), score = unlist(raw[paste0("k", 1:5)]) + 1)
  }
  # examinees, items, and raters are renumbered 1, 2, ... in the sorted order of their original identifiers;
  # K is taken as the largest observed score
  list(J = length(unique(d$examinee)), I = length(unique(d$item)), R = length(unique(d$rater)),
       K = max(d$score), N = nrow(d),
       examinee = match(d$examinee, sort(unique(d$examinee))), item = match(d$item, sort(unique(d$item))),
       rater = match(d$rater, sort(unique(d$rater))), score = as.integer(d$score))
}

# ---- models (Stan file and data flags) ----

models <- list(
  proposed = list(file = "stan/proposed.stan", flags = list()),
  m0   = list(file = "stan/comparison.stan", flags = list(use_u = 0, use_v = 0, hier_u = 0, hier_v = 0)),
  muv  = list(file = "stan/comparison.stan", flags = list(use_u = 1, use_v = 1, hier_u = 0, hier_v = 0)),
  mu_h = list(file = "stan/comparison.stan", flags = list(use_u = 1, use_v = 0, hier_u = 1, hier_v = 0)),
  mv_h = list(file = "stan/comparison.stan", flags = list(use_u = 0, use_v = 1, hier_u = 0, hier_v = 1)),
  mu   = list(file = "stan/facet_steps.stan", flags = list(by_rater = 0)),
  mv   = list(file = "stan/facet_steps.stan", flags = list(by_rater = 1)))
if (!(model %in% names(models))) stop("model must be one of: ", paste(names(models), collapse = ", "))

# ---- estimation ----

dat <- load_data(dataset)
cat(sprintf("dataset = %s (J = %d, I = %d, R = %d, K = %d, N = %d), model = %s\n",
            dataset, dat$J, dat$I, dat$R, dat$K, dat$N, model))
sm <- stan_model(models[[model]]$file)
elapsed <- system.time(
  fit <- sampling(sm, data = c(dat, models[[model]]$flags), iter = iter, warmup = warmup, chains = chains,
                  seed = 1, refresh = 0)
)[["elapsed"]]

# ---- output ----

pars <- c("theta", "beta_i", "beta_r", "d", "u", "v", "sigma_u", "sigma_v")
pars <- pars[pars %in% fit@model_pars]
pars <- pars[sapply(pars, function(p) length(extract(fit, pars = p)[[p]]) > 0)]  # drop empty parameters
s <- summary(fit, pars = pars)$summary
cat(sprintf("\nsampling time: %.0f sec, max Rhat: %.3f, divergent transitions: %d\n", elapsed,
            max(s[, "Rhat"], na.rm = TRUE),
            sum(sapply(get_sampler_params(fit, inc_warmup = FALSE), function(x) sum(x[, "divergent__"])))))
cat("\n== posterior means (SD) of the parameters ==\n")
for (p in pars) {
  e <- extract(fit, pars = p)[[p]]
  if (length(dim(e)) == 1) {
    cat(sprintf("%s: %.3f (%.3f)\n", p, mean(e), sd(e)))
  } else if (length(dim(e)) == 2) {
    cat(p, ":\n", sep = "")
    out <- round(data.frame(mean = colMeans(e), sd = apply(e, 2, sd)), 3)
    if (p == "d") rownames(out) <- paste0("m=", 1 + seq_len(nrow(out)))  # d_2, ..., d_K
    print(out)
  } else {
    m <- round(apply(e, c(2, 3), mean), 3)
    dimnames(m) <- list(seq_len(nrow(m)), paste0("m=", 1 + seq_len(ncol(m))))
    cat(p, " (posterior means; rows: elements of the facet, columns: categories):\n", sep = "")
    print(m)
  }
}
if (any(c("sigma_u", "sigma_v") %in% pars)) {
  cat("\n== 95% credible intervals of the standard deviation parameters ==\n")
  for (p in intersect(c("sigma_u", "sigma_v"), pars)) {
    e <- as.numeric(extract(fit, pars = p)[[p]])
    cat(sprintf("%s: [%.3f, %.3f]\n", p, quantile(e, 0.025), quantile(e, 0.975)))
  }
}

ll <- extract_log_lik(fit, merge_chains = FALSE)
w <- waic(ll)
l <- loo(ll, r_eff = relative_eff(exp(ll)))
cat("\n== information criteria (estimate, SE) ==\n")
cat(sprintf("WAIC:     %.1f (%.1f)\n", w$estimates["waic", "Estimate"], w$estimates["waic", "SE"]))
cat(sprintf("PSIS-LOO: %.1f (%.1f)   observations with Pareto k > 0.7: %d\n",
            l$estimates["looic", "Estimate"], l$estimates["looic", "SE"], sum(pareto_k_values(l) > 0.7)))
