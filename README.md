# ShrinkMFRM

R and Stan code and data for the paper

> Uto, M. (submitted). A Bayesian many-facet Rasch model with shrinkage priors on facet-specific step parameters. *Behavior Research Methods*.

Please cite the paper if you use the code or the data.

## Contents

| File | Description |
|---|---|
| `main.R` | Estimates a model on a dataset and prints the parameter estimates, WAIC, and PSIS-LOO |
| `stan/proposed.stan` | The proposed model |
| `stan/comparison.stan` | The comparison models M_0, M_uv, M_u^h, and M_v^h (selected by data flags) |
| `stan/facet_steps.stan` | The comparison models M_u and M_v |
| `data/japanese_essay.csv` | The Japanese Essay dataset (holistic ratings, long format) |

## Requirements

R with the packages `rstan` (version 2.26 or later, with a working C++ toolchain), `loo`, `sirt` (for Ratings1 and Ratings2), and `immer` (for Immer01b). The paper used R 4.5.0 and RStan 2.36.

## Usage

Run the script from the repository root:

```
Rscript main.R <dataset> <model> [iter] [warmup] [chains]
```

- `dataset`: `ratings1`, `ratings2`, `immer01b`, or `japanese_essay`
- `model`: `proposed`, `m0`, `mu`, `mv`, `muv`, `mu_h`, or `mv_h`
- `iter`, `warmup`, `chains` default to 6000, 3000, and 4 (the settings used in the paper)

Example:

```
Rscript main.R japanese_essay proposed
```

The script prints the posterior means of the parameters (with posterior standard deviations for the vector parameters), the 95% credible intervals of the standard deviation parameters (sigma_u and sigma_v), and WAIC and PSIS-LOO with their standard errors. The Ratings1, Ratings2, and Immer01b datasets are loaded from the R packages `sirt` and `immer`; the five rubric criteria are treated as the items, and the scores 0-3 are recoded to 1-4. Examinees, items, and raters are renumbered in the sorted order of their original identifiers. With the default settings, one run takes from about a minute (Ratings1) to about an hour (Immer01b with M_uv) on a recent desktop machine, and Immer01b requires several gigabytes of memory for the pointwise log-likelihood.

## Data

`data/japanese_essay.csv` contains one row per rating with the columns `examinee` (1-134), `item` (1-4, the writing task), `rater` (1-18), and `score` (1-5, the holistic score). Each response was scored by five raters.
