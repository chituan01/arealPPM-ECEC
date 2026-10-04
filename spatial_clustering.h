#ifndef SPATIAL_CLUSTERING_H
#define SPATIAL_CLUSTERING_H

// #define ARMA_USE_SUPERLU 1
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// Main MCMC sampler function exposed to R
List Gibbs_SpatialClustering(arma::mat y, arma::cube X_O, arma::cube X_E,
                             arma::sp_mat W, List hyper_param, List mcmc_param,
                             List flags);

#endif
