#ifndef FIXED_PARTITION_CLUSTERING_H
#define FIXED_PARTITION_CLUSTERING_H

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// Main MCMC sampler function exposed to R
List Gibbs_FixedPartition(arma::mat y, arma::cube X_O, arma::cube X_E,
                          arma::sp_mat W, arma::uvec optimal_partition,
                          List hyper_param, List mcmc_param,
                          List flags);

#endif
