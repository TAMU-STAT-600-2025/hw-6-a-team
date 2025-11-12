// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::uvec MyKmeans_c(const arma::mat& X, int K,
                      const arma::mat& M, int numIter = 100){
  // Inputs have already been validated and initialized in R.
  const int n = X.n_rows;
  const int p = X.n_cols;
  
  arma::mat centers = M;                       // K x p
  arma::mat new_centers(K, p, arma::fill::zeros);
  arma::uvec Y(n);                             // 0-based labels internally
  
  for (int iter = 0; iter < numIter; ++iter) {
    // ---- Assignment step (vectorized distance) ----
    // D(i,k) = ||x_i||^2 + ||m_k||^2 - 2 x_i m_k^T
    arma::vec x2 = arma::sum(arma::square(X), 1);       // n x 1
    arma::vec m2 = arma::sum(arma::square(centers), 1); // K x 1
    arma::mat D   = arma::repmat(x2, 1, K)
      + arma::repmat(m2.t(), n, 1)
      - 2.0 * (X * centers.t());            // n x K
      
      for (int i = 0; i < n; ++i) {
        Y[i] = D.row(i).index_min(); // closest center index in 0..K-1
      }
      
      // ---- Update step ----
      new_centers.zeros();
      arma::Col<arma::uword> counts(K, arma::fill::zeros);
      
      for (int i = 0; i < n; ++i) {
        arma::uword k = Y[i];
        ++counts[k];
        for (int j = 0; j < p; ++j) {
          new_centers(k, j) += X(i, j);
        }
      }
      
      // ---- Empty cluster check (match HW2 behavior: error) ----
      for (int k = 0; k < K; ++k) {
        if (counts[k] == 0u) {
          Rcpp::stop("Empty cluster detected at iteration " +
            std::to_string(iter + 1) +
            ". Please try a different initialization M.");
        }
      }
      
      for (int k = 0; k < K; ++k) {
        const double denom = static_cast<double>(counts[k]);
        for (int j = 0; j < p; ++j) {
          new_centers(k, j) /= denom;
        }
      }
      
      // ---- Convergence: centers unchanged (exact equality like R's identical) ----
      bool same = arma::approx_equal(new_centers, centers, "absdiff", 0.0);
      centers = new_centers;
      if (same) break;
  }
  
  // convert to 1..K for R
  return Y + 1u;
}
