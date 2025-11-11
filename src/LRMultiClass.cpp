// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// For simplicity, no test data, only training data, and no error calculation.
// X - n x p data matrix
// y - n length vector of classes, from 0 to K-1
// numIter - number of iterations, default 50
// eta - damping parameter, default 0.1
// lambda - ridge parameter, default 1
// beta_init - p x K matrix of starting beta values (always supplied in right format)
// [[Rcpp::export]]
Rcpp::List LRMultiClass_c(const arma::mat& X, const arma::uvec& y, const arma::mat& beta_init,
                               int numIter = 50, double eta = 0.1, double lambda = 1){
    // All input is assumed to be correct
    
    // Initialize some parameters
    int K = max(y) + 1; // number of classes
    int p = X.n_cols;
    int n = X.n_rows;
    arma::mat beta = beta_init; // to store betas and be able to change them if needed
    arma::vec objective(numIter + 1); // to store objective values
    
    // Initialize anything else that you may need
    auto softmax_rows_stable = [&](const arma::mat& A)->arma::mat {
      arma::vec rmax = arma::max(A, 1);       // n x 1
      arma::mat Z = A.each_col() - rmax;      // stabilize per row
      Z = arma::exp(Z);                       // elementwise exp
      arma::vec rsum = arma::sum(Z, 1);       // n x 1
      Z.each_col() /= rsum;                   // normalize rows
      return Z;                               // n x K
    };
    auto objective_value = [&](const arma::mat& B)->double {
      arma::mat lin0 = X * B;                 // n x K
      arma::mat P0 = softmax_rows_stable(lin0);
      double nll = 0.0;
      for (int i = 0; i < n; ++i) {
        double py = P0(i, y(i));
        if (py <= 0) py = std::numeric_limits<double>::min();
        nll -= std::log(py);
      }
      double pen = 0.5 * lambda * arma::accu(arma::square(B));
      return nll + pen;
    };
    arma::mat lin(n, K, arma::fill::zeros);
    arma::mat P(n, K, arma::fill::zeros);
    arma::mat G(p, K, arma::fill::zeros);
    arma::mat Hk(p, p, arma::fill::zeros);
    arma::mat Xw(n, p, arma::fill::zeros);
    arma::vec wk(n, arma::fill::zeros);
    
    // Newton's method cycle - implement the update EXACTLY numIter iterations
    objective(0) = objective_value(beta);
    for (int iter = 0; iter < numIter; ++iter) {
      // probabilities under current beta
      lin = X * beta;               
      P = softmax_rows_stable(lin); 
      
      // Gradient: X^T (P - Y_onehot) + lambda * beta
      G.zeros();
      for (int k = 0; k < K; ++k) {
        arma::vec diff = P.col(k);                               // n
        diff.elem(arma::find(y == static_cast<arma::uword>(k))) -= 1.0;
        G.col(k) = X.t() * diff;                                 // p
      }
      G += lambda * beta;
      
      // Per-class block Hessian and damped Newton update
      for (int k = 0; k < K; ++k) {
        wk = P.col(k) % (1.0 - P.col(k));                        // n
        arma::vec swk = arma::sqrt(wk);
        Xw = X.each_col() % swk;                                 // n x p
        Hk = Xw.t() * Xw;                                        // p x p
        Hk.diag() += lambda;                                     // ridge on all coeffs
        
        arma::vec step = arma::solve(Hk, G.col(k), arma::solve_opts::fast);
        beta.col(k) -= eta * step;
      }
      
      // objective after update
      objective(iter + 1) = objective_value(beta);
    }
    
    
    // Create named list with betas and objective values
    return Rcpp::List::create(Rcpp::Named("beta") = beta,
                              Rcpp::Named("objective") = objective);
}
