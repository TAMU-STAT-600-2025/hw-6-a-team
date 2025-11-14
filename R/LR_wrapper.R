#' Multi-class Logistic Regression via Damped Newton's Method
#'
#' @description
#' Fits a multi-class logistic regression model using damped Newton's method.
#'
#' @param X Numeric matrix (n x p). The first column should be all 1s (intercept).
#' @param y Integer/numeric vector (length n) of class labels \code{0,1,...,K-1}.
#' @param numIter Positive integer; number of Newton iterations (default 50).
#' @param eta Positive numeric; damping/step size (default 0.1).
#' @param lambda Nonnegative numeric; ridge penalty (default 1).
#' @param beta_init Optional numeric matrix (p x K) of initial coefficients.
#'   If \code{NULL}, initialized to zeros.
#'
#' @return A list with
#' \item{beta}{Numeric matrix (p x K) of coefficients after \code{numIter} iterations.}
#' \item{objective}{Numeric vector of length \code{numIter + 1}; objective values at
#'   the start and after each iteration. Objective is
#'   \eqn{-\sum_i \log p_{y_i}(x_i) + (\lambda/2) \|\beta\|_F^2}.}
#' @export
#'
#' @examples
#' # Give example
#' set.seed(1)
#' n <- 200; p <- 6; K <- 3
#' X <- cbind(1, matrix(rnorm(n*(p-1)), n, p-1))  # intercept in col 1
#' beta_true <- matrix(runif(p*K, -0.5, 0.5), p, K)
#' # Generate probabilities and y
#' lin <- X %*% beta_true
#' lin <- lin - apply(lin, 1, max)               # stabilize
#' P <- exp(lin); P <- P / rowSums(P)
#' y <- apply(P, 1, function(prob) sample(0:(K-1), 1, prob = prob))
#'
#' fit <- LRMultiClass(X, y, numIter = 30, eta = 0.2, lambda = 0.5)
#' str(fit)
#' 
#' 
LRMultiClass <- function(X, y, beta_init = NULL, numIter = 50, eta = 0.1, lambda = 1){
  
  # Compatibility checks from HW3 and initialization of beta_init
  if(!is.matrix(X)) stop("X should be a matrix")
  if(!all(X[,1] == 1)) stop("The first column of X should be all 1s")
  if(!is.vector(y)) stop("y should be a vector")
  if(!all(is.finite(X))) stop("X contains non-finite values")
  if(!all(is.finite(y))) stop("y contains non-finite values")
  if(!is.numeric(numIter) || length(numIter) != 1 || numIter <= 0 || numIter != round(numIter)) stop("numIter should be a positive integer")
  if(!is.numeric(eta) || length(eta) != 1 || eta <= 0) stop("eta should be a positive number")
  if(!is.numeric(lambda) || length(lambda) != 1 || lambda < 0) stop("lambda should be a non-negative number")
  
  n <- nrow(X)
  p <- ncol(X)
  if(length(y) != n) stop("Length of y should be equal to the number of rows in X")
  K <- length(unique(y))
  if(!all(sort(unique(y)) == 0:(K-1))) stop("y should contain class labels from 0 to K-1")
  
  if(is.null(beta_init)){
    beta_init <- matrix(0, nrow = p, ncol = K)
  } else {
    if(nrow(beta_init) != p || ncol(beta_init) != K) stop("Dimensions of beta_init should be p x K")
    beta_init <- beta_init
  }

  # Call C++ LRMultiClass_c function to implement the algorithm
  out = LRMultiClass_c(X, y, beta_init, numIter, eta, lambda)
  
  # Return the class assignments
  return(out)
}