#' K-means clustering
#'
#' @description
#' K-means clustering with input checking and initialization in R,
#' and the main algorithm implemented in C++ via \code{MyKmeans_c()}.
#'
#' @param X Numeric matrix or data.frame of size \eqn{n \times p},
#'   where each row is an observation.
#' @param K Integer, number of clusters. Must be between 1 and \code{nrow(X)}.
#' @param M Optional numeric matrix of initial cluster centers of size
#'   \eqn{K \times p}. If \code{NULL} (default), \code{K} distinct rows
#'   of \code{X} are chosen at random.
#' @param numIter Integer, maximum number of iterations (default 100).
#'
#' @return
#' An integer vector of length \eqn{n} with cluster assignments taking
#' values in \code{1, ..., K}.
#'
#' @export
#'
#' @examples
#' set.seed(123)
#' X <- rbind(
#'   matrix(rnorm(50, mean = 0, sd = 0.3), ncol = 2),
#'   matrix(rnorm(50, mean = 3, sd = 0.3), ncol = 2)
#' )
#' cl <- MyKmeans(X, K = 2, numIter = 50)
#' table(cl)
MyKmeans <- function(X, K, M = NULL, numIter = 100) {
  # ---- Input checks ----
  if (is.data.frame(X)) {
    X <- as.matrix(X)
  }
  if (!is.matrix(X) || !is.numeric(X)) {
    stop("X must be a numeric matrix or data.frame.", call. = FALSE)
  }
  if (any(!is.finite(X))) {
    stop("X contains non-finite values.", call. = FALSE)
  }
  
  n <- nrow(X)
  p <- ncol(X)
  
  if (length(K) != 1L || !is.numeric(K) || K != as.integer(K) ||
      K < 1L || K > n) {
    stop("K must be an integer in [1, nrow(X)].", call. = FALSE)
  }
  K <- as.integer(K)
  
  if (length(numIter) != 1L || !is.numeric(numIter) ||
      numIter != as.integer(numIter) || numIter < 1L) {
    stop("numIter must be a positive integer.", call. = FALSE)
  }
  numIter <- as.integer(numIter)
  
  # ---- Initialization for M ----
  if (is.null(M)) {
    set <- sample.int(n, K, replace = FALSE)
    M <- X[set, , drop = FALSE]
  } else {
    if (is.data.frame(M)) {
      M <- as.matrix(M)
    }
    if (!is.matrix(M) || !is.numeric(M)) {
      stop("M must be a numeric matrix when provided.", call. = FALSE)
    }
    if (nrow(M) != K || ncol(M) != p) {
      stop("M must be a K x ncol(X) matrix.", call. = FALSE)
    }
    if (any(!is.finite(M))) {
      stop("M contains non-finite values.", call. = FALSE)
    }
  }
  
  # ---- Call C++ core ----
  # Here MyKmeans_c should implement the main K-means iterations only.
  Y <- MyKmeans_c(X, K, M, numIter)
  
  # Expect MyKmeans_c to return integer labels 1..K of length n
  Y
}
