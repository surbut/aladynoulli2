
library(MASS)



compute_kernel_matrix <- function(T, length_scale, var_scale) {
  time_diff_matrix <- outer(1:T, 1:T, "-")^2
  K <- var_scale * exp(-0.5 * time_diff_matrix / length_scale^2)
  K + diag(1e-6, T)  # Add small jitter for numerical stability
}

# Function to sample from multivariate normal
rmvn <- function(n, mu, K) {
  p <- length(mu)
  Z <- matrix(rnorm(n*p), nrow=n, ncol=p)
  L <- chol(K)
  X <- Z %*% L
  X + rep(mu, each=n)
}


# Helper Functions
rbf_kernel <- function(t1, t2, length_scale, variance) {
  return(variance * exp(-0.5 * (t1 - t2)^2 / length_scale^2))
}




softmax <- function(x) {
  exp_x <- exp(x - max(x))  # Subtract max for numerical stability
  return(exp_x / sum(exp_x))
}

# Function to apply softmax to Lambda
apply_softmax_to_lambda <- function(Lambda) {
  N <- dim(Lambda)[1]
  K <- dim(Lambda)[2]
  T <- dim(Lambda)[3]
  
  theta <- array(0, dim = c(N, K, T))
  for (i in 1:N) {
    for (t in 1:T) {
      theta[i, , t] <- softmax(Lambda[i, , t])
    }
  }
  return(theta)
}


log_sum_exp <- function(x) {
  max_x <- max(x)
  max_x + log(sum(exp(x - max_x)))
}



logistic <- function(x) {
  return(1 / (1 + exp(-x)))
}
# 