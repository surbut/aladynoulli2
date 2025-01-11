
# Function to create increasing trend
create_increasing_trend <- function(start, end, T) {
  x <- seq(0, 1, length.out = T)
  y <- start + (end - start) * pnorm((x - 0.5) * 5)  # Using cumulative normal for S-shape
  return(y)
}



generate_tensor_data <- function(N = 1000, D = 5, T = 50, K = 3, num_covariates = 5) {
  library(mvtnorm)
  library(MASS)
  library(pracma)
  library(reshape2)
  library(dplyr)
  library(einsum)
  
  set.seed(123)
  
  
  
  # Create time differences matrix
  time_diff <- outer(seq_len(T), seq_len(T), "-")
  
  # Generate scales
  length_scales_lambda <- runif(K, T / 3, T / 2)
  var_scales_lambda <- runif(K, 0.8, 1.2)
  length_scales_phi <- runif(K, T / 3, T / 2)
  var_scales_phi <- runif(K, 0.8, 1.2)


  
  
  
  # Simulate mu
  mu_d <- array(NA, dim = c(D, T))
  for (d in 1:D) {
    time_points <- seq(0, 1, length.out = T)
    base_trend <- create_increasing_trend(qlogis(0.01), qlogis(0.05), T)

    # Add some randomness to the trend
    cov_matrix_mu <- exp(-0.5 * 0.1 * outer(time_points, time_points, "-")^2)
    random_effect <- mvrnorm(1, mu = rep(0, T), Sigma = cov_matrix_mu)
    mu_d[d, ] <- base_trend + random_effect 
  }
  
  # Generate lambda, phi matrices
  g_i <- array(rnorm(num_covariates * N), dim = c(N, num_covariates))
  lambda_ik <- array(NA, dim = c(N, K, T))
  phi_kd <- array(NA, dim = c(K, D, T))
  Gamma_k <- matrix(rnorm(num_covariates * K), nrow = K, ncol = num_covariates)
  
  # Simulate lambda
  for (k in 1:K) {
    cov_matrix <- exp(-0.5 * var_scales_lambda[k] * (time_diff ^ 2) / length_scales_lambda[k] ^ 2)
    for (i in 1:N) {
      mean_lambda <- g_i[i, ] %*% Gamma_k[k, ]
      lambda_ik[i, k, ] <- mvrnorm(1, mu = rep(mean_lambda, T), Sigma = cov_matrix)
    }
  }
  
  # Apply softmax to lambda
  theta <- apply(lambda_ik, c(1,3), function(x) exp(x) / sum(exp(x)))
  theta <- aperm(theta, c(2,1,3))  # Reorder dimensions to match original lambda_ik
  
  # Simulate phi
  for (k in 1:K) {
    cov_matrix <- exp(-0.5 * var_scales_phi[k] * (time_diff ^ 2) / length_scales_phi[k] ^ 2)
    for (d in 1:D) {
      phi_kd[k, d, ] <- mvrnorm(1, mu = mu_d[d,], Sigma = cov_matrix) ## maximum rank of this matrix is Ttot
    }
  }
  
  eta <- plogis(phi_kd)
  # Generate pi and Y
  # Assuming theta and eta are already defined
  pi_temp <- einsum('nkt,kdt->ndt', theta, eta)
  
  N <- dim(pi_temp)[1]
  D <- dim(pi_temp)[2]
  T <- dim(pi_temp)[3]
  
  # Initialize arrays
  Y <- array(0, dim = c(N, D, T))
  S <- matrix(T, nrow = N, ncol = D)  # Initialize all to last time point (censored)
  pi_values <- array(NA, dim = c(N, D, T))
  
  for (i in 1:N) {
    for (d in 1:D) {
      for (t in 1:T) {
        if (t == 1 || sum(Y[i, d, 1:(t-1)]) == 0) {
          # Disease hasn't occurred yet
          pi_values[i, d, t] <- pi_temp[i, d, t]
          
          if (runif(1) < pi_temp[i, d, t]) {
            Y[i, d, t] <- 1
            S[i, d] <- t
            break  # Stop at first occurrence of the event
          }
        }
        # If disease has occurred, pi_values remains NA
      }
    }
  }
  
  
  # Return all generated data and parameters
  return(list(
    Y = Y,
    G = g_i,
    var_scales_lambda = var_scales_lambda,
    length_scales_lambda = length_scales_lambda,
    var_scales_phi = var_scales_phi,
    length_scales_phi = length_scales_phi,
    S=S,
    mu_d = mu_d,
    lambda_ik = lambda_ik,
    phi_kd = phi_kd,
    Gamma_k = Gamma_k,
    pi = pi_values,
    pi_temp=pi_temp,
    theta = theta,
    eta = eta
  ))
}


plot_individuals=function(S, num_individuals = 3) {
  N=nrow(S)
  D=ncol(S)
  Ttot=max(S)
  par(mfrow = c(num_individuals, 1), mar = c(4, 4, 2, 1))
  for (i in 1:num_individuals) {
    individual <- sample(1:N, 1)
    event_matrix <- matrix(0, nrow = D, ncol = Ttot)
    for (d in 1:D) {
      event_time <- S[individual, d]
      if (event_time < Ttot) {  # Check if event occurred before T
        event_matrix[d, event_time + 1] <- 1  # +1 because R is 1-indexed
      }
    }
    image(1:Ttot, 1:D, t(event_matrix), xlab = "Time", ylab = "Disease", 
          main = paste("Individual", individual), 
          col = c("white", "red"))
  }
}