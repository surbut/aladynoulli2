library(rsvd)  # For fast randomized SVD
library(mgcv)  # For Gaussian Process utilities
library(MASS)  # For mvrnorm function

# Helper functions with numerical stability
softmax <- function(x) {
  x <- pmin(pmax(x, -100), 100)  # Clip values
  exp_x <- exp(x - max(x))
  exp_x / sum(exp_x)
}

sigmoid <- function(x) {
  x <- pmin(pmax(x, -100), 100)  # Clip values
  1 / (1 + exp(-x))
}



# Gradient computation functions per element
# compute_gradient_log_likelihood_lambda <- function(Lambda, Phi, Y, i, k, t) {
#   D <- dim(Phi)[2]
#   K <- dim(Lambda)[2]
#   
#   gradient <- 0
#   for (d in 1:D) {
#     theta <- softmax(Lambda[i,,t])
#     pi_idt <- sum(theta * sigmoid(Phi[,d,t]))
#     dL_dpi <- Y[i,d,t]/pi_idt - (1-Y[i,d,t])/(1-pi_idt)
#     dpi_dlambda <- sigmoid(Phi[k,d,t]) * theta[k] * (1 - theta[k])
#     gradient <- gradient + dL_dpi * dpi_dlambda
#   }
#   
#   return(gradient)
# }
# 
# compute_gradient_log_likelihood_phi <- function(Lambda, Phi, Y, k, d, t) {
#   N <- dim(Lambda)[1]
#   
#   gradient <- 0
#   for (i in 1:N) {
#     theta <- softmax(Lambda[i,,t])
#     pi_idt <- sum(theta * sigmoid(Phi[,d,t]))
#     dL_dpi <- Y[i,d,t]/pi_idt - (1-Y[i,d,t])/(1-pi_idt)
#     dpi_dphi <- theta[k] * sigmoid(Phi[k,d,t]) * (1 - sigmoid(Phi[k,d,t]))
#     gradient <- gradient + dL_dpi * dpi_dphi
#   }
#   
#   return(gradient)
# }
# 
# compute_gradient_log_prior_lambda <- function(Lambda_i, K_inv_lambda) {
#   return(-K_inv_lambda %*% Lambda_i)
# }
# 
# compute_gradient_log_prior_phi <- function(Phi_k, K_inv_phi) {
#   return(-K_inv_phi %*% Phi_k)
# }
# 
# # SGLD update functions
# update_lambda <- function(Lambda, Phi, Y, K_inv_lambda, i, k, t, epsilon) {
#   grad_likelihood <- compute_gradient_log_likelihood_lambda(Lambda, Phi, Y, i, k, t)
#   grad_prior <- (K_inv_lambda %*% Lambda[i,k,])[t]
#   full_grad <- grad_likelihood - grad_prior
#   
#   eta <- rnorm(1, 0, sqrt(epsilon))
#   Lambda[i,k,t] <- Lambda[i,k,t] + 0.5 * epsilon * full_grad + eta
#   
#   return(Lambda)
# }
# 
# update_phi <- function(Lambda, Phi, Y, K_inv_phi, k, d, t, epsilon) {
#   grad_likelihood <- compute_gradient_log_likelihood_phi(Lambda, Phi, Y, k, d, t)
#   grad_prior <- (K_inv_phi %*% Phi[k,d,])[t]
#   full_grad <- grad_likelihood - grad_prior ## bc you're differentiatng w.r.t theta
#   
#   eta <- rnorm(1, 0, sqrt(epsilon))
#   Phi[k,d,t] <- Phi[k,d,t] + 0.5 * epsilon * full_grad + eta
#   
#   return(Phi)
# }

### Now do for the whole vector 
compute_gradient_log_likelihood_lambda <- function(Lambda, Phi, Y, i, k) {
  D <- dim(Phi)[2]
  K <- dim(Lambda)[2]
  Ttot <- dim(Lambda)[3]
  
  # Initialize gradient vector
  gradient <- numeric(Ttot)
  
  # Compute for each time point
  for(t in 1:Ttot) {
    theta <- softmax(Lambda[i,,t])  # K-dimensional vector
    pi_id <- sapply(1:D, function(d) sum(theta * sigmoid(Phi[,d,t])))  # D-dimensional vector
    
    dL_dpi <- Y[i,,t] / pi_id - (1 - Y[i,,t]) / (1 - pi_id)  # D-dimensional vector
    dpi_dlambda <- sapply(1:D, function(d) sigmoid(Phi[k,d,t]) * theta[k] * (1 - theta[k]))  # D-dimensional vector
    
    gradient[t] <- sum(dL_dpi * dpi_dlambda)
  }
  
  return(gradient)
}

compute_gradient_log_likelihood_phi <- function(Lambda, Phi, Y, k, d) {
  N <- dim(Lambda)[1]
  Ttot <- dim(Lambda)[3]
  
  # Initialize gradient vector
  gradient <- numeric(Ttot)
  
  # Compute for each time point
  for(t in 1:Ttot) {
    theta <- sapply(1:N, function(i) softmax(Lambda[i,,t]))  # K x N matrix
    pi_idt <- sapply(1:N, function(i) sum(theta[,i] * sigmoid(Phi[,d,t])))  # N-dimensional vector
    
    dL_dpi <- Y[,d,t] / pi_idt - (1 - Y[,d,t]) / (1 - pi_idt)  # N-dimensional vector
    dpi_dphi <- sapply(1:N, function(i) theta[k,i] * sigmoid(Phi[k,d,t]) * (1 - sigmoid(Phi[k,d,t])))  # N-dimensional vector
    
    gradient[t] <- sum(dL_dpi * dpi_dphi)
  }
  
  return(gradient)
}

# Update functions with smaller step sizes and value clipping
update_lambda <- function(Lambda, Phi, Y, K_inv_lambda, i, k, epsilon) {
  grad_likelihood <- compute_gradient_log_likelihood_lambda(Lambda, Phi, Y, i, k)
  grad_prior <- K_inv_lambda %*% Lambda[i,k,]
  
  # Clip gradients
  grad_likelihood <- pmin(pmax(grad_likelihood, -100), 100)
  grad_prior <- pmin(pmax(grad_prior, -100), 100)
  
  full_grad <- grad_likelihood - grad_prior
  
  # Smaller noise term
  eta <- rnorm(length(full_grad), 0, sqrt(epsilon) * 0.1)
  
  # Update with clipping
  Lambda[i,k,] <- Lambda[i,k,] + 0.5 * epsilon * full_grad + eta
  Lambda[i,k,] <- pmin(pmax(Lambda[i,k,], -10), 10)  # Clip values
  
  return(Lambda)
}

update_phi <- function(Lambda, Phi, Y, K_inv_phi, k, d, epsilon) {
  grad_likelihood <- compute_gradient_log_likelihood_phi(Lambda, Phi, Y, k, d)
  grad_prior <- K_inv_phi %*% Phi[k,d,]
  
  # Clip gradients
  grad_likelihood <- pmin(pmax(grad_likelihood, -100), 100)
  grad_prior <- pmin(pmax(grad_prior, -100), 100)
  
  full_grad <- grad_likelihood - grad_prior
  
  # Smaller noise term
  eta <- rnorm(length(full_grad), 0, sqrt(epsilon) * 0.1)
  
  # Update with clipping
  Phi[k,d,] <- Phi[k,d,] + 0.5 * epsilon * full_grad + eta
  Phi[k,d,] <- pmin(pmax(Phi[k,d,], -10), 10)  # Clip values
  
  return(Phi)
}
# Main Aladynoulli function
aladynoulli_langevin <- function(Y, G, n_topics = 3, n_iters = 1000, step_size_lambda = 0.01, step_size_phi = 0.01, 
                        length_scales_lambda, var_scales_lambda, length_scales_phi, var_scales_phi) {
  N <- dim(Y)[1]  # Number of individuals
  D <- dim(Y)[2]  # Number of diseases
  Ttot <- dim(Y)[3]  # Number of time points
  P <- ncol(G)  # Number of genetic covariates
  K <- n_topics   # Number of topics
  
  # Initialize parameters using mcmc_init_two
  init_state <- mcmc_init_two(Y, G, n_topics, length_scales_lambda, var_scales_lambda, length_scales_phi, var_scales_phi)
  
  Lambda <- init_state$Lambda
  Phi <- init_state$Phi
  Gamma <- init_state$Gamma
  
  # Set up Gaussian Process priors
  K_lambda <- lapply(1:K, function(k) {
    time_diff_matrix <- outer(1:Ttot, 1:Ttot, "-") ^ 2
    var_scales_lambda[k] * exp(-0.5 * time_diff_matrix / length_scales_lambda[k]^2) + diag(1e-6, Ttot)
  })
  K_phi <- lapply(1:K, function(k) {
    time_diff_matrix <- outer(1:Ttot, 1:Ttot, "-") ^ 2
    var_scales_phi[k] * exp(-0.5 * time_diff_matrix / length_scales_phi[k]^2) + diag(1e-6, Ttot)
  })
  
  K_inv_lambda <- lapply(K_lambda, solve)
  K_inv_phi <- lapply(K_phi, solve)
  
  # Storage for samples and diagnostics
  samples <- list(
    Lambda = array(0, dim = c(n_iters, dim(Lambda))),
    Phi = array(0, dim = c(n_iters, dim(Phi))),
    Gamma = array(0, dim = c(n_iters, dim(Gamma)))
  )
  log_likelihoods <- numeric(n_iters)
  log_priors_lambda <- numeric(n_iters)
  log_priors_phi <- numeric(n_iters)
  

    
    # Main MCMC loop
    for (iter in 1:n_iters) {
      # Compute log-priors
      log_prior_lambda <- 0
      log_prior_phi <- 0
      for (k in 1:K) {
        # For each individual and topic, compute GP prior
        for (i in 1:N) {
          log_prior_lambda <- log_prior_lambda + 
            dmvnorm(Lambda[i,k,], mean = rep(0, Ttot), sigma = K_lambda[[k]], log = TRUE)
        }
        # For each disease and topic, compute GP prior
        for (d in 1:D) {
          log_prior_phi <- log_prior_phi + 
            dmvnorm(Phi[k,d,], mean = rep(0, Ttot), sigma = K_phi[[k]], log = TRUE)
        }
      }
      log_priors_lambda[iter] <- log_prior_lambda
      log_priors_phi[iter] <- log_prior_phi

    # Update Lambda
    for (i in 1:N) {
      for (k in 1:K) {
        Lambda <- update_lambda(Lambda, Phi, Y, K_inv_lambda[[k]], i, k, step_size_lambda)
      }
    }
    
    # Update Phi
    for (k in 1:K) {
      for (d in 1:D) {
        Phi <- update_phi(Lambda, Phi, Y, K_inv_phi[[k]], k, d, step_size_phi)
      }
    }
    
    # Update Gamma using Gibbs sampler
    for (k in 1:K) {
      Lambda_k <- Lambda[, k, ]  # N x T matrix for topic k
      K_inv <- K_inv_lambda[[k]]  # T x T inverse covariance matrix
      
      # For each individual, we have T observations
      # Need to reshape for the regression
      posterior_precision <- t(G) %*% G + diag(1, P)  # P x P
      posterior_mean <- solve(posterior_precision, 
                            t(G) %*% Lambda_k)  # P x T matrix
      
      # Sample new Gamma_k
      for(t in 1:Ttot) {
        Gamma[k,] <- mvrnorm(1, 
                            mu = posterior_mean[,t], 
                            Sigma = solve(posterior_precision))
      }
    }
    
    # Store samples
    samples$Lambda[iter, , , ] <- Lambda
    samples$Phi[iter, , , ] <- Phi
    samples$Gamma[iter, , ] <- Gamma
    
    # Compute log-likelihood
    log_lik <- 0
    for (i in 1:N) {
      for (d in 1:D) {
        for (t in 1:Ttot) {
          pi_idt <- sum(softmax(Lambda[i,,t]) * sigmoid(Phi[,d,t]))
          log_lik <- log_lik + Y[i,d,t] * log(pi_idt) + (1 - Y[i,d,t]) * log(1 - pi_idt)
        }
      }
    }
    log_likelihoods[iter] <- log_lik
    
    # Print progress
    if (iter %% 100 == 0) {
      cat("Iteration", iter, "Log-likelihood:", log_lik, 
          "Log-prior Lambda:", log_prior_lambda, 
          "Log-prior Phi:", log_prior_phi, "\n")
    }
  }
  
  return(list(
    samples = samples,
    log_likelihoods = log_likelihoods,
    log_priors_lambda = log_priors_lambda,
    log_priors_phi = log_priors_phi,
    init_state = init_state
  ))
}

# Example usage
set.seed(123)
N <- 100  # Number of individuals
D <- 5    # Number of diseases
Ttot <- 10  # Number of time points
P <- 3    # Number of genetic covariates
n_topics <- 3

# Generate synthetic data


# data <- generate_tensor_data(num_covariates = P,K = n_topics,T = Ttot,D = D,N =N)
# 
# Y <- data$Y
# G <- data$G
# plot_individuals(data$S,num_individuals = 3)
# # Here you initialize the MCMC
# 
# 
# # Run the model
# result <- aladynoulli(Y, G, n_topics = n_topics, n_iters = 1000,
#                       length_scales_lambda = length_scales_lambda,
#                       var_scales_lambda = var_scales_lambda,
#                       length_scales_phi = length_scales_phi,
#                       var_scales_phi = var_scales_phi)
# 
# # Plot log-likelihood trace
# plot(result$log_likelihoods, type = "l", xlab = "Iteration", ylab = "Log-likelihood")

