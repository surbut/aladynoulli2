


aladynoulli <- function(Y, G, n_topics = 3,n_iters,initial_values,step_size_lambda=0.01, step_size_phi=0.01, target_accept_rate = 0.2) {
  
  
  N = n_individuals <- dim(Y)[1]  # Number of individuals
  D <- dim(Y)[2]  # Number of diseases
  Ttot <- dim(Y)[3]  # Number of time points
  P <- ncol(G)  # Number of genetic covariates
  K <- n_topics   # Number of topics
  n_diseases=D
  
  
  # Matrix of indexed to ignore. 
  # Create a matrix of the indexes for the time-to-event for each patient-disease
  
  ## start from the truth 
  ### reasonable to just do two
  precomputed_indices <- precompute_likelihood_indices(Y)
  
  # Here you initialize the MCMC
  #initial_values <- mcmc_init_two(y = Y, G = G)
  current_state=initial_values
  var_scales_lambda=current_state$var_scales_lambda
  length_scales_lambda=current_state$length_scales_lambda
  var_scales_phi=current_state$var_scales_phi
  length_scales_phi=current_state$length_scales_phi

  K_lambda <- lapply(1:n_topics, function(k) {
    time_diff_matrix <- outer(1:Ttot, 1:Ttot, "-") ^ 2
    var_scales_lambda[k] * exp(-0.5 * time_diff_matrix / length_scales_lambda[k] ^
                                 2) + diag(1e-6, Ttot)
  })
  K_phi <- lapply(1:n_topics, function(k) {
    time_diff_matrix <- outer(1:Ttot, 1:Ttot, "-") ^ 2
    var_scales_phi[k] * exp(-0.5 * time_diff_matrix / length_scales_phi[k] ^
                              2) + diag(1e-6, Ttot)
  })
  
  chol_lambda <- lapply(K_lambda, chol)
  chol_phi <- lapply(K_phi, chol)
  
  # Precompute log determinants and inverses for the log_gp_prior_vec function
  K_inv_lambda <- lapply(K_lambda, function(K) {
    K_inv <- solve(K)
    log_det_K <- determinant(K, logarithm = TRUE)$modulus
    list(K_inv = K_inv, log_det_K = log_det_K)
  })
  K_inv_phi <- lapply(K_phi, function(K) {
    K_inv <- solve(K)
    log_det_K <- determinant(K, logarithm = TRUE)$modulus
    list(K_inv = K_inv, log_det_K = log_det_K)
  })



  # Initialize storage for samples and diagnostics
  samples <- list(
    Lambda = array(0, dim = c(
      n_iters, dim(current_state$Lambda)
    )),
    Phi = array(0, dim = c(n_iters, dim(
      current_state$Phi
    ))),
    Gamma = array(0, dim = c(
      n_iters, dim(current_state$Gamma)
    ))
  )
  log_likelihoods <- numeric(n_iters)
  log_posteriors <- numeric(n_iters)
  acceptance_rates <- list(Lambda = 0, Phi = 0)
  
  
  # Initialize acceptance counters
  acceptance_counters <- list(Lambda = 0, Phi = 0)
  total_proposals <- list(Lambda = 0, Phi = 0)

  current_state=initial_values
  for (iter in 1:n_iters) {
    # Update Lambda
    for (i in 1:n_individuals) {
      for (k in 1:n_topics) {
        # Efficient sampling from GP prior
        
        z <- rnorm(Ttot)
        perturbation <- step_size_lambda * drop(chol_lambda[[k]] %*% z) ##simpler than sampling from MVRNORM
        proposed_Lambda_ik <- current_state$Lambda[i,k,]+ perturbation  
        
        # Calculate log-likelihood and log-prior for current and proposed states
        current_log_lik <- compute_log_likelihood(current_state$Lambda,
                                                  current_state$Phi,
                                                  precomputed_indices)
        
        
        proposed_Lambda <- current_state$Lambda
        proposed_Lambda[i, k, ] <- proposed_Lambda_ik
        proposed_log_lik <- compute_log_likelihood(proposed_Lambda,
                                                   current_state$Phi,
                                                   precomputed_indices)
        
        mean_lambda=rep(G[i, ] %*% current_state$Gamma[k, ],Ttot)
        
        current_log_prior_lambda <- log_gp_prior_vec(
          current_state$Lambda[i, k, ],
          mean_lambda,
          K_inv_lambda[[k]]$K_inv,
          K_inv_lambda[[k]]$log_det_K
        )
        proposed_log_prior_lambda <- log_gp_prior_vec(
          proposed_Lambda_ik,
          mean_lambda,
          K_inv_lambda[[k]]$K_inv,
          K_inv_lambda[[k]]$log_det_K
        )

        
        # Calculate acceptance ratio
        log_accept_ratio <- (proposed_log_lik + proposed_log_prior_lambda) - (current_log_lik + current_log_prior_lambda)
        
        if (log(runif(1)) < log_accept_ratio) {
          current_state$Lambda[i, k, ] <- proposed_Lambda_ik
          acceptance_counters$Lambda <- acceptance_counters$Lambda + 1
          #c(print(paste0("accept!", iter)))
        }
        total_proposals$Lambda <- total_proposals$Lambda + 1
      }
    }

    
    
    
    
    # Update Phi (similar changes as for Lambda)
    for (k in 1:n_topics) {
      for (d in 1:n_diseases) {
        z <- rnorm(Ttot)
        perturbation <- step_size_phi * drop(chol_phi[[k]] %*% z)
        proposed_Phi_kd <- current_state$Phi[k,d,] + perturbation
        
        current_log_lik <- compute_log_likelihood(current_state$Lambda,
                                                  current_state$Phi,
                                                  precomputed_indices)
        proposed_Phi <- current_state$Phi
        proposed_Phi[k, d, ] <- proposed_Phi_kd
        proposed_log_lik <- compute_log_likelihood(current_state$Lambda,
                                                   proposed_Phi,
                                                   precomputed_indices)
        
        current_log_prior_phi <- log_gp_prior_vec(
          current_state$Phi[k, d, ],
          current_state$mu_d[d, ],
          K_inv_phi[[k]]$K_inv,
          K_inv_phi[[k]]$log_det_K
        )
        proposed_log_prior_phi <- log_gp_prior_vec(
          proposed_Phi_kd,
          current_state$mu_d[d, ],
          K_inv_phi[[k]]$K_inv,
          K_inv_phi[[k]]$log_det_K
        )
        
        log_accept_ratio <- (proposed_log_lik + proposed_log_prior_phi) - (current_log_lik + current_log_prior_phi)
        
        if (log(runif(1)) < log_accept_ratio) {
          current_state$Phi[k, d, ] <- proposed_Phi_kd
          acceptance_counters$Phi <- acceptance_counters$Phi + 1

          #print("accepted! :)")
        }
        total_proposals$Phi <- total_proposals$Phi + 1
      }
    }
    
    
    # Adapt step sizes every 100 iterations
    # Adapt step sizes every 100 iterations
    if (iter > 100 && iter %% 100 == 0) {
      current_accept_rate_lambda <- acceptance_counters$Lambda / total_proposals$Lambda
      current_accept_rate_phi <- acceptance_counters$Phi / total_proposals$Phi
      
      step_size_lambda <- step_size_lambda * exp(current_accept_rate_lambda - target_accept_rate)
      step_size_phi <- step_size_phi * exp(current_accept_rate_phi - target_accept_rate)
    }

    # When printing progress
    cat(
      "Acceptance counters: Lambda =",
      acceptance_counters$Lambda ,#/ total_proposals$Lambda,
      "Phi =",
      acceptance_counters$Phi, #/ total_proposals$Phi,
      "\n"
    )
    
    
    # Update Gamma using Gibbs sampler
    for (k in 1:n_topics) {
      Lambda_k <- current_state$Lambda[, k, ]  # N x T matrix for topic k
      K_inv <- K_inv_lambda[[k]]$K_inv  # T x T inverse covariance matrix
      
      # Compute posterior precision (inverse covariance), see standard MVN derivatino using design matrix on X instead of N
      posterior_precision <- diag(1, P)  # Prior precision (assuming standard normal prior, because we're asumming gamma_kp is N(0,1)
      posterior_mean <- rep(0, P)  # Prior mean
      
      for (i in 1:N) {
        Xi <- matrix(rep(G[i, ], Ttot), nrow = Ttot, byrow = TRUE)  # T x P matrix
        precision_contrib <- t(Xi) %*% K_inv %*% Xi
        posterior_precision <- posterior_precision + precision_contrib
        posterior_mean <- posterior_mean + t(Xi) %*% K_inv %*% Lambda_k[i, ]
      }
      
      # Compute posterior covariance and mean
      posterior_covariance <- solve(posterior_precision, tol = 1e-20)
      posterior_mean <- posterior_covariance %*% posterior_mean
      
      # Sample new Gamma_k
      current_state$Gamma[k, ] <- mvrnorm(1, mu = posterior_mean, Sigma = posterior_covariance)
    }
    
    
    # Store samples and diagnostics
    samples$Lambda[iter, , , ] <- current_state$Lambda
    samples$Phi[iter, , , ] <- current_state$Phi
    samples$Gamma[iter, , ] <- current_state$Gamma
    
    log_likelihoods[iter] <- current_log_lik
    log_posteriors[iter] <- log_sum_exp(c(
      current_log_lik + current_log_prior_lambda + current_log_prior_phi +
        sum(dnorm(current_state$Gamma, 0, 1, log = TRUE))
    ))
    
    cat("current_log_lik:", current_log_lik, "\n")
    cat("current_log_prior_lambda:", current_log_prior_lambda, "\n")
    cat("current_log_prior_phi:", current_log_prior_phi, "\n")
    cat("log_prior_gamma:", sum(dnorm(current_state$Gamma, 0, 1, log = TRUE)), "\n")
    
  # Print progress
  #if (iter %% 10 == 0) {
  cat(
    "Iteration",
    iter,
    "Log posterior:",
    log_posteriors[iter],
    "Log-likelihood:",
    log_likelihoods[iter],
    "\n"
  )
  cat(
    "Acceptance rates: Lambda =",
    acceptance_rates$Lambda / iter,
    "Phi =",
    acceptance_rates$Phi / iter,
    "\n"
  )
  #}
}

# Calculate final acceptance rates
for (param in names(acceptance_rates)) {
  acceptance_rates[[param]] <- acceptance_rates[[param]] / n_iters
}

return(
  list(
    samples = samples,
    log_likelihoods = log_likelihoods,
    acceptance_rates = list(
      Lambda = acceptance_counters$Lambda / total_proposals$Lambda,
      Phi = acceptance_counters$Phi / total_proposals$Phi
    ),
    log_posteriors = log_posteriors
  )
)}



#### Usage ###

# data <- generate_tensor_data(num_covariates = 5,K = 3,T = 20,D = 5,N = 100)
# # 
# Y <- data$Y
# G <- data$G
# plot_individuals(data$S,num_individuals = 3)
# # Here you initialize the MCMC
# initial_values <- mcmc_init_two(y = Y, G = G, num_topics = 3, length_scales_lambda = rep(10, 3),
#                                 var_scales_lambda = rep(1, 3),
#                                 length_scales_phi = rep(10, 3),
#                                 var_scales_phi = rep(1, 3))
# a=aladynoulli(Y, G, n_topics = 3,n_iters = 5000,initial_values = initial_values)
# # 
# # 
