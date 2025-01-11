# mcmc_sampler
### TO DO:
## Update one component at a time, 

mcmc_sampler_softmax <- function(y, g_i, n_iterations, initial_values) {
  current_state <- initial_values
  n_individuals <- dim(current_state$Lambda)[1]
  n_topics <- dim(current_state$Lambda)[2]
  T <- dim(current_state$Lambda)[3]
  n_diseases <- dim(current_state$Phi)[2]
  P <- ncol(g_i)
  
  # Initialize storage for samples and diagnostics
  samples <- list(
    Lambda = array(0, dim = c(
      n_iterations, dim(current_state$Lambda)
    )),
    Phi = array(0, dim = c(n_iterations, dim(
      current_state$Phi
    ))),
    Gamma = array(0, dim = c(
      n_iterations, dim(current_state$Gamma)
    ))
  )
  log_likelihoods <- numeric(n_iterations)
  log_posteriors <- numeric(n_iterations)
  acceptance_rates <- list(Lambda = 0, Phi = 0)
  
  # Initialize proposal standard deviations
  adapt_sd <- list(Lambda = array(0.01, dim = dim(current_state$Lambda)),
                   Phi = array(0.01, dim = dim(current_state$Phi)))
  
  # Precompute inverse covariance matrices
  K_inv_lambda <- lapply(1:n_topics, function(k)
    precompute_K_inv(
      T,
      current_state$length_scales_lambda[k],
      current_state$var_scales_lambda[k]
    ))
  K_inv_phi <- lapply(1:n_topics, function(k)
    precompute_K_inv(
      T,
      current_state$length_scales_phi[k],
      current_state$var_scales_phi[k]
    ))
  
  for (iter in 1:n_iterations) {
    # Update Lambda
    proposed_Lambda <- current_state$Lambda + array(rnorm(prod(dim(
      current_state$Lambda
    )), 0, adapt_sd$Lambda),
    dim = dim(current_state$Lambda))
    
    ## update lambda one component at a time
    ## create a function called 'update lambda', the number of metropolis steps you take should be either equal to 
    ## the number of individuals x K: for each vector you propose, accept or reject it,
    ## right way to do it is with the variance covariance matrix of the proposal (Roberts/Rosenthal 2009)
    ## the other way is to update all the components at once, but this is not the right way to do it
    ## update vectors, each one at a time 
    ## update the whole matrix at once using the covariance matrix of the chain, 
    ## multipling by 2.38^2, related to diffusion processes, way to get the optimal accept/reject ratio
    
    
    current_log_lik <- log_likelihood(y, current_state$Lambda, current_state$Phi)
    proposed_log_lik <- log_likelihood(y, proposed_Lambda, current_state$Phi)
    
    
    current_log_prior_lambda <- log_sum_exp(unlist(sapply(1:n_individuals, function(i) {
      sapply(1:n_topics, function(k) {
        log_gp_prior_vec(
          current_state$Lambda[i, k, ],
          rep(g_i[i, ] %*% current_state$Gamma[k, ], T),
          K_inv_lambda[[k]]$K_inv,
          K_inv_lambda[[k]]$log_det_K
        )
      })
    })))
    
    proposed_log_prior_lambda <- log_sum_exp(unlist(sapply(1:n_individuals, function(i) {
      sapply(1:n_topics, function(k) {
        log_gp_prior_vec(
          proposed_Lambda[i, k, ],
          rep(g_i[i, ] %*% current_state$Gamma[k, ], T),
          K_inv_lambda[[k]]$K_inv,
          K_inv_lambda[[k]]$log_det_K
        )
      })
    })))
    
    log_accept_ratio <- (proposed_log_lik + proposed_log_prior_lambda) -
      (current_log_lik + current_log_prior_lambda)
    
    if (log(runif(1)) < log_accept_ratio) {
      current_state$Lambda <- proposed_Lambda
      adapt_sd$Lambda <- adapt_sd$Lambda * 1.01
      acceptance_rates$Lambda <- acceptance_rates$Lambda + 1
    } else {
      adapt_sd$Lambda <- adapt_sd$Lambda * 0.99
    }
    
    

    
    # Update Phi
    proposed_Phi <- current_state$Phi + array(rnorm(prod(dim(
      current_state$Phi
    )), 0, adapt_sd$Phi), dim = dim(current_state$Phi))
    
    current_log_lik <- log_likelihood(y, current_state$Lambda, current_state$Phi)
    proposed_log_lik <- log_likelihood(y, current_state$Lambda, proposed_Phi)
    
    
    
    current_log_prior_phi <- log_sum_exp(unlist(sapply(1:n_topics, function(k) {
      sapply(1:n_diseases, function(d) {
        log_gp_prior_vec(
          current_state$Phi[k, d, ],
          current_state$mu_d[d, ],
          K_inv_phi[[k]]$K_inv,
          K_inv_phi[[k]]$log_det_K
        )
      })
    })))
    
    proposed_log_prior_phi <- log_sum_exp(unlist(sapply(1:n_topics, function(k) {
      sapply(1:n_diseases, function(d) {
        log_gp_prior_vec(
          proposed_Phi[k, d, ],
          current_state$mu_d[d, ],
          K_inv_phi[[k]]$K_inv,
          K_inv_phi[[k]]$log_det_K
        )
      })
    })))
    
    log_accept_ratio <- (proposed_log_lik + proposed_log_prior_phi) -
      (current_log_lik + current_log_prior_phi)
    
    if (log(runif(1)) < log_accept_ratio) {
      current_state$Phi <- proposed_Phi
      adapt_sd$Phi <- adapt_sd$Phi * 1.01
      acceptance_rates$Phi <- acceptance_rates$Phi + 1
    } else {
      adapt_sd$Phi <- adapt_sd$Phi * 0.99
    }
    

    # Update Gamma using Gibbs sampler
    for (k in 1:n_topics) {
      Lambda_k <- current_state$Lambda[, k, ]  # N x T matrix for topic k
      K_inv <- K_inv_lambda[[k]]$K_inv  # T x T inverse covariance matrix
      
      # Compute posterior precision (inverse covariance), see standard MVN derivatino using design matrix on X instead of N
      posterior_precision <- diag(1, P)  # Prior precision (assuming standard normal prior, because we're asumming gamma_kp is N(0,1)
      posterior_mean <- rep(0, P)  # Prior mean
      
      for (i in 1:N) {
        Xi <- matrix(rep(g_i[i, ], T), nrow = T, byrow = TRUE)  # T x P matrix
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
    log_posteriors[iter] <- log_sum_exp(c(current_log_lik + current_log_prior_lambda + current_log_prior_phi +
                                            sum(dnorm(current_state$Gamma, 0, 1, log = TRUE))))
    
    cat("current_log_lik:", current_log_lik, "\n")
    cat("current_log_prior_lambda:",
        current_log_prior_lambda,
        "\n")
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
    acceptance_rates[[param]] <- acceptance_rates[[param]] / n_iterations
  }
  
  return(
    list(
      samples = samples,
      log_likelihoods = log_likelihoods,
      acceptance_rates = acceptance_rates,
      log_posteriors = log_posteriors
    )
  )
}