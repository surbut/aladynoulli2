## model specific functions

# Precompute indices function, run only once
precompute_likelihood_indices <- function(Y) {
  n_individuals <- dim(Y)[1]
  n_diseases <- dim(Y)[2]
  Ttot <- dim(Y)[3]
  
  # Initialize lists to store indices
  event_indices <- list()
  at_risk_indices <- list()
  
  for (i in 1:n_individuals) {
    for (d in 1:n_diseases) {
      event_time <- which(Y[i, d, ] == 1)[1]  # Take only the first (and only) event time
      if (!is.na(event_time)) {
        # Add the event time
        event_indices <- c(event_indices, list(c(i, d, event_time)))
        
        # Add at-risk times up to the event
        if (event_time > 1) {
          at_risk_indices <- c(at_risk_indices, list(cbind(i, d, 1:(event_time-1))))
        }
      } else {
        # If no event, consider at risk for all time points
        at_risk_indices <- c(at_risk_indices, list(cbind(i, d, 1:Ttot)))
      }
    }
  }
  
  list(event_indices = do.call(rbind, event_indices),
       at_risk_indices = do.call(rbind, at_risk_indices))
}


# Updated log-likelihood function
compute_log_likelihood <- function(Lambda, Phi, precomputed_indices) {
  n_individuals <- dim(Lambda)[1]
  n_topics <- dim(Lambda)[2]
  n_diseases <- dim(Phi)[2]
  Ttot <- dim(Lambda)[3]
  
  theta <- apply_softmax_to_lambda(Lambda)
  eta <- plogis(Phi)  # logistic function
  pi <- einsum::einsum("nkt,kdt->ndt", theta, eta)
  
  event_indices <- precomputed_indices$event_indices
  at_risk_indices <- precomputed_indices$at_risk_indices
  
  log_lik <- 0
  
  # Handle events, including those at time 1
  if (nrow(event_indices) > 0) {
    log_lik <- log_lik + sum(log(pi[event_indices]))
  }
  
  # Handle at-risk periods
  if (nrow(at_risk_indices) > 0) {
    log_lik <- log_lik + sum(log(1 - pi[at_risk_indices]))
  }
  
  return(log_lik)
}




precompute_K_inv <- function(T, length_scale, var_scale) {
  time_diff_matrix <- outer(1:T, 1:T, "-")^2
  Kern <- var_scale * exp(-0.5 * time_diff_matrix / length_scale^2)
  Kern <- Kern + diag(1e-6, T)  # Add small jitter for numerical stability
  K_inv <- solve(Kern)
  log_det_K <- determinant(Kern, logarithm = TRUE)$modulus
  #cat("K_inv diagonal:", diag(K_inv)[1:5], "log_det_K:", log_det_K, "\n")  # Add this line
  return(list(K_inv = K_inv, log_det_K = log_det_K))
}

log_gp_prior_vec <- function(eta, mean, K_inv, log_det_K) {
  T <- length(eta)
  centered_eta <- eta - mean
  quad_form <- sum(centered_eta * (K_inv %*% centered_eta))
  log_prior <- -0.5 * (log_det_K + quad_form + T * log(2 * base::pi))
  #cat("log_det_K:", log_det_K, "quad_form:", quad_form, "T:", T, "log_prior:", log_prior, "\n")
  return(log_prior)
}


# Version using kernel matrix directly
log_gp_prior_vec_direct <- function(eta, mean, K) {
  centered_eta <- eta - mean
  log_det_K <- determinant(K, logarithm = TRUE)$modulus
  quad_form <- t(centered_eta) %*% solve(K, centered_eta)
  log_prior <- -0.5 * (log_det_K + as.numeric(quad_form) + length(eta) * log(2 * pi))
  return(as.numeric(log_prior))
}

# update_phi <- function(Phi, k, d, new_values) {
#   Phi_copy <- Phi
#   Phi_copy[k, d, ] <- new_values
#   return(Phi_copy)
# }
# 
# 
# update_lambda <- function(Lambda, i, k, new_values) {
#   Lambda_copy <- Lambda
#   Lambda_copy[i, k, ] <- new_values
#   return(Lambda_copy)
# }


