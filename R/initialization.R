
# Initialization Function
initialize_mcmc <- function(y, G, n_topics, n_diseases, T,length_scales_lambda,length_scales_phi,var_scales_lambda,var_scales_phi,sigsmall) {
  N <- dim(y)[1]  # Number of individuals
  P <- ncol(G)  # Number of genetic covariates
  
  
  time_diff <- outer(seq_len(T), seq_len(T), "-")
  
  Gamma_init <- matrix(rnorm(n_topics * P, mean = 0, sd = 1),
                       nrow = n_topics,
                       ncol = P)
  
  lambda_init <- array(0, dim = c(N, n_topics, T))
  K=n_topics
  for (k in 1:K) {
    # Simulate lambda_ik(t) using a different covariance matrix for each topic
    cov_matrix <- exp(-0.5 * var_scales_lambda[k] * (time_diff^2) / length_scales_lambda[k]^
                        2)
    
    for (i in 1:N) {
      mean_lambda <- G[i, ] %*% Gamma_init[k, ]
      
      lambda_init[i, k, ] <- mvrnorm(
        1,
        mu = rep(mean_lambda, T), Sigma = cov_matrix
      )
    }
  }
  
  mudraw <- apply(y, c(2,3), mean)
  logmudraw <- logit(pmax(mudraw, 1e-10))  # Ensure no negative values before logit
  smoothlogmudraw <- t(apply(logmudraw, 1, function(x) predict(loess(x ~ seq_len(T)))))
  
  # Initialize Phi based on smoothed mu_d
  Phi_init <- array(0, dim = c(n_topics, n_diseases, T))
  for (k in 1:n_topics) {
    for (d in 1:n_diseases) {
      t <- seq_len(T)
      Sigma <- var_scales_phi[k] * exp(-0.5 * outer(t, t, "-")^2 / length_scales_phi[k]^2)
      Phi_init[k, d, ] <- smoothlogmudraw[d, ] + mvrnorm(1, mu = rep(0, T), Sigma = Sigma)
    }
  }
  
  mu_d_init <- smoothlogmudraw
  
  
  
  
  return(
    list(
      Lambda = lambda_init,
      Phi = Phi_init,
      Gamma = Gamma_init,
      mu_d = mu_d_init,
      length_scales_lambda = length_scales_lambda,
      var_scales_lambda = var_scales_lambda,
      length_scales_phi = length_scales_phi,
      var_scales_phi = var_scales_phi
    )
  )
}
 #


### here we do the cool stuff with the SVD initialization, reprojection onto the individual geneti ccovariates

mcmc_init_two <- function(y, G,num_topics, length_scales_lambda, var_scales_lambda, length_scales_phi, var_scales_phi) {
  
  N <- dim(y)[1]  # Number of individuals
  D <- dim(y)[2]  # Number of diseases
  Ttot <- dim(y)[3]  # Number of time points
  P <- ncol(G)  # Number of genetic covariates
  K <- num_topics   # Number of topics
  
  # 1. Perform SVD on the time-averaged data
  Y_mean <- apply(y, c(1,2), mean)
  svd_result <- rsvd(Y_mean, k = K)
  A1 <- svd_result$u %*% diag(sqrt(svd_result$d[1:K]))
  A2 <- t(diag(sqrt(svd_result$d[1:K])) %*% t(svd_result$v))
  
  # 2. Create time basis (polynomial without intercept because we are using the genetics or disease prevalence)
  time_basis <- cbind(1, poly(seq_len(Ttot), degree = min(Ttot-1, 3), simple = TRUE))
  
  # 3. Initialize and project Lambda
  lambda_init <- array(0, dim = c(N, K, Ttot))
  
  Gamma_init <- matrix(0, nrow = K, ncol = P)
  for (k in 1:K) {
    Gamma_init[k, ] <- coef(lm(A1[, k] ~ G - 1)) ## because centered around genetics
    for (i in 1:N) {
      mean_lambda <- rep(G[i, ] %*% Gamma_init[k, ], T)
      lambda_init[i, k, ] <- mean_lambda + 
        mvrnorm(1, mu = rep(0, Ttot), Sigma = var_scales_lambda[k] * exp(-0.5 * outer(seq_len(Ttot), seq_len(Ttot), "-")^2 / length_scales_lambda[k]^2))
    }
  }
  
  # 4. Calculate mu_d and initialize Phi
  mudraw <- apply(y, c(2,3), mean)
  logmudraw <- qlogis(pmax(pmin(mudraw, 1-1e-10), 1e-10))  # Ensure values are within (0,1) before logit
  mu_d_init <- t(apply(logmudraw, 1, function(x) predict(loess(x ~ seq_len(Ttot)))))
  
  Phi_init <- array(0, dim = c(K, D, Ttot))
  for (k in 1:K) {
    for (d in 1:D) {
      Sigma <- var_scales_phi[k] * exp(-0.5 * outer(seq_len(Ttot), seq_len(Ttot), "-")^2 / length_scales_phi[k]^2)
      Phi_init[k, d, ] <- mu_d_init[d, ] + mvrnorm(1, mu = rep(0, Ttot), Sigma = Sigma)
    }
  }
  
  return(
    list(
      Lambda = lambda_init,
      Phi = Phi_init,
      Gamma = Gamma_init,
      mu_d = mu_d_init,
      length_scales_lambda = length_scales_lambda,
      var_scales_lambda = var_scales_lambda,
      length_scales_phi = length_scales_phi,
      var_scales_phi = var_scales_phi
    )
  )
  }
