# 
# samples=readRDS("../logit_factorization/mcmc_samples_complex_noLAMPHI.rds")
# plot(samples$samples$Lambda[, 1, 1, 1], type = 'l', main = "Trace plot for Lambda[1,1,1]")
# plot(samples$samples$Lambda[, 1, 3, 1], type = 'l', main = "Trace plot for Lambda[1,3,1]")
# plot(samples$samples$Phi[, 1, 1, 1], type = 'l', main = "Trace plot for Phi[1,1,1]")
# plot(samples$samples$Gamma[, 1, 1], type = 'l', main = "Trace plot for Gamma[1,1]")
# plot(samples$log_posteriors)
# plot(samples$log_likelihoods)
# 
# 
# 
# library(coda)
# library(bayesplot)
# library(ggplot2)
# 
# # Convert your samples to mcmc objects
# lambda_mcmc <- as.mcmc(samples$Lambda[, 1, 1, 1])  # Example for one element
# phi_mcmc <- as.mcmc(samples$Phi[, 1, 1, 1])
# gamma_mcmc <- as.mcmc(samples$Gamma[, 1, 1])
# 
# 
# # Autocorrelation plots
# acf(lambda_mcmc)
# acf(phi_mcmc)
# acf(gamma_mcmc)
# 
# # Effective Sample Size
# effectiveSize(lambda_mcmc)
# effectiveSize(phi_mcmc)
# effectiveSize(gamma_mcmc)
# 
# # Posterior summaries
# summary(lambda_mcmc)
# summary(phi_mcmc)
# summary(gamma_mcmc)
# 
# # Compare with true values (assuming you have access to these)
# true_lambda <- lambda_ik[1, 1, 1]  # Example for one element
# true_phi <- qlogis(phi_kd[1, 1, 1])
# true_gamma <- Gamma_k[1, 1]
# 
# cat("True Lambda:", true_lambda, "\n")
# cat("Estimated Lambda (mean):", mean(lambda_mcmc), "\n")
# cat("True Phi:", true_phi, "\n")
# cat("Estimated Phi (mean):", mean(phi_mcmc), "\n")
# cat("True Gamma:", true_gamma, "\n")
# cat("Estimated Gamma (mean):", mean(gamma_mcmc), "\n")
# 
# # Plot posterior distributions with true values
# ggplot(data.frame(lambda = as.vector(lambda_mcmc)), aes(x = lambda)) +
#   geom_density() +
#   geom_vline(xintercept = true_lambda, color = "red") +
#   ggtitle("Posterior distribution of Lambda[1,1,1]")
# ```
# 
# ## pi
# 
# lambda_mean=apply(samples$Lambda,c(2,3,4),mean)
# phi_mean=plogis(apply(samples$Phi,c(2,3,4),mean))
# theta=apply(lambda_mean,c(1,3),function(x){softmax(x)})
# pi_post=array(data = 0,dim=c(N,D,T))
# for(t in 1:T){
#   pi_post[, , t] <- t(theta[, ,t ]) %*% phi_mean[, , t]
# }
# 
# par(mfrow = c(2, 2))
# for (i in sample(1:N, 4)) {
#   matplot(
#     t(pi_post[i, , ]),
#     type = 'l',
#     main = paste("Pi for individual", i),
#     xlab = "Time",
#     ylab = "Pi"
#   )
# }
# 
# for (i in sample(1:N, 4)) {
#   matplot(
#     t(pi_values[i, , ]),
#     type = 'l',
#     main = paste("True Pi for individual", i),
#     xlab = "Time",
#     ylab = "Pi"
#   )
# }
# 
# 
# 
# negative_log_posterior <- function(params, y, g_i, K_inv_lambda, K_inv_phi, mu_d) {
#   N <- dim(y)[1]
#   D <- dim(y)[2]
#   T <- dim(y)[3]
#   K <- length(K_inv_lambda)
#   P <- ncol(g_i)
# 
#   Lambda <- array(params[1:(N*K*T)], dim=c(N,K,T))
#   Phi <- array(params[(N*K*T+1):(N*K*T+K*D*T)], dim=c(K,D,T))
#   Gamma <- matrix(params[(N*K*T+K*D*T+1):length(params)], nrow=K)
# 
#   # Likelihood
#   ll <- log_likelihood(y, Lambda, Phi)
# 
#   # Priors
#   lp_lambda <- sum(sapply(1:N, function(i) {
#     sapply(1:K, function(k) {
#       log_gp_prior_vec(Lambda[i,k,], rep(g_i[i,] %*% Gamma[k,], T), K_inv_lambda[[k]]$K_inv, K_inv_lambda[[k]]$log_det_K)
#     })
#   }))
# 
#   lp_phi <- sum(sapply(1:K, function(k) {
#     sapply(1:D, function(d) {
#       log_gp_prior_vec(Phi[k,d,], mu_d[d,], K_inv_phi[[k]]$K_inv, K_inv_phi[[k]]$log_det_K)
#     })
#   }))
# 
#   lp_gamma <- sum(dnorm(Gamma, 0, 1, log=TRUE))
# 
#   return(-(ll + lp_lambda + lp_phi + lp_gamma))
# }
# 
# 
# 
# compute_map_estimates <- function(y, g_i, initial_values, K_inv_lambda, K_inv_phi, mu_d) {
#   N <- dim(y)[1]
#   D <- dim(y)[2]
#   T <- dim(y)[3]
#   K <- dim(initial_values$Lambda)[2]
#   P <- ncol(g_i)
# 
#   # Flatten initial values into a single vector
#   initial_params <- c(
#     as.vector(initial_values$Lambda),
#     as.vector(initial_values$Phi),
#     as.vector(initial_values$Gamma)
#   )
# 
#   # Optimize
#   result <- optim(
#     par = initial_params,
#     fn = negative_log_posterior,
#     y = y,
#     g_i = g_i,
#     K_inv_lambda = K_inv_lambda,
#     K_inv_phi = K_inv_phi,
#     mu_d = mu_d,
#     method = "L-BFGS-B",
#     control = list(maxit = 100)
#   )
# 
#   # Reshape results
#   map_estimates <- list(
#     Lambda = array(result$par[1:(N*K*T)], dim=c(N,K,T)),
#     Phi = array(result$par[(N*K*T+1):(N*K*T+K*D*T)], dim=c(K,D,T)),
#     Gamma = matrix(result$par[(N*K*T+K*D*T+1):length(result$par)], nrow=K)
#   )
# 
#   return(map_estimates)
# }
# 
# # Compute K_inv for Lambda and Phi
# Lambda_init=initial_values$Lambda
# Phi_init=initial_values$Phi
# Gamma_init=initial_values$Gamma
# mu_d_init=initial_values$mu_d
# 
#   K_inv_lambda <- lapply(1:n_topics, function(k)
#     precompute_K_inv(T, length_scales_lambda[k], var_scales_lambda[k]))
#   K_inv_phi <- lapply(1:n_topics, function(k)
#     precompute_K_inv(T, length_scales_phi[k], var_scales_phi[k]))
# 
#   # Compute MAP estimates
#   map_estimates <- compute_map_estimates(y, g_i,
#                                          list(Lambda=Lambda_init, Phi=Phi_init, Gamma=Gamma_init),
#                                          K_inv_lambda, K_inv_phi, mu_d_init)
# 
