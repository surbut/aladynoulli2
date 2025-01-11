# 1. First generate synthetic data
N <- 100      # Number of individuals
D <- 5        # Number of diseases
Ttot <- 50    # Number of time points
K <- 3        # Number of topics
P <- 5        # Number of genetic covariates

library(aladynoulli)
# Generate synthetic data
sim_data <- generate_tensor_data(
  N = N, 
  D = D, 
  T = Ttot, 
  K = K, 
  num_covariates = P
)

# Plot some example individuals to visualize the data
plot_individuals(sim_data$S, num_individuals = 3)

# 2. Run the Aladynoulli model
results <- aladynoulli_langevin(
  Y = sim_data$Y,
  G = sim_data$G,
  n_topics = K,
  n_iters = 1000,
  step_size_lambda = 0.01,
  step_size_phi = 0.01,
  length_scales_lambda = sim_data$length_scales_lambda,
  var_scales_lambda = sim_data$var_scales_lambda,
  length_scales_phi = sim_data$length_scales_phi,
  var_scales_phi = sim_data$var_scales_phi
)

# 3. Basic diagnostics
par(mfrow=c(3,1))
plot(results$log_likelihoods, type = "l", 
     main = "Log Likelihood Trace", 
     xlab = "Iteration", 
     ylab = "Log Likelihood")

plot(results$log_priors_lambda, type = "l", 
     main = "Log Prior Lambda Trace", 
     xlab = "Iteration", 
     ylab = "Log Prior Lambda")

plot(results$log_priors_phi, type = "l", 
     main = "Log Prior Phi Trace", 
     xlab = "Iteration", 
     ylab = "Log Prior Phi")

# 4. Compare estimated vs true parameters
burnin <- 500
Lambda_posterior_mean <- apply(results$samples$Lambda[burnin:1000,,,], c(2,3,4), mean)
Phi_posterior_mean <- apply(results$samples$Phi[burnin:1000,,,], c(2,3,4), mean)
Gamma_posterior_mean <- apply(results$samples$Gamma[burnin:1000,,], c(2,3), mean)

# Compare with true values
cat("Lambda MSE:", mean((Lambda_posterior_mean - sim_data$lambda_ik)^2), "\n")
cat("Phi MSE:", mean((Phi_posterior_mean - sim_data$phi_kd)^2), "\n")
cat("Gamma MSE:", mean((Gamma_posterior_mean - sim_data$Gamma_k)^2), "\n")