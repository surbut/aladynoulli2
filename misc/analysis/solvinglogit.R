
# Load required libraries
set.seed(123)

### make simwithlogit a function to simulate

library("aladynoulli")

library(rsvd)  # For fast randomized SVD
library(mgcv) 

data <- generate_tensor_data(num_covariates = 5,K = 3,T = 20,D = 5,N = 100)
#
Y <- data$Y
G <- data$G
plot_individuals(data$S,num_individuals = 3)
# Here you initialize the MCMC
initial_values <- mcmc_init_two(y = Y, G = G, num_topics = 3, length_scales_lambda = rep(10, 3),
                                var_scales_lambda = rep(1, 3),
                                length_scales_phi = rep(10, 3),
                                var_scales_phi = rep(1, 3))

#a=aladynoulli(Y, G, n_topics = 3,n_iters=5000,
              initial_values=initial_values,step_size_lambda=0.01, step_size_phi=0.01,
              target_accept_rate = 0.2)
#saveRDS(a,"~/Desktop/aladynoulli.rds")
a=readRDS("~/Desktop/aladynoulli.rds")
a$acceptance_rates
plot(a$log_posteriors)
plot(a$log_likelihoods)
dim(a$samples$Lambda)
dim(a$samples$Phi)
dim(a$samples$Gamma)

plot(a$samples$Lambda[,1,1,1])

plot(a$samples$Phi[,1,1,1])


library(ggplot2)
library(tidyr)
library(dplyr)
library(patchwork)
library(viridis)

plot_simulation_summary_gg <- function(sim_data) {
  # Extract data
  mu_d <- sim_data$mu_d
  phi_kd <- sim_data$phi_kd
  theta <- sim_data$theta
  eta <- sim_data$eta
  
  # Convert to long format for ggplot
  # Baseline trajectories
  mu_df <- as.data.frame(t(mu_d)) %>%
    mutate(time = 1:nrow(.)) %>%
    pivot_longer(-time, names_to = "disease", values_to = "logit_prob") %>%
    mutate(disease = factor(disease))
  

  
  # Average topic memberships
  avg_theta <- apply(theta, c(2,3), mean)
  theta_df <- as.data.frame(t(theta[1,,])) %>%
    mutate(time = 1:nrow(.)) %>%
    pivot_longer(-time, names_to = "topic", values_to = "probability") %>%
    mutate(topic = factor(topic))
  
  # Create plots
  p1 <- ggplot(mu_df, aes(x = time, y = logit_prob, color = disease)) +
    geom_line() +
    theme_minimal() +
    scale_color_viridis_d() +
    labs(title = "Baseline Disease Trajectories (mu_d)",
         x = "Time", y = "logit(probability)") +
    theme(legend.position = "right")
  
  avg_pi <- apply(sim_data$pi_temp, c(2,3), mean)
  pi_melt=melt(avg_pi)
  colnames(pi_melt) =c("disease","time","probability")
  pi_melt$disease=as.factor(pi_melt$disease)
  p2 <- ggplot(pi_melt, aes(x = time, y = probability, color = disease)) +
    geom_line() +
    theme_minimal() +
    scale_color_viridis_d() +
    labs(title = "Average disease probabilities (pi_idt)",
         x = "Time", y = "Probability") +
    theme(legend.position = "right")
  
  p3 <- ggplot(eta_df, aes(x = time, y = probability, color = disease)) +
    geom_line() +
    theme_minimal() +
    scale_color_viridis_d() +
    labs(title = "Topic 1 Disease Probabilities sigmoid(phi)",
         x = "Time", y = "Probability") +
    theme(legend.position = "right")
  
  p4 <- ggplot(theta_df, aes(x = time, fill = as.factor(topic), y = probability)) +
    geom_area() +
    theme_minimal() +
    scale_color_viridis_d() +
    labs(title = "Topic Memberships Sample Id (theta)",
         x = "Time", y = "Probability") +
    theme(legend.position = "right")
  
  # Combine plots using patchwork
  (p1 + p3) / (p4 + p2)
}

plot_disease_correlations_gg <- function(sim_data) {
  Y <- sim_data$Y
  N <- dim(Y)[1]
  D <- dim(Y)[2]
  
  # Calculate disease occurrence correlations
  disease_occurs <- matrix(0, N, D)
  for(i in 1:N) {
    for(d in 1:D) {
      disease_occurs[i,d] <- any(Y[i,d,])
    }
  }
  
  cor_matrix <- cor(disease_occurs)
  
  # Convert to long format
  cor_df <- as.data.frame(cor_matrix) %>%
    mutate(disease1 = factor(1:D)) %>%
    pivot_longer(-disease1, names_to = "disease2", values_to = "correlation") %>%
    mutate(disease2 = factor(disease2))
  
  ggplot(cor_df, aes(x = disease1, y = disease2, fill = correlation)) +
    geom_tile() +
    scale_fill_viridis() +
    theme_minimal() +
    labs(title = "Disease Occurrence Correlations") +
    geom_text(aes(label = sprintf("%.2f", correlation)), size = 3) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

plot_individual_trajectories_gg <- function(sim_data, n_individuals=3) {
  theta <- sim_data$theta
  Y <- sim_data$Y
  K <- dim(theta)[2]
  T <- dim(theta)[3]
  
  # Create plots for selected individuals
  plot_list <- list()
  for(i in 1:n_individuals) {
    # Topic memberships
    theta_df <- as.data.frame(t(theta[i,,])) %>%
      mutate(time = 1:T) %>%
      pivot_longer(-time, names_to = "topic", values_to = "probability") %>%
      mutate(topic = factor(topic))
    
    # Disease events
    events <- apply(Y[i,,], 1, cumsum)
    events_df <- as.data.frame(events) %>%
      mutate(time = 1:T) %>%
      pivot_longer(-time, names_to = "disease", values_to = "events") %>%
      mutate(disease = factor(disease))
    
    p1 <- ggplot(theta_df, aes(x = time, y = probability, color = topic)) +
      geom_line() +
      theme_minimal() +
      scale_color_viridis_d() +
      labs(title = paste("Individual", i, "Topic Memberships"),
           x = "Time", y = "Probability")
    
    p2 <- ggplot(events_df, aes(x = time, y = events, color = disease)) +
      geom_step() +
      theme_minimal() +
      scale_color_viridis_d() +
      labs(title = paste("Individual", i, "Disease Events"),
           x = "Time", y = "Cumulative Events")
    
    plot_list[[length(plot_list) + 1]] <- p1
    plot_list[[length(plot_list) + 1]] <- p2
  }
  
  # Arrange plots using patchwork
  wrap_plots(plot_list, ncol = 2)
}

# Usage:
sim_data <- generate_tensor_data()
plot_simulation_summary_gg(sim_data)
plot_disease_correlations_gg(sim_data)
plot_individual_trajectories_gg(sim_data)