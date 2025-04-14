library(tidyverse)
library(ggplot2)
library(patchwork)
library(viridis)
library(ggridges)
library(gridExtra)

# Load and process model data
load_model_data <- function(model_path, sig_refs_path) {
  model <- torch::load(model_path, weights_only = FALSE)
  sig_refs <- torch::load(sig_refs_path, weights_only = FALSE)
  
  # Convert tensors to R objects
  sigs <- as.array(sig_refs$signature_refs$detach()$cpu()$numpy())
  phi <- as.array(model$model_state_dict$phi$detach()$cpu()$numpy())
  lambda_ <- as.array(model$model_state_dict$lambda$detach()$cpu()$numpy())
  kappa <- as.array(model$model_state_dict$kappa$detach()$cpu()$numpy())
  
  return(list(sigs = sigs, phi = phi, lambda_ = lambda_, kappa = kappa))
}

# Helper function for softmax
softmax_by_k <- function(x) {
  # Apply softmax along K dimension (dimension 2)
  exp_x <- exp(x)
  sweep(exp_x, c(1,3), apply(exp_x, c(1,3), sum), "/")
}

# Calculate signature-specific probabilities
get_signature_probs <- function(phi, lambda_, kappa) {
  # Convert phi to probabilities using sigmoid
  phi_prob <- 1 / (1 + exp(-phi))
  
  # Calculate population-level theta using softmax
  theta <- softmax_by_k(lambda_)
  
  # Calculate average theta across individuals
  avg_theta <- apply(theta, c(2,3), mean)
  
  # Calculate signature-specific probabilities
  sig_probs <- array(0, dim = c(dim(phi)[1], dim(phi)[2], dim(phi)[3]))
  for(t in 1:dim(phi)[3]) {
    for(k in 1:dim(phi)[1]) {
      sig_probs[k,,t] <- phi_prob[k,,t] * avg_theta[k,t] * kappa
    }
  }
  
  return(sig_probs)
}

# Panel A: Show age effect and signature contributions
plot_age_signature_effects <- function(phi, lambda_, kappa, selected_diseases) {
  # Get signature-specific probabilities
  sig_probs <- get_signature_probs(phi, lambda_, kappa)
  
  # Calculate total probability and dominant signature
  total_probs <- apply(sig_probs, c(2,3), sum)
  dominant_sig <- apply(sig_probs, c(2,3), which.max)
  
  # Create data frame for plotting
  df <- data.frame(
    time = rep(1:dim(phi)[3], length(selected_diseases)),
    disease = rep(selected_diseases, each = dim(phi)[3]),
    total_prob = as.vector(total_probs[selected_diseases,]),
    dominant_sig = as.vector(dominant_sig[selected_diseases,])
  )
  
  p <- ggplot(df, aes(x = time, y = total_prob)) +
    geom_line(aes(color = factor(dominant_sig)), size = 1) +
    geom_ribbon(aes(ymin = 0, ymax = total_prob, fill = factor(dominant_sig)), alpha = 0.2) +
    scale_color_viridis_d() +
    scale_fill_viridis_d() +
    facet_wrap(~disease, scales = "free_y", nrow = 1) +
    labs(title = "A: Population-Level Disease Probabilities and Dominant Signatures",
         x = "Age",
         y = "Disease Probability",
         color = "Dominant Signature",
         fill = "Dominant Signature") +
    theme_minimal() +
    theme(legend.position = "right")
  
  return(p)
}

# Panel B: Show signature contributions
plot_signature_contributions <- function(phi, lambda_, kappa, selected_diseases) {
  # Get signature-specific probabilities
  sig_probs <- get_signature_probs(phi, lambda_, kappa)
  
  # Create data frame for plotting
  df <- data.frame()
  for(d in selected_diseases) {
    for(k in 1:dim(phi)[1]) {
      df <- rbind(df, data.frame(
        time = 1:dim(phi)[3],
        disease = d,
        signature = k,
        probability = sig_probs[k,d,]
      ))
    }
  }
  
  p <- ggplot(df, aes(x = time, y = probability, fill = factor(signature))) +
    geom_area(position = "stack") +
    scale_fill_viridis_d() +
    facet_wrap(~disease, scales = "free_y", nrow = 1) +
    labs(title = "B: Population-Level Signature Contributions",
         x = "Age",
         y = "Probability",
         fill = "Signature") +
    theme_minimal() +
    theme(legend.position = "right")
  
  return(p)
}

# Panel C: Show signature evolution
plot_signature_evolution <- function(phi, lambda_) {
  # Calculate signature strengths over time
  theta <- softmax_by_k(lambda_)
  
  # Calculate average signature strength
  avg_theta <- apply(theta, c(2,3), mean)
  
  # Create data frame for plotting
  df <- data.frame(
    signature = rep(1:dim(avg_theta)[1], dim(avg_theta)[2]),
    time = rep(1:dim(avg_theta)[2], each = dim(avg_theta)[1]),
    strength = as.vector(avg_theta)
  )
  
  p <- ggplot(df, aes(x = time, y = strength, fill = factor(signature))) +
    geom_area(position = "stack") +
    scale_fill_viridis_d() +
    labs(title = "C: Signature Evolution Over Time",
         x = "Age",
         y = "Signature Strength",
         fill = "Signature") +
    theme_minimal() +
    theme(legend.position = "right")
  
  return(p)
}

# Panel D: Show disease clusters
plot_disease_clusters <- function(phi, time_points = c(1, 10, 20)) {
  # Calculate disease correlations at different time points
  phi_prob <- 1 / (1 + exp(-phi))
  
  # Create data frame for plotting
  df <- data.frame()
  for(t in time_points) {
    cor_matrix <- cor(t(phi_prob[,,t]))
    for(i in 1:nrow(cor_matrix)) {
      for(j in 1:ncol(cor_matrix)) {
        if(i < j) {
          df <- rbind(df, data.frame(
            time = t,
            disease1 = i,
            disease2 = j,
            correlation = cor_matrix[i,j]
          ))
        }
      }
    }
  }
  
  p <- ggplot(df, aes(x = factor(disease1), y = factor(disease2), fill = correlation)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", 
                        midpoint = 0, limits = c(-1, 1)) +
    facet_wrap(~time, nrow = 1) +
    labs(title = "D: Disease Correlations Over Time",
         x = "Disease",
         y = "Disease",
         fill = "Correlation") +
    theme_minimal() +
    theme(axis.text.x = element_blank(),
          axis.text.y = element_blank(),
          legend.position = "right")
  
  return(p)
}

# Plot top diseases for each signature
plot_top_diseases <- function(phi, selected_sigs = c(1,2,3)) {
  phi_prob <- 1 / (1 + exp(-phi))
  plots <- list()
  
  for(sig in selected_sigs) {
    # Get top diseases for this signature
    avg_phi <- apply(phi_prob[sig,,], 1, mean)
    top_diseases <- order(avg_phi, decreasing = TRUE)[1:5]
    
    df <- data.frame(
      time = rep(1:dim(phi)[3], length(top_diseases)),
      disease = factor(rep(top_diseases, each = dim(phi)[3])),
      value = as.vector(t(phi_prob[sig,top_diseases,]))
    )
    
    p <- ggplot(df, aes(x = time, y = value, color = disease)) +
      geom_line() +
      theme_minimal() +
      labs(title = paste("Signature", sig),
           x = "Age",
           y = "Probability") +
      theme(legend.position = "right")
    
    plots[[length(plots) + 1]] <- p
  }
  
  return(plots)
}

# Plot cluster correspondence
plot_cluster_correspondence <- function(phi, time_points = c(10, 30, 50)) {
  phi_prob <- 1 / (1 + exp(-phi))
  plots <- list()
  
  for(t in time_points) {
    cor_mat <- cor(t(phi_prob[,,t]))
    df <- expand.grid(x = 1:nrow(cor_mat), y = 1:ncol(cor_mat))
    df$correlation <- as.vector(cor_mat)
    
    p <- ggplot(df, aes(x = x, y = y, fill = correlation)) +
      geom_tile() +
      scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                          midpoint = 0, limits = c(-1, 1)) +
      labs(title = paste("Age", t)) +
      theme_minimal() +
      theme(axis.text = element_blank())
    
    plots[[length(plots) + 1]] <- p
  }
  
  return(plots)
}

# Plot signature prevalence
plot_signature_prevalence <- function(lambda_) {
  theta <- softmax_by_k(lambda_)
  avg_theta <- apply(theta, c(2,3), mean)
  
  df <- data.frame(
    time = rep(1:dim(avg_theta)[2], dim(avg_theta)[1]),
    signature = factor(rep(1:dim(avg_theta)[1], each = dim(avg_theta)[2])),
    prevalence = as.vector(t(avg_theta))
  )
  
  p <- ggplot(df, aes(x = time, y = prevalence, color = signature)) +
    geom_line() +
    theme_minimal() +
    labs(title = "Signature Prevalence Across Age",
         x = "Age",
         y = "Prevalence") +
    theme(legend.position = "right")
  
  return(p)
}

# Main function
main <- function() {
  # Load data
  model_path <- "/Users/sarahurbut/Dropbox/resultshighamp/results/output_0_10000/model.pt"
  sig_refs_path <- "/Users/sarahurbut/Dropbox/data_for_running/reference_trajectories.pt"
  data <- load_model_data(model_path, sig_refs_path)
  
  # Create all plots
  top_disease_plots <- plot_top_diseases(data$phi)
  cluster_plots <- plot_cluster_correspondence(data$phi)
  prev_plot <- plot_signature_prevalence(data$lambda_)
  
  # Arrange plots using patchwork
  layout <- "
  AAABBB
  CCCDDD
  EEEFFF
  GGGGGG
  "
  
  final_plot <- (top_disease_plots[[1]] + top_disease_plots[[2]] +
                 top_disease_plots[[3]] + cluster_plots[[1]] +
                 cluster_plots[[2]] + cluster_plots[[3]] +
                 prev_plot) +
    plot_layout(design = layout) &
    theme(plot.margin = margin(5, 5, 5, 5))
  
  # Save plot
  ggsave("figure2.png", final_plot, width = 15, height = 12, dpi = 300)
}

# Run main function
main() 