library(tidyverse)
library(viridis)
library(reshape2)
library(patchwork)
library(grid)
library(dendextend)

# Load the data


mgb_checkpoint <- readRDS("~/Dropbox/mgb_model.rds")
aou_checkpoint <- readRDS("~/Dropbox/aou_model.rds")
ukb_checkpoint <- readRDS("~/Dropbox/ukb_model.rds")


mgb_params=readRDS("~/Dropbox/mgb_params.rds")
aou_params=readRDS("~/Dropbox/aou_params.rds")
param=ukb_params=readRDS("~/Dropbox/ukb_params.rds")

# Helper functions
sigmoid <- function(x) {
  1/(1 + exp(-x))
}

softmax_by_k <- function(x, dim = 2) {
  # Apply softmax along specified dimension
  if(is.matrix(x)) {
    exp_x <- exp(x)
    return(t(t(exp_x) / colSums(exp_x)))
  } else if(is.array(x) && length(dim(x)) == 3) {
    exp_x <- exp(x)
    return(sweep(exp_x, c(1,3), apply(exp_x, c(1,3), sum), "/"))
  }
}

# Calculate pi predictions
calculate_pi_pred <- function(lambda_params, phi, kappa) {
  # Get dimensions
  dims <- dim(lambda_params)
  N <- dims[1]
  K <- dims[2]
  T <- dims[3]
  D <- dim(phi)[2]
  
  # Calculate theta using softmax
  theta <- softmax_by_k(lambda_params)
  
  # Calculate phi probability
  phi_prob <- sigmoid(phi)
  
  # Calculate pi_pred using matrix multiplication for each time point
  pi_pred <- array(0, dim = c(N, D, T))
  for(t in 1:T) {
    pi_pred[,,t] <- theta[,,t] %*% phi_prob[,,t] * as.numeric(model_params$kappa)
  }
  
  return(pi_pred)
}

# Base theme for all plots
base_theme <- theme_minimal() +
  theme(
    text = element_text(size = 8),
    axis.text = element_text(size = 7),
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 9, hjust = 0.5),
    legend.text = element_text(size = 7),
    legend.title = element_text(size = 8),
    plot.margin = margin(2, 2, 2, 2)
  )



get_top_diseases_for_signature <- function(signature_idx, n_top = 7,param=param) {
  # Get psi values for this signature
  psi_vals <- param$psi[signature_idx, ]
  # Return top n diseases by psi value
  return(disease_names[order(psi_vals, decreasing = TRUE)[1:n_top]])
}

# Panel A: Show two representative signatures with their characteristic diseases
plot_signature <- function(signature_idx, title = NULL, param=param, disease_names=disease_names, log=TRUE) {
  # Get top diseases for this signature based on psi
  top_diseases <- get_top_diseases_for_signature(signature_idx=signature_idx, param=param)
  
  if(log==FALSE) {
    # Create data frame for this signature's phi values
    signature_data <- data.frame(
      age = 1:52,  # assuming 52 age points
      sigmoid(t(param$phi[signature_idx, , ]))#-param$logit_prev))  # transpose to get diseases as columns
    )
    colnames(signature_data)[-1] <- disease_names
    
    # Convert to long format
    plot_data <- signature_data %>%
      gather(disease, phi_value, -age) %>%
      mutate(
        highlighted = disease %in% top_diseases,
        disease_factor = if_else(highlighted, disease, "Other")
      )
    
    # Create plot
    ggplot(plot_data, aes(x = age, y = phi_value, group = disease)) +
      geom_line(data = filter(plot_data, !highlighted),
                alpha = 0.1, color = "grey70") +
      geom_line(data = filter(plot_data, highlighted),
                aes(color = disease), size = 1) +
      scale_color_brewer(palette = "Set2") +
      theme_minimal() +
      labs(
        x = "Age",
        y = paste0("Hazard Ratio for Sig ", signature_idx-1),
        title = "",
        color = "Disease"
      ) +
      theme(
        text = element_text(size = 12),
        legend.position = "right",
        panel.grid.minor = element_blank()
      )
  } else {
    # Create data frame for this signature's phi values
    signature_data <- data.frame(
      age = 1:52,  # assuming 52 age points
      t(param$phi[signature_idx, , ])#-param$logit_prev)  # transpose to get diseases as columns
    )
    colnames(signature_data)[-1] <- disease_names
    
    # Convert to long format
    plot_data <- signature_data %>%
      gather(disease, phi_value, -age) %>%
      mutate(
        highlighted = disease %in% top_diseases,
        disease_factor = if_else(highlighted, disease, "Other")
      )
    
    # Create plot
    ggplot(plot_data, aes(x = age, y = phi_value, group = disease)) +
      geom_line(data = filter(plot_data, !highlighted),
                alpha = 0.1, color = "grey70") +
      geom_line(data = filter(plot_data, highlighted),
                aes(color = disease), size = 1) +
      scale_color_brewer(palette = "Set2") +
      theme_minimal() +
      labs(
        x = "Age",
        y = paste0("Log Hazard Ratio for Sig ", signature_idx-1),
        title = "",
        color = "Disease"
      ) +
      theme(
        text = element_text(size = 12),
        legend.position = "right",
        panel.grid.minor = element_blank()
      )
  }
}




#### Plot sig 6

plot_signature(signature_idx = 6,"",ukb_params,disease_names,log = FALSE)

plot_signature(signature_idx = 6,"",ukb_params,disease_names,log = TRUE)





















# Plot disease probabilities heatmap with pointers
plot_disease_probabilities <- function(pi_pred, selected_diseases, disease_names) {
  # Create data frame for heatmap
  n_diseases <- length(disease_names)
  n_time <- dim(pi_pred)[3]
  
  # Create time points vector
  time_points <- 1:n_time
  
  # Create base data frame
  plot_data <- data.frame(
    disease = rep(disease_names, each = n_time),
    time = rep(time_points, times = n_diseases),
    value = as.vector(pi_pred[1,,])
  ) %>%
  mutate(
    time = time + 29,
    highlighted = disease %in% selected_diseases
  )
  
  # Create main heatmap
  p <- ggplot(plot_data, aes(x = time, y = disease, fill = value)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "red", name = "Probability") +
    base_theme +
    theme(
      axis.text.y = element_text(size = 6),
      legend.position = "right"
    ) +
    labs(x = "Age", y = "") +
    # Add pointers for selected diseases
    geom_point(data = filter(plot_data, highlighted), 
              aes(x = max(time) + 1), 
              shape = ">", size = 2)
  
  # Add selected disease names on the right
  selected_y_pos <- match(selected_diseases, disease_names)
  for(i in seq_along(selected_diseases)) {
    if(!is.na(selected_y_pos[i])) {
      p <- p + annotate("text", 
                       x = max(plot_data$time) + 3,
                       y = selected_y_pos[i],
                       label = selected_diseases[i],
                       hjust = 0,
                       size = 2.5)
    }
  }
  
  return(p)
}

###

# Selected diseases for pointers
selected_diseases <- c(
  "Type 2 diabetes",
  "Myocardial infarction",
  "Alzheimer's disease",
  "Breast cancer",
  "Rheumatoid arthritis"
)

plot_disease_probabilities(pi_pred,selected_diseases,ukb_checkpoint$disease_names[,1])


######

# Create all plots
disease_names <- ukb_checkpoint$disease_names[,1]

# Calculate pi predictions
pi_pred <- calculate_pi_pred(param$lambda, param$phi, param$kappa)



# Plot disease probabilities heatmap with pointers
plot_disease_probabilities <- function(
    pi_pred,           # Array [N, D, T] or [D, T]
    disease_names,     # List of disease names
    selected_diseases, # List of disease names to highlight
    age_offset = 30,   # Value to add to time index for Age axis
    figsize = c(12, 16)){ # Figure dimensions

  pi_pred <- apply(pi_pred, c(2,3), mean) 
  rownames(pi_pred) <- disease_names  # Add disease names as row names
  
  # Melt the matrix with proper names
  plot_data <- melt(pi_pred) %>%
    rename(Disease = Var1, Time = Var2, value = value) %>%
    mutate(
      Time = Time + age_offset - 1,
      highlighted = Disease %in% selected_diseases
    )
  
  p <- ggplot(plot_data, aes(x = Time, y = Disease)) +
    geom_tile(aes(fill = value)) +
    scale_x_continuous(
      breaks = seq(min(plot_data$Time), max(plot_data$Time), by = 5)
    ) +
    scale_fill_gradient(
      low = "white",
      high = "red",
      name = "Average Probability",
      limits = c(0, 0.003),  # Set limits based on the distribution
      breaks = seq(0, 0.003, by = 0.0005),
      labels = function(x) sprintf("%.4f", x)
    )+
    theme_minimal() +
    theme(
      text = element_text(size = 8),
      axis.text.y = element_text(size = 6),
      axis.text.x = element_text(size = 7),
      panel.grid.minor = element_blank(),
      plot.title = element_text(size = 9, hjust = 0.5),
      legend.text = element_text(size = 7),
      legend.title = element_text(size = 8),
      plot.margin = margin(2, 2, 2, 2),
      legend.position = "right"
    ) +
    labs(
      x = "Age",
      y = ""
    )
  return(p)
}

plot_disease_probabilities(
  pi_pred,           # Array [N, D, T] or [D, T]
  disease_names,     # List of disease names
  selected_diseases, # List of disease names to highlight
  age_offset = 29)

#####


create_base_heatmap <- function(plot_data) {
  ggplot(plot_data, aes(x = signature-1, y = disease, fill = value)) +
    geom_tile() +
    scale_fill_gradient2(
      low = "navy",
      mid = "white",
      high = "red",
      midpoint = 0,
      limits = c(-5, 3),
      name = "Log Odds Ratio (psi)"
    ) +
    theme_minimal() +
    theme(
      axis.text.y = element_text(size = 6, hjust = 1),
      axis.text.x = element_text(size = 8),
      panel.grid = element_blank(),
      axis.title = element_text(size = 10),
      plot.title = element_text(size = 12, hjust = 0.5),
      legend.position = "right"
    ) +
    labs(
      x = "Signature",
      y = ""
    )
}

####


plot_comparison_heatmaps <- function(param,checkpoint) {
psi_matrix <- param$psi
initial_clusters <- checkpoint$clusters
# 1. Initial cluster ordering
# Order by initial cluster and then by max psi within cluster
max_psi <- apply(psi_matrix, 2, max)
initial_order <- order(initial_clusters, -max_psi)
# 2. Post-hoc clustering ordering
# Perform hierarchical clustering on diseases
row_dist <- dist(t(psi_matrix))
row_hclust <- hclust(row_dist, method = "complete")
cluster_order <- row_hclust$order
# Create plot data for both orderings
plot_data_initial <- as.data.frame(t(psi_matrix[, initial_order])) %>%
mutate(disease = disease_names[initial_order]) %>%
gather(signature, value, -disease) %>%
mutate(
signature = as.numeric(str_remove(signature, "V")),
disease = factor(disease, levels = disease_names[initial_order])
)
plot_data_clustered <- as.data.frame(t(psi_matrix[, cluster_order])) %>%
mutate(disease = disease_names[cluster_order]) %>%
gather(signature, value, -disease) %>%
mutate(
signature = as.numeric(str_remove(signature, "V")),
disease = factor(disease, levels = disease_names[cluster_order])
)
# Create the two heatmaps
p1 <- create_base_heatmap(plot_data_initial) +
labs(title = "Ordered by Initial Clusters") +
# Add lines to separate initial clusters
geom_hline(
yintercept = which(diff(initial_clusters[initial_order]) != 0) + 0.5,
color = "white",
size = 0.5
)
p2 <- create_base_heatmap(plot_data_clustered) +
labs(title = "Post-hoc Clustering")
# Combine plots
combined <- p1 + p2 +
plot_layout(guides = "collect") &
theme(legend.position = "right")
return(combined)
}


#### 

plot_comparison_heatmaps(ukb_params,checkpoint = ukb_checkpoint)


#######


# Plot biobank comparison
plot_biobank_patterns <- function(mgb_checkpoint, aou_checkpoint, ukb_checkpoint) {
  # Define signature mappings
  cv_signatures <- list(mgb = 5, aou = 16, ukb = 5)
  malig_signatures <- list(mgb = 11, aou = 11, ukb = 6)
  
  # Get clusters and disease names
  mgb_clusters <- mgb_checkpoint$clusters
  aou_clusters <- aou_checkpoint$clusters
  ukb_clusters <- ukb_checkpoint$clusters
  
  # Function to get diseases for a signature
  get_signature_diseases <- function(diseases, clusters, sig_num) {
    indices <- which(clusters == sig_num)
    setNames(indices, diseases[indices])
  }
  
  # Get diseases for each signature and biobank
  mgb_cv <- get_signature_diseases(mgb_checkpoint$disease_names, mgb_clusters, cv_signatures$mgb)
  aou_cv <- get_signature_diseases(aou_checkpoint$disease_names, aou_clusters, cv_signatures$aou)
  ukb_cv <- get_signature_diseases(ukb_checkpoint$disease_names[,1], ukb_clusters, cv_signatures$ukb)
  
  mgb_malig <- get_signature_diseases(mgb_checkpoint$disease_names, mgb_clusters, malig_signatures$mgb)
  aou_malig <- get_signature_diseases(aou_checkpoint$disease_names, aou_clusters, malig_signatures$aou)
  ukb_malig <- get_signature_diseases(ukb_checkpoint$disease_names[,1], ukb_clusters, malig_signatures$ukb)
  
  # Find shared diseases
  cv_shared <- Reduce(intersect, list(names(mgb_cv), names(aou_cv), names(ukb_cv)))
  malig_shared <- Reduce(intersect, list(names(mgb_malig), names(aou_malig), names(ukb_malig)))
  
  # Create plot data
  plot_single_pattern <- function(checkpoint,params, diseases, sig_idx, biobank_name) {
    phi <- params$phi
    patterns <- map_dfr(diseases, function(d) {
      disease_idx <- which(checkpoint$disease_names == d)
      data.frame(
        age = 1:51,
        value = phi[sig_idx+1, disease_idx, 1:51],
        disease = d,
        biobank = biobank_name
      )
    })
    return(patterns)
  }
  
  # Combine data for CV signatures
  cv_data <- bind_rows(
    plot_single_pattern(mgb_checkpoint, mgb_params,cv_shared, cv_signatures$mgb, "MGB"),
    plot_single_pattern(aou_checkpoint,aou_params, cv_shared, cv_signatures$aou, "AoU"),
    plot_single_pattern(ukb_checkpoint,ukb_params, cv_shared, cv_signatures$ukb, "UKB")
  )
  
  # Combine data for malignancy signatures
  malig_data <- bind_rows(
    plot_single_pattern(mgb_checkpoint,mgb_params, malig_shared, malig_signatures$mgb, "MGB"),
    plot_single_pattern(aou_checkpoint,aou_params,  malig_shared, malig_signatures$aou, "AoU"),
    plot_single_pattern(ukb_checkpoint, ukb_params, malig_shared, malig_signatures$ukb, "UKB")
  )
  
  # Create plots
  plot_patterns <- function(data, title) {
    ggplot(data, aes(x = age + 29, y = value, color = disease)) +
      geom_line(aes(linetype = biobank), size = 0.5) +
      facet_wrap(~biobank, ncol = 3) +
      scale_color_brewer(palette = "Set2") +
      base_theme +
      theme(
        strip.text = element_text(size = 8),
        legend.position = "right"
      ) +
      labs(
        x = "Age",
        y = "Log Hazard Ratio",
        title = title,
        color = "Disease",
        linetype = "Biobank"
      )
  }
  
  p1 <- plot_patterns(cv_data, "Cardiovascular Signatures")
  p2 <- plot_patterns(malig_data, "Malignancy Signatures")
  
  # Combine plots
  combined <- p1 / p2 +
    plot_layout(guides = "collect") &
    theme(legend.position = "right")
  
  return(combined)
}


pbc=plot_biobank_patterns(mgb_checkpoint,aou_checkpoint,ukb_checkpoint)

###


# Create individual plots
p1 <- plot_signature(signature_idx = 6,"",ukb_parms,disease_names,log = FALSE)
p2 <- plot_signature(signature_idx = 6,"",ukb_parms,disease_names,log = TRUE)
p3 <- plot_disease_probabilities(pi_pred, selected_diseases, disease_names)
p4 <- plot_cluster_comparison(param)
p5 <- plot_biobank_patterns(mgb_checkpoint, aou_checkpoint, ukb_checkpoint)

# Save plots
ggsave("figure2_temporal.pdf", p1 + p2 + plot_layout(guides = "collect"), width = 8, height = 3)
ggsave("figure2_probabilities.pdf", p3, width = 8, height = 10)
ggsave("figure2_clusters.pdf", p4, width = 6, height = 6)
ggsave("figure2_biobank.pdf", p5, width = 12, height = 8)