library(ggplot2)
library(patchwork)
library(viridis)
library(dplyr)


# Install if needed
## restart R
library(reticulate)

library(pals)
## convert ro R
#use_condaenv("r-tensornoulli")
use_condaenv("/opt/miniconda3/envs/new_env_pyro2", required = TRUE)
torch <- import("torch")
tensor_to_r <- function(tensor) {
  as.array(tensor$detach()$cpu()$numpy())
}


gen = read.csv("~/aladynoulli2/pyScripts/big_stuff/all_patient_genetics.csv")
prs_names = read.csv("prs_names.csv")
all_patient_diseases = data.frame(read.csv("~/aladynoulli2/pyScripts/all_patient_diseases.csv", fill = T))
#Y = data.frame(read.csv("~/aladynoulli2/pyScripts/big_stuff/Y_summed_400k.csv"))
sig_refs = read.csv("~/aladynoulli2/pyScripts/reference_thetas.csv", header = T)

#all_thetas_tensor <- torch$load("big_stuff/all_patient_thetas_alltime.pt", weights_only =
                                  #FALSE)
## do not load data.table()
# Convert to an R array
all_thetas_array <- readRDS("all_thetas_array_time.rds")
#rm(all_thetas_tensor)
mean_thetas = apply(all_thetas_array, c(1, 2), mean) ## should give you time averaged mean
E=readRDS("E_full_tensor.rds")
library(reshape2)
library(microViz)
kelly<- distinct_palette(pal = "kelly")
kelly2=kelly
kelly2[6]=kelly[5]
kelly2[5]=kelly[6]

brewerPlus <- distinct_palette()
scales::show_col(brewerPlus)

c25 <- c(
  "dodgerblue2", "#E31A1C", # red
  "green4",
  "#6A3D9A", # purple
  "#FF7F00", # orange
  "black", "gold1",
  "skyblue2", "#FB9A99", # lt pink
  "palegreen2",
  "#CAB2D6", # lt purple
  "#FDBF6F", # lt orange
  "gray70", "khaki2",
  "maroon", "orchid1", "deeppink1", "blue1", "steelblue4",
  "darkturquoise", "green1", "yellow4", "yellow3",
  "darkorange4", "brown"
)

tab21_colors <- c(
  "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78",
  "#2ca02c", "#98df8a", "#d62728", "#ff9896",
  "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
  "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
  "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
  "#000000" # black for 21st signature
)
python_clusters=read.csv("python_clusters.csv")
traj_func = function(disease_ix) {
  name = all_patient_diseases[disease_ix, 1]
  print(all_patient_diseases[disease_ix, 1])
  diseased = which(Y[, disease_ix] == 1)
  event_times=data.frame(E[diseased,])+30
  
  time_averaged_theta = mean_thetas[diseased, ]
  time_theta = all_thetas_array[diseased, , ]
  goodgen = gen[diseased, ]
  K = dim(time_averaged_theta)[2]
  T = dim(time_theta)[3]
  set.seed(42)
  clust = kmeans(time_averaged_theta, centers = 3)
  event_times$cluster=clust$cluster
  event_time_vec <- event_times[, disease_ix]
  df_plot <- data.frame(
    event_time = event_time_vec,
    cluster = as.factor(clust$cluster)
  )
 pd= ggplot(df_plot, aes(x = event_time, y = cluster, fill = cluster)) +
    geom_density_ridges(alpha = 0.7, rel_min_height = 0.01) +
    theme_ridges() +
    labs(
      x = "Event Time (e.g., Age at Event)",
      y = "Cluster",
      title = "Distribution of Event Times by Cluster"
    ) +
    theme(legend.position = "none")
  print(pd)
  
  ggsave(plot = pd,filename = paste0("event_times",disease_ix,".pdf"),dpi = 300)
  
  
  print(table(clust$cluster))
  ### time center
  time_diff_by_cluster = array(NA, dim = c(3, K, T))
  time_means_by_cluster_array = array(NA, dim = c(3, K, T))
  for (t in 1:T) {
    time_spec_theta = data.frame(time_theta[, , t]) #Nd x K for a given T
    time_spec_theta$cluster = clust$cluster #which cluster each person in
    #time_spec_theta$cluster = python_clusters$cluster #which cluster each person in
    time_means_by_cluster <- aggregate(. ~ cluster, data = time_spec_theta, FUN = mean) #get the time averaged theta for that cluster across singatures
    time_means_by_cluster_array[, , t] = as.matrix(time_means_by_cluster[,-1])
    time_diff_by_cluster[, , t] <- sweep(as.matrix(time_means_by_cluster[, -1]), 2, sig_refs[, t], "-")
    
    
  }
  
  
  m = reshape2::melt(time_diff_by_cluster)
  names(m) = c("Cluster", "Sig", "Time", "Value")
  m$Sig = m$Sig - 1
  
  
  # m: data.frame with columns Cluster, Sig, Time, Value
  library(ggsci)
  # Option 1: Stacked area plot (all deviations)
  p1 = ggplot(m, aes(x = Time, y = Value, fill = as.factor(Sig))) +
    geom_area(position = "stack") + #scale_fill_manual(values = unname(glasbey())) +
    scale_fill_manual(values = brewerPlus)+
    facet_wrap( ~ Cluster, ncol = 1, scales = "free") +
    theme_classic() +
    labs(
      fill = "Signature",
      y = paste0("Theta Deviation from Reference ,", name),
      x = "Time"
    )
  
  
  library(ggplot2)
  library(viridis)
  
  # Option 1: Stacked area plot (all deviations)
  # p1 <- ggplot(m, aes(x = Time, y = Value, fill = as.factor(Sig))) +
  #   geom_area(position = "stack", alpha = 0.9) +
  #   scale_fill_viridis_d(option = "C", name = "Signature") +
  #   facet_wrap(~ Cluster, ncol = 1, scales = "free_y", 
  #              labeller = labeller(Cluster = function(x) paste0("Cluster ", x, " (n=", table(clust$cluster)[as.numeric(x)], ")"))) +
  #   theme_classic(base_size = 14) +
  #   labs(
  #     y = paste0("Theta Deviation from Reference (", name, ")"),
  #     x = "Time (e.g., Age)"
  #   ) +
  #   theme(
  #     strip.text = element_text(face = "bold", size = 13),
  #     plot.title = element_text(face = "bold", hjust = 0.5)
  #   )
  # 
  # print(p1)
  
  ggsave(plot = p1, filename = paste0("stackedtraj_", name, ".pdf"), dpi = 300, width = 8, height = 6)
  ggsave(
  plot = p1,
  filename = paste0("stackedtraj", name, ".pdf"),
   dpi = 300
  )
  
  # If goodgen is a data.table, convert to data.frame for easier manipulation
  goodgen_df <- as.data.frame(goodgen)[, -1]
  
  # Assume: goodgen is the PRS matrix for all diseased patients, clust$cluster is the cluster assignment
  
  library(dplyr)
  library(tidyr)
  
  
  # Add cluster assignment as a column
  goodgen_df$cluster <- clust$cluster
  #goodgen_df$cluster = python_clusters$cluster #
  # Compute mean PRS for each cluster
  prs_means_by_cluster <- aggregate(. ~ cluster, data = goodgen_df, FUN = mean)
  
  # (Optional) Move cluster column to rownames for a more matrix-like output
  rownames(prs_means_by_cluster) <- paste0("Cluster_", prs_means_by_cluster$cluster)
  prs_means_by_cluster$cluster <- NULL
  
  # Print or save
  print(prs_means_by_cluster)
  write.csv(prs_means_by_cluster,
            paste0("prs_means_by_cluster", name, ".csv"))
  
  p = data.frame(prs_means_by_cluster)
  prs_means_by_cluster$cluster = c(1:3)
  
  m = reshape2::melt(data.frame(prs_means_by_cluster), id.vars = "cluster")
  
  library(ggsci)
  
  absmin = min(m$value)
  absmax = max(m$value)
  p2 = ggplot(m, aes(x = cluster, y = variable, fill = value)) + geom_tile() +
    scale_fill_gradient2(
      low = "blue",
      high = "red",
      mid = "white",
      midpoint = 0,
      limits = c(absmin - 0.01, absmax + 0.01)
    ) + labs(x = "Grouping", y = "PRS", fill = "Mean Value")
  print(p2)
  
  library(ggplot2)
  library(reshape2)
  library(viridis)
  
  
  # Assume: prs_means_by_cluster (clusters x PRS), goodgen (PRS matrix), clust$cluster (assignments), prs_names (vector of PRS names)
  
  # Calculate in-cluster and out-of-cluster means for each PRS and cluster
  R <- ncol(prs_means_by_cluster) - 1
  n_clusters <- nrow(prs_means_by_cluster)
  prs_long <- data.frame()
  for (c in 1:n_clusters) {
    in_idx <- which(clust$cluster == c)
    out_idx <- which(clust$cluster != c)
    for (r in 1:R) {
      mean_in <- mean(goodgen_df[in_idx, r])
      mean_out <- mean(goodgen_df[out_idx, r])
      sd_in <- sd(goodgen_df[in_idx, r])
      sd_out <- sd(goodgen_df[out_idx, r])
      n_in <- length(in_idx)
      n_out <- length(out_idx)
      # Pooled standard deviation
      pooled_sd <- sqrt(((n_in - 1) * sd_in^2 + (n_out - 1) * sd_out^2) / (n_in + n_out - 2))
      d <- ifelse(pooled_sd > 0, (mean_in - mean_out) / pooled_sd, 0)
      # Calculate p-value for this cluster/signature
      pval <- tryCatch(
        t.test(theta_df[in_idx, k], theta_df[out_idx, k])$p.value,
        error = function(e) NA
      )
      prs_long <- rbind(prs_long,
                        data.frame(
                          Cluster = c,
                          PRS = prs_names[r, 1],
                          Mean = mean_in,
                          D = d
                        ))
    }
  }
  
  write.csv(prs_long, paste0("signature_cohens_d_", name, ".csv"), row.names = FALSE)
  
  # For heatmap: tile fill = Mean, annotation = D
  p <- ggplot(prs_long, aes(
    x = as.factor(Cluster),
    y = PRS,
    fill = Mean
  )) +
    geom_tile(color = "white") +
    scale_fill_gradient2(
      low = "blue",
      high = "red",
      mid = "white",
      midpoint = 0
    ) +
    geom_text(aes(label = sprintf("%.2f", Mean)),
              size = 3,
              color = "black") +
    labs(
      x = "Cluster",
      y = "PRS",
      fill = "Mean PRS",
      title = paste0("PRS Means by Cluster ", name)
    ) +
    theme_minimal(base_size = 14) +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5),
          axis.text.y = element_text(size = 12))
  
  print(p)
  
  ggsave(
    plot = p,
    filename = paste0("stackedgenetics", name, ".pdf"),
    dpi = 300
  )
  
  
  theta_df <- data.frame(time_averaged_theta)
  theta_df$cluster <- clust$cluster

K_sigs <- ncol(time_averaged_theta)
n_clusters <- 3
theta_long <- data.frame()

for (c in 1:n_clusters) {
  in_idx <- which(clust$cluster == c)
  out_idx <- which(clust$cluster != c)
  
  for (k in 1:K_sigs) {
    mean_in <- mean(theta_df[in_idx, k])
    mean_out <- mean(theta_df[out_idx, k])
    sd_in <- sd(theta_df[in_idx, k])
    sd_out <- sd(theta_df[out_idx, k])
    n_in <- length(in_idx)
    n_out <- length(out_idx)
    
    # Pooled standard deviation
    pooled_sd <- sqrt(((n_in - 1) * sd_in^2 + (n_out - 1) * sd_out^2) / (n_in + n_out - 2))
    d <- ifelse(pooled_sd > 0, (mean_in - mean_out) / pooled_sd, 0)
    
    # Calculate p-value for this cluster/signature
    pval <- tryCatch(
      t.test(theta_df[in_idx, k], theta_df[out_idx, k])$p.value,
      error = function(e) NA
    )
    
    theta_long <- rbind(theta_long,
                        data.frame(
                          Cluster = c,
                          Signature = k-1,  # 0-indexed signatures
                          Mean = mean_in,
                          D = d,
                          pval = pval
                        ))
  }
}

# Print and save signature Cohen's d results
print("Cohen's d for Disease Signatures:")
print(theta_long)
write.csv(theta_long, paste0("signature_cohens_d_", name, ".csv"), row.names = FALSE)

# Make text bigger in plot
p_sig <- ggplot(theta_long, aes(
  x = as.factor(Cluster),
  y = as.factor(Signature),
  fill = D
)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(
    low = "blue",
    high = "red",
    mid = "white",
    midpoint = 0
  ) +
  geom_text(aes(label = sprintf("%.2f", D)), size = 6, color = "black") +
  labs(
    x = "Cluster",
    y = "Signature",
    fill = "Mean",
    title = paste0("Signature Cohen's D by Cluster ", name)
  ) +
  theme_minimal(base_size = 18) +
  theme(axis.text.x = element_text(size = 16, angle = 0, hjust = 0.5),
        axis.text.y = element_text(size = 16))

print(p_sig)
ggsave(plot = p_sig, filename = paste0("signature_cohens_d_", name, ".pdf"), dpi = 300)
  
  return(list(
    clusters = clust,
    tat = time_averaged_theta,
    prs_cohens_d = prs_long,
    signature_cohens_d = theta_long
  ))
}




c=traj_func(18)
traj_func(113)
traj_func(67)

####
ukb_param=readRDS("~/aladynoulli2/pyScripts/big_stuff/ukb_params.rds")
E=sapply(seq(1:nrow(ukb_param$Y)),function(x)
  {
  ifelse(sum(ukb_param$Y[x,113,])==1,which(ukb_param$Y[x,113,]==1)+29,81)
})


# Suppose idxs is your vector of patient indices
idxs <- head(which(E != 81))


# Plot the lines and capture the colors used
matplot(t(all_thetas[idxs, 6, ]), type = "l", lty = 1, xlab = "Age", ylab = "Theta (K=5)", main = "Softmax Trajectories for K=5")
mat_colors <- 1:length(idxs)  # Default: matplot uses 1,2,3,... as colors, or you can specify col= explicitly

# Add vertical lines at event times, matching the color of each line
for (i in seq_along(idxs)) {
  abline(v = E[idxs[i]] - 29, col = mat_colors[i], lty = 2, lwd = 2)
}

legend("topright", legend = paste("Patient", idxs), col = mat_colors, lty = 1)


### case_thetas

case_thetas=colMeans(all_thetas[which(E!=81), 6, ])

control_thetas=colMeans(all_thetas[, 6, ])

r=rbind(case_thetas,control_thetas)
matplot(t(r), type = "l", lty = 1, xlab = "Age", ylab = "Theta (K=5)", main = "Softmax Trajectories for K=5",col = c("red","blue"))
mat_colors <- c("red","blue")  # Default: matplot uses 1,2,3,... as colors, or you can specify col= explicitly
legend("topright", legend = c("case","control"), col = mat_colors, lty = 1)


##

library(ggplot2)
library(tidyr)
library(dplyr)
library(patchwork)

ages <- 30:81
k_idx <- 6  # or your K of interest

# --- Panel 1: Individual patients ---
set.seed(7)
idxs <- sample(which(E != 81),size = 5)
df_indiv <- data.frame(
  Age = rep(ages, length(idxs)),
  Theta = as.vector(t(all_thetas[idxs, k_idx, ])),
  Patient = factor(rep(idxs, each = length(ages))),
  Event = rep(E[idxs], each = length(ages))
)

p1 <- ggplot(df_indiv, aes(x = Age, y = Theta, color = Patient, group = Patient)) +
  geom_line(size = 1.1) +
  geom_vline(aes(xintercept = Event, color = Patient), linetype = "solid", size = 1) +
  geom_vline(aes(xintercept = Event-2, color = Patient), linetype = "dashed", size = 1) +
  labs(title = "Softmax Trajectories (Individual Patients)", y = paste0("Theta (K=", k_idx, ")"), x = "Age") +
  theme_minimal(base_size = 15) +
  theme(legend.position = "bottom")

# --- Panel 2: Case vs Control means ---
case_thetas = colMeans(all_thetas[which(E != 81), k_idx, ])
control_thetas = colMeans(all_thetas[, k_idx, ])
df_mean <- data.frame(
  Age = rep(ages, 2),
  Theta = c(case_thetas, control_thetas),
  Group = rep(c("Case", "Control"), each = length(ages))
)

p2 <- ggplot(df_mean, aes(x = Age, y = Theta, color = Group)) +
  geom_line(size = 1.2) +
  labs(title = "Mean Softmax Trajectories", y = paste0("Theta (K=", k_idx, ")"), x = "Age") +
  scale_color_manual(values = c("Case" = "red", "Control" = "blue")) +
  theme_minimal(base_size = 15) +
  theme(legend.position = "bottom")

# --- Combine panels ---
g=p1 + p2 + plot_layout(ncol = 1)
ggsave(plot =g,filename = "sofmaxtraj.pdf",width=10,height = 10 )


#### early late

early_thetas = data.frame(colMeans(all_thetas[which(E <= 55), , ]))
late_thetas = data.frame(colMeans(all_thetas[which(E > 65&E!=81), , ]))
colnames(early_thetas)=colnames(late_thetas)=c(30:81)
early_thetas$group="young"
early_thetas$sig=c(0:20)
late_thetas$group="old"
late_thetas$sig=c(0:20)
library(microViz)

library(reshape2)
m=melt(rbind(early_thetas,late_thetas),id.vars=c("group","sig"))
ggplot(m,aes(x=variable,y=value,col=as.factor(sig),group=as.factor(sig)))+theme_classic()+
geom_smooth()+facet_wrap(~as.factor(group),ncol=1)+scale_color_manual(values = brewerPlus)



# 1. Compute velocity
early_theta_mat <- as.matrix(early_thetas[, as.character(30:81)])
early_velocity <- t(apply(early_theta_mat, 1, diff))
colnames(early_velocity) <- as.character(31:81)
early_velocity <- data.frame(early_velocity)
early_velocity$group <- "young"
early_velocity$sig <- 0:20

late_theta_mat <- as.matrix(late_thetas[, as.character(30:81)])
late_velocity <- t(apply(late_theta_mat, 1, diff))
colnames(late_velocity) <- as.character(31:81)
late_velocity <- data.frame(late_velocity)
late_velocity$group <- "old"
late_velocity$sig <- 0:20

# 2. Reshape
library(reshape2)
m_vel <- melt(rbind(early_velocity, late_velocity), id.vars = c("group", "sig"))

# 3. Plot
library(ggplot2)
ggplot(m_vel, aes(x = variable, y = value, col = as.factor(sig), group = as.factor(sig))) +
  geom_smooth(se = FALSE) +
  facet_wrap(~group) +
  labs(x = "Age", y = "Velocity (Δθ)", title = "Signature Velocity Over Time by Group")

mean_func = funclasction(disease_ix) {
  name = all_patient_diseases[disease_ix, 1]
  print(all_patient_diseases[disease_ix, 1])
  diseased = which(Y[, disease_ix] == 1)
  
  dat_of_interest = df[diseased, -1]
  goodgen = gen[diseased, ]
  
  set.seed(42)
  clust = kmeans(dat_of_interest, centers = 3)
  # If goodgen is a data.table, convert to data.frame for easier manipulation
  goodgen_df <- as.data.frame(goodgen)[, -1]
  
  # Add cluster assignment as a column
  goodgen_df$cluster <- clust$cluster
  
  # Compute mean PRS for each cluster
  prs_means_by_cluster <- aggregate(. ~ cluster, data = goodgen_df, FUN = mean)
  
  # (Optional) Move cluster column to rownames for a more matrix-like output
  rownames(prs_means_by_cluster) <- paste0("Cluster_", prs_means_by_cluster$cluster)
  prs_means_by_cluster$cluster <- NULL
  
  # Print or save
  print(prs_means_by_cluster)
  write.csv(prs_means_by_cluster,
            paste0("prs_means_by_cluster", name, ".csv"))
  
  p = data.frame(prs_means_by_cluster)
  prs_means_by_cluster$cluster = c(1:3)
  
  m = melt(data.frame(prs_means_by_cluster), id.vars = "cluster")
  
  library(ggsci)
  
  ggplot(m, aes(x = cluster, y = variable, fill = value)) + geom_tile() +
    scale_fill_gradient2(
      low = "blue",
      high = "red",
      mid = "white",
      midpoint = 0
    )
  return(clust)
}








# Function to create a side-by-side visualization of the same genetic factors across diseases
compare_genetic_factors_across_diseases <- function(csv_files,
                                                    output_pdf = NULL,
                                                    selected_factors = NULL,
                                                    num_top_factors = 5,
                                                    sort_by = "max_diff") {
  # Read all CSV files
  data_list <- list()
  disease_ids <- c()
  
  for (file in csv_files) {
    disease_id <- as.numeric(gsub(".*disease_([0-9]+)\\.csv$", "\\1", file))
    disease_ids <- c(disease_ids, disease_id)
    data <- read.csv(file)
    data$Disease_ID <- disease_id
    data$Disease_Name <- disease_names[as.character(disease_id)]
    data$Significant_Cluster0 = data$Significant_Cluster0 == "True"
    data$Significant_Cluster1 = data$Significant_Cluster1 == "True"
    data$Significant_Cluster2 = data$Significant_Cluster2 == "True"
    # Calculate metrics for sorting/highlighting differences
    data_list[[length(data_list) + 1]] <- data
  }
  
  # Combine data from all diseases
  all_data <- bind_rows(data_list)
  
  # If no specific factors are selected, find those with the biggest differences
  if (is.null(selected_factors)) {
    # Calculate differential metrics for each factor across all diseases
    factor_metrics <- all_data %>%
      group_by(Factor) %>%
      summarize(
        # Maximum absolute mean value across all diseases and clusters
        Max_Abs_Mean = max(abs(
          c(
            Mean_Value_Cluster0,
            Mean_Value_Cluster1,
            Mean_Value_Cluster2
          )
        )),
        
        # Maximum difference between any two clusters across all diseases
        Max_Difference = max(
          abs(Mean_Value_Cluster0 - Mean_Value_Cluster1),
          abs(Mean_Value_Cluster0 - Mean_Value_Cluster2),
          abs(Mean_Value_Cluster1 - Mean_Value_Cluster2)
        ),
        
        # Number of significant findings
        Num_Significant = sum(
          Significant_Cluster0,
          Significant_Cluster1,
          Significant_Cluster2
        ),
        
        # Present in all diseases?
        In_All_Diseases = n_distinct(Disease_ID) == length(disease_ids)
      ) %>%
      filter(In_All_Diseases) # Only include factors present in all diseases
    
    # Sort factors based on selected method
    if (sort_by == "max_diff") {
      top_factors <- factor_metrics %>%
        arrange(desc(Max_Difference)) %>%
        head(num_top_factors) %>%
        pull(Factor)
    } else if (sort_by == "significance") {
      top_factors <- factor_metrics %>%
        arrange(desc(Num_Significant)) %>%
        head(num_top_factors) %>%
        pull(Factor)
    } else if (sort_by == "max_value") {
      top_factors <- factor_metrics %>%
        arrange(desc(Max_Abs_Mean)) %>%
        head(num_top_factors) %>%
        pull(Factor)
    }
    
    selected_factors <- top_factors
  }
  
  # Filter to selected factors
  filtered_data <- all_data %>%
    filter(Factor %in% selected_factors)
  
  # Prepare data for plotting
  plot_data <- filtered_data %>%
    pivot_longer(
      cols = starts_with("Mean_Value_Cluster"),
      names_to = "Cluster",
      values_to = "Value"
    ) %>%
    mutate(
      Cluster_Num = as.numeric(gsub("Mean_Value_Cluster", "", Cluster)),
      Cluster_Label = paste0("Cluster ", Cluster_Num)
    )
  
  # Add significance data
  sig_data <- filtered_data %>%
    pivot_longer(
      cols = starts_with("Significant_Cluster"),
      names_to = "Sig_Cluster",
      values_to = "Is_Significant"
    ) %>%
    mutate(
      Cluster_Num = as.numeric(gsub("Significant_Cluster", "", Sig_Cluster)),
      Cluster_Label = paste0("Cluster ", Cluster_Num)
    )
  
  # Merge significance with plot data
  plot_data <- plot_data %>%
    left_join(
      sig_data %>% select(Factor, Disease_ID, Cluster_Label, Is_Significant),
      by = c("Factor", "Disease_ID", "Cluster_Label")
    )
  
  # Find global min/max for consistent color scale
  max_abs_val <- max(abs(plot_data$Value))
  
  # Generate plot
  p <- ggplot(plot_data, aes(x = Cluster_Label, y = Factor)) +
    facet_grid(. ~ Disease_Name) +
    geom_tile(aes(fill = Value)) +
    geom_text(aes(
      label = sprintf("%.2f%s", Value, ifelse(Is_Significant, "***", "")),
      color = abs(Value) > max_abs_val / 2
    ), size = 3) +
    scale_fill_gradient2(
      low = "blue",
      high = "red",
      mid = "white",
      midpoint = 0
    ) +
    scale_color_manual(values = c("black", "white"), guide = "none") +
    labs(title = "Comparison of Genetic Factors Across Diseases",
         subtitle = "*** p < 0.05 (Significant)",
         fill = "Mean Value") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold"),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5)
    )
  
  if (!is.null(output_pdf)) {
    ggsave(output_pdf, p, width = 20, height = 12)
    message(paste0("Saved comparison plot to ", output_pdf))
  }
  
  return(p)
}