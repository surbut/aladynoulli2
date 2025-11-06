library(tidyverse)
library(viridis)
library(reshape2)
library(patchwork)
library(grid)
library(dendextend)

# Load the data


mgb_checkpoint <- readRDS("~/Library/CloudStorage//Dropbox-Personal/mgb_model.rds")
aou_checkpoint <- readRDS("~/Library/CloudStorage//Dropbox-Personal/aou_model.rds")
ukb_checkpoint <- readRDS("~/Library/CloudStorage//Dropbox-Personal/ukb_model.rds")


mgb_params=readRDS("~/Library/CloudStorage//Dropbox-Personal/mgb_params.rds")
aou_params=readRDS("~/Library/CloudStorage/Dropbox-Personal/aou_params.rds")
param=ukb_params=readRDS("~/Library/CloudStorage/Dropbox-Personal/ukb_params.rds")

a=array(data = NA,dim = c(21,348,52));for(i in c(1:21))
{a[i,,]=ukb_params$phi[i,,]-ukb_params$logit_prev}
library(reshape2)

# Convert difference array directly to a 3-column long format
delta <- a[,,50] - a[,,1]  # 21 x 348 matrix

#delta <- ukb_params$phi[,,50] - ukb_params$phi[,,1]  # 21 x 348 matrix
m <- melt(delta)
colnames(m) <- c("Sig", "Disease", "OR")

# Optionally, convert to factors if you want to control ordering or labeling
m$Sig <- factor(m$Sig)
m$Disease <- factor(m$Disease)



p <- ggplot(m, aes(x=Sig, y=Disease, fill=OR)) +
  geom_tile() +
  scale_fill_gradient2(
    low="#000C80", mid="white", high="#E64B35",
    midpoint=0) +
  labs(y="Disease", x="Signature", 
       title=paste0("Difference in Disease-Signature Deviation from Mean between time 50 and 0"))


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
    pi_pred[,,t] <- theta[,,t] %*% phi_prob[,,t] * as.numeric(kappa)
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
library(ggsci)
disease_names=ukb_checkpoint$disease_names[,1]

p1=plot_signature(signature_idx = 6,"",ukb_params,disease_names,log = TRUE)+scale_color_nejm()
p2=plot_signature(signature_idx = 7,"",ukb_params,disease_names,log = TRUE)+scale_color_jama()
p3=plot_signature(signature_idx = 12,"",ukb_params,disease_names,log = TRUE)+scale_color_npg()
p4=plot_signature(signature_idx = 15,"",ukb_params,disease_names,log = TRUE)+scale_color_aaas()




combined <- (p1 + p2) / (p3 + p4)
plot_layout(guides = "collect") &
theme(legend.position = "bottom")

ggsave(combined,file="~/Dropbox/aladynoulli_illustrator/combined_sigs.pdf",dpi = 300,width = 20)


library(dplyr)
library(tidyr)
library(ggplot2)

devs=melt(a)

names(devs)=c("Sigs","Disease","Time","value")

# Reshape: pivot to wide format for time 1 and 50
devs_wide <- devs %>%
  filter(Time %in% c(1, 50)) %>%
  pivot_wider(names_from = Time, values_from = value, names_prefix = "Time_")

devs_wide$DiseaseName <- disease_names[devs_wide$Disease]

# Now plot

devs_wide=devs_wide[devs_wide$Sigs%in%c(1:20),]
devs_wide$Sigs=devs_wide$Sigs-1
ggplotly(
  ggplot(devs_wide, aes(x = (Time_1), y = (Time_50))) +
    geom_point(aes(color = DiseaseName), alpha = 0.7) +
    facet_wrap(~ Sigs, scales = "free") +
    labs(x = "OR at Time 1", y = "OR at Time 50",
         title = "Disease Deviations from Population: Time 1 vs Time 50 by Signature") +
    theme_minimal() +
    theme(legend.position = "none")  # legend removed, but DiseaseName still used for hover
)
















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
  "Rheumatoid arthritis",
  "Infertility, female",
  "Dysmenorrhea",
  "Parkinson's disease",   
  "Macular degeneration (senile) of retina NOS",
  "Major depressive disorder",
  "Type 2 diabetes"   
)


# Calculate pi predictions
param=ukb_params
pi_pred <- calculate_pi_pred(param$lambda, param$phi, param$kappa)


plot_disease_probabilities(pi_pred,selected_diseases,ukb_checkpoint$disease_names[,1])


######

# Create all plots
disease_names <- ukb_checkpoint$disease_names[,1]


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


a=array(data = NA,dim = c(21,348,52));for(i in c(1:21)){a[i,,]=ukb_params$phi[i,,]-ukb_params$logit_prev}
lapply(seq(1:21),function(x){image(as.matrix(t(a[x,,])),main=paste0("Sig",x-1))})

library(gridExtra) # or library(patchwork)

# Create a list of plots
# Create a list of plots
plot_list <- lapply(c(1,25,51), function(x) {
  m <- melt(a[,,x])
  colnames(m) <- c("Sig", "Disease", "OR")
  
  # For the first two plots, don't show legend
  if (x != 51) {
    p <- ggplot(m, aes(x=Sig, y=Disease, fill=OR)) +
      geom_tile() +
      scale_fill_gradient2(limits=c(-5,3), 
                           low="#000C80", mid="white", high="#E64B35",
                           midpoint=0) +
      labs(y="Disease", x="Signature", title=paste0("OR at Age ", 29+x)) +
      theme(legend.position = "none")
  } else {
    # For the last plot, show legend
    p <- ggplot(m, aes(x=Sig, y=Disease, fill=OR)) +
      geom_tile() +
      scale_fill_gradient2(limits=c(-5,3), 
                           low="#000C80", mid="white", high="#E64B35",
                           midpoint=0) +
      labs(y="Disease", x="Signature", title=paste0("OR at Age ", 29+x), 
           fill="Log Odds Ratio (psi)") +
      theme(legend.position = "right")
  }
  return(p)
})

# Arrange plots in a 1x3 grid
ggsave(plot = grid.arrange(grobs = plot_list, ncol = 3),
       filename = "~/Dropbox/aladynoulli_illustrator/combinedphi.pdf", 
       dpi=300, width = 15)



m <- melt(a[,,50]-a[,,1])
colnames(m) <- c("Sig", "Disease", "OR")

p <- ggplot(m, aes(x=Sig, y=Disease, fill=OR)) +
    geom_tile() +
    scale_fill_gradient2( 
                         low="#000C80", mid="white", high="#E64B35",
                         midpoint=0) +
    labs(y="Disease", x="Signature", title=paste0("Difference in Disease-Signature Deviation from Mean between time 50 and 0"), 
         fill="Log Odds Ratio (psi)") +
    theme(legend.position = "right")
  

####


MI=fread("~/aladynoulli2/pyScripts/cluster_scores_disease_112.csv")
MI$trait="MI"
MDD=fread("~/aladynoulli2/pyScripts/cluster_scores_disease_66.csv")
MDD$trait="MDD"
AF=fread("~/aladynoulli2/pyScripts/cluster_scores_disease_127.csv")
AF$trait="AF"
PD=fread("~/aladynoulli2/pyScripts/cluster_scores_disease_76.csv")
PD$trait="PD"
AMD=fread("~/aladynoulli2/pyScripts/cluster_scores_disease_85.csv")
AMD$trait="AMD"
BC=fread("~/aladynoulli2/pyScripts/cluster_scores_disease_17.csv")
BC$trait="BC"
CBV=fread("~/aladynoulli2/pyScripts/cluster_scores_disease_132.csv")
CBV$trait="CBV"

df=rbind(BC,rbind(AMD,rbind(PD,rbind(AF,rbind(MI,MDD)))))

effects=df[,c("Factor","Mean_Value_Cluster0","Mean_Value_Cluster1",
"Mean_Value_Cluster2","trait")]

m=melt(effects)
library(ggplot2)
library(ggsci)
ggsave(plot=ggplot(m,aes(x=as.factor(Factor),y=value,fill=as.factor(variable)))+
  geom_bar(stat = "identity",position = "dodge")+
scale_fill_nejm()+labs(x="PRS",y="Mean Effect",Fill="Cluster")+
  facet_wrap(~trait,ncol=1)+theme_classic(),file="clusterplot.pdf")


###

# Load required libraries
library(tidyverse)
library(data.table)
# Read the data (assuming it's in a file called "ldsc_data.tsv")
# If your data is already in an R environment, you can skip this step
# and just use the data frame directly
ldsc_data <- fread("~/Downloads/99-ldsc-for-plot (2).tsv")

# Filter for positive genetic correlations and non-NA values
#positive_correlations <- ldsc_data %>%
 # filter(rg > 0, !is.na(rg))

positive_correlations = ldsc_data 
# Create a factor for signature to maintain order
positive_correlations$sig=positive_correlations$p1
positive_correlations$sig <- factor(positive_correlations$sig)

# Create a new column for significance (typically p < 0.05 is considered significant)
positive_correlations <- positive_correlations %>%
  mutate(significant = ifelse(p < 0.05, "*", ""))

# Get unique traits for better labeling
unique_traits <- positive_correlations %>%
  select(p2,trait) %>%
  distinct()
positive_correlations=positive_correlations[!is.na(positive_correlations$rg),]
# Create the heatmap with light red to dark red color scale
pos_cor=ggplot(positive_correlations, aes(x = sig, y = trait, fill = rg)) +
  geom_tile(color = "white") +
  #scale_fill_gradient(low = "#FFCCCC", high = "#990000") +
  scale_fill_gradient2(low = "blue", high = "#990000",mid="white") +
  geom_text(aes(label = significant), color = "black", size = 5) +
  theme_minimal() +
  labs(
    title = "Positive Genetic Correlations Heatmap",
    x = "Signature",
    y = "Trait",
    fill = "Genetic Correlation (rg)"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(size = 8)
  )


ggsave(plot = pos_cor,file="~/Dropbox/aladynoulli_illustrator/component_pdfs/figure4/ldsc.pdf",dpi = 300)
# For a more detailed visualization, we could also create a heatmap 
# that shows both color intensity and the actual rg values
ggplot(positive_correlations, aes(x = signature, y = trait, fill = rg)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "#FFCCCC", high = "#990000") +
  geom_text(aes(label = ifelse(significant == "*", 
                               paste0(round(rg, 2), "*"), 
                               round(rg, 2))), 
            color = "black", size = 3) +
  theme_minimal() +
  labs(
    title = "Positive Genetic Correlations with Values",
    x = "Signature",
    y = "Trait",
    fill = "Genetic Correlation (rg)"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(size = 8)
  )



###

# Load necessary packages
library(data.table)
library(ggplot2)
library(dplyr)

# Read in your files
sig5 <- fread("~/Dropbox/result326/10_loci/SIG5_AUC_ukb_eur_regenie_af1.sig.lead.sumstats.txt")
setwd("~/Dropbox/tetgwas/result/10_loci/")
trait_files <- list.files(pattern="_ukb_eur_regenie_af1.sig.lead.sumstats.txt")
trait_files <- setdiff(trait_files, "SIG5_AUC_ukb_eur_regenie_af1.sig.lead.sumstats.txt")
trait_list <- lapply(trait_files, fread)


# Extract RSIDs
sig5$rsid <- as.character(sig5$rsid)
trait_rsids <- unique(unlist(lapply(trait_list, function(x) as.character(x$rsid))))

# Annotate Signature 5 SNPs
sig5 <- sig5 %>%
  mutate(
    Category = case_when(
      !(rsid %in% trait_rsids) ~ "Signature5 Specific",
      rsid %in% trait_rsids ~ "Shared with Trait"
    )
  )

# Get Trait-specific SNPs
trait_snps <- bind_rows(trait_list) %>%
  filter(!(rsid %in% sig5$rsid)) %>%
  mutate(Category = "Trait Specific") %>%
  select(CHR = `#CHR`, POS, rsid, LOG10P, Category)

# Prepare Signature5 SNPs
sig5_plot <- sig5 %>%
  select(CHR = `#CHR`, POS, rsid, LOG10P, Category)

# Combine all SNPs
all_snps <- bind_rows(sig5_plot, trait_snps)

# Make sure chromosome is numeric
all_snps$CHR <- as.numeric(all_snps$CHR)

# Compute cumulative positions
all_snps <- all_snps %>% arrange(CHR, POS)
chr_offsets <- all_snps %>%
  group_by(CHR) %>%
  summarize(chr_len = max(POS)) %>%
  mutate(cum_len = cumsum(lag(chr_len, default = 0)))

all_snps <- all_snps %>%
  left_join(chr_offsets, by = "CHR") %>%
  mutate(pos_cum = POS + cum_len)

# Set colors
colors <- c(
  "Signature5 Specific" = "red",
  "Trait Specific" = "orange",
  "Shared with Trait" = "darkgreen"
)

# Plot
ggplot(all_snps, aes(x = pos_cum, y = LOG10P, color = Category)) +
  geom_point(alpha = 0.8, size = 1.2) +
  scale_color_manual(values = colors) +
  scale_x_continuous(
    label = unique(all_snps$CHR),
    breaks = chr_offsets$cum_len + chr_offsets$chr_len / 2
  ) +
  labs(x = "Chromosome", y = expression(-log[10](p)), title = "Signature 5 vs Component Traits") +
  theme_minimal() +
  theme(
    legend.position = "right",
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  )


### 500k selectiveity


library(data.table)
library(dplyr)
library(ggplot2)

# Load SIG5
sig5 <- fread("~/Dropbox/result326/10_loci/SIG5_AUC_ukb_eur_regenie_af1.sig.lead.sumstats.txt")
setwd("~/Dropbox/tetgwas/result/10_loci/")
trait_files <- list.files(pattern = "_ukb_eur_regenie_af1.sig.lead.sumstats.txt")
trait_files <- setdiff(trait_files, "SIG5_AUC_ukb_eur_regenie_af1.sig.lead.sumstats.txt")
trait_list <- lapply(trait_files, fread)

# Combine all traits into a single data.frame
traits_combined <- bind_rows(trait_list)

# Make sure CHR and POS are numeric
sig5$`#CHR` <- as.numeric(sig5$`#CHR`)
sig5$POS <- as.numeric(sig5$POS)
traits_combined$`#CHR` <- as.numeric(traits_combined$`#CHR`)
traits_combined$POS <- as.numeric(traits_combined$POS)

# Find shared SNPs by ±500kb matching
sig5 <- sig5 %>%
  rowwise() %>%
  mutate(
    Shared_with_Trait = any(
      (traits_combined$`#CHR` == `#CHR`) &
        (abs(traits_combined$POS - POS) <= 500000)
    )
  )

# Annotate Signature 5 SNPs
sig5 <- sig5 %>%
  mutate(
    Category = case_when(
      Shared_with_Trait ~ "Shared with Trait",
      TRUE ~ "Signature5 Specific"
    )
  )

# Now annotate trait SNPs: find which ones are >500kb from all sig5 SNPs
traits_combined <- traits_combined %>%
  rowwise() %>%
  mutate(
    Shared_with_Sig5 = any(
      (sig5$`#CHR` == `#CHR`) &
        (abs(sig5$POS - POS) <= 500000)
    )
  )

# Keep trait-specific only
trait_snps <- traits_combined %>%
  filter(!Shared_with_Sig5) %>%
  mutate(Category = "Trait Specific") %>%
  select(`#CHR`, POS, rsid, LOG10P, Category)

# Prepare Signature5 SNPs
sig5_plot <- sig5 %>%
  select(`#CHR`, POS, rsid, LOG10P, Category)

# Combine all SNPs
all_snps <- bind_rows(sig5_plot, trait_snps)

# Make sure chromosome is numeric
all_snps$`#CHR` <- as.numeric(all_snps$`#CHR`)

# Compute cumulative positions
all_snps <- all_snps %>% arrange(`#CHR`, POS)
chr_offsets <- all_snps %>%
  group_by(`#CHR`) %>%
  summarize(chr_len = max(POS)) %>%
  mutate(cum_len = cumsum(lag(chr_len, default = 0)))

all_snps <- all_snps %>%
  left_join(chr_offsets, by = "#CHR") %>%
  mutate(pos_cum = POS + cum_len)

# Set colors
colors <- c(
  "Signature5 Specific" = "red",
  "Trait Specific" = "orange",
  "Shared with Trait" = "darkgreen"
)

# Plot
ggplot(all_snps, aes(x = pos_cum, y = LOG10P, color = Category)) +
  geom_point(alpha = 0.8, size = 1.2) +
  scale_color_manual(values = colors) +
  scale_x_continuous(
    label = unique(all_snps$`#CHR`),
    breaks = chr_offsets$cum_len + chr_offsets$chr_len / 2
  ) +
  labs(x = "Chromosome", y = expression(-log[10](p)), title = "Signature 5 vs Component Traits (±500kb Matching)") +
  theme_minimal() +
  theme(
    legend.position = "right",
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  )


