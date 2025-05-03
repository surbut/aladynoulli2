# Load necessary libraries
library(tidyverse)
library(pheatmap)
library(RColorBrewer)
library(viridis)

# Create a mapping of disease IDs to names
disease_names <- c(
  "17" = "Breast cancer [female]",
  "66" = "Major depressive disorder", 
  "112" = "Myocardial infarction"
)

# Function to create optimized heatmaps emphasizing between-cluster differences
plot_genetic_cluster_differences <- function(csv_file, 
                                             output_pdf = NULL,
                                             sort_method = "effect_diff", # Options: "effect_diff", "variance", "significance"
                                             filter_significant_only = TRUE,
                                             p_value_threshold = 0.05,
                                             show_top_n = 30) {
  
  # Extract disease ID from filename
  disease_id <- as.numeric(gsub(".*disease_([0-9]+)\\.csv$", "\\1", csv_file))
  disease_name <- disease_names[as.character(disease_id)]
  
  # Read the data
  data <- read.csv(csv_file)
  data$Significant_Cluster0=data$Significant_Cluster0=="True"
  data$Significant_Cluster1=data$Significant_Cluster1=="True"
  data$Significant_Cluster2=data$Significant_Cluster2=="True"
  # Calculate metrics for sorting/highlighting differences
  data <- data %>%
    mutate(
      # Calculate variance across clusters (higher = more differential)
      Variance_Between_Clusters = apply(
        data %>% select(Mean_Value_Cluster0, Mean_Value_Cluster1, Mean_Value_Cluster2),
        1, 
        var
      ),
      
      # Calculate max absolute difference between any two clusters
      Max_Effect_Diff = pmax(
        abs(Mean_Value_Cluster0 - Mean_Value_Cluster1),
        abs(Mean_Value_Cluster0 - Mean_Value_Cluster2),
        abs(Mean_Value_Cluster1 - Mean_Value_Cluster2)
      ),
      
      # Calculate significance score (minimum p-value)
      Min_P_Value = pmin(
        P_Value_Corrected_Cluster0,
        P_Value_Corrected_Cluster1, 
        P_Value_Corrected_Cluster2
      ),
   
      
      # Flag factors that are significant in any cluster
      Any_Significant = Significant_Cluster0 | Significant_Cluster1 | Significant_Cluster2
    )
  
  # Filter to only significant factors if requested
  if (filter_significant_only) {
    data <- data %>% filter(Any_Significant)
  }
  
  # Sort data based on selected method to emphasize differences
  if (sort_method == "effect_diff") {
    # Sort by maximum difference between any two clusters
    data <- data %>% arrange(desc(Max_Effect_Diff))
  } else if (sort_method == "variance") {
    # Sort by variance between clusters
    data <- data %>% arrange(desc(Variance_Between_Clusters))
  } else if (sort_method == "significance") {
    # Sort by statistical significance
    data <- data %>% arrange(Min_P_Value)
  }
  
  # Limit to top N factors with biggest differences
  if (nrow(data) > show_top_n) {
    data <- data %>% head(show_top_n)
  }
  
  # Extract cluster sizes
  cluster_sizes <- c(
    data$Cluster_Size_0[1],
    data$Cluster_Size_1[1],
    data$Cluster_Size_2[1]
  )
  
  # Create matrix for heatmap
  heatmap_data <- data %>%
    select(Factor, Mean_Value_Cluster0, Mean_Value_Cluster1, Mean_Value_Cluster2) %>%
    column_to_rownames("Factor") %>%
    as.matrix()
  
  # Rename columns for display
  colnames(heatmap_data) <- paste0("Cluster ", 0:2, "\n(n=", cluster_sizes, ")")
  
  # Create annotation for significance
  sig_data <- data %>%
    select(Factor, Significant_Cluster0, Significant_Cluster1, Significant_Cluster2) %>%
    column_to_rownames("Factor")
  
  # Find global min/max for consistent color scale
  max_abs_val <- max(abs(heatmap_data))
  breaks <- seq(-max_abs_val, max_abs_val, length.out = 101)
  
  # Create custom cell labels with significance markers
  custom_labels <- matrix("", nrow = nrow(heatmap_data), ncol = ncol(heatmap_data))
  for (i in 1:nrow(heatmap_data)) {
    for (j in 1:ncol(heatmap_data)) {
      # Format value with 2 decimal places
      val_str <- sprintf("%.2f", heatmap_data[i, j])
      
      # Add significance marker
      if (j == 1 && sig_data[i, "Significant_Cluster0"]) {
        val_str <- paste0(val_str, "***")
      } else if (j == 2 && sig_data[i, "Significant_Cluster1"]) {
        val_str <- paste0(val_str, "***")
      } else if (j == 3 && sig_data[i, "Significant_Cluster2"]) {
        val_str <- paste0(val_str, "***")
      }
      
      # Add effect size if available
      if (j == 1) {
        val_str <- paste0(val_str, "\nd=", sprintf("%.2f", data$Effect_Size_Cluster0[i]))
      } else if (j == 2) {
        val_str <- paste0(val_str, "\nd=", sprintf("%.2f", data$Effect_Size_Cluster1[i]))
      } else if (j == 3) {
        val_str <- paste0(val_str, "\nd=", sprintf("%.2f", data$Effect_Size_Cluster2[i]))
      }
      
      custom_labels[i, j] <- val_str
    }
  }
  
  # Create row annotation with metrics
  row_annotation <- data.frame(
    Variance = data$Variance_Between_Clusters,
    Max_Diff = data$Max_Effect_Diff,
    row.names = rownames(heatmap_data)
  )
  
  # Set up color scales for annotations
  annotation_colors <- list(
    Variance = colorRampPalette(brewer.pal(9, "YlOrRd"))(100),
    Max_Diff = colorRampPalette(brewer.pal(9, "YlOrRd"))(100)
  )
  
  # Create the heatmap
  p <- pheatmap(
    heatmap_data,
    color = colorRampPalette(rev(brewer.pal(11, "RdBu")))(100),
    breaks = breaks,
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    annotation_row = row_annotation,
    annotation_colors = annotation_colors,
    display_numbers = custom_labels,
    fontsize_number = 8,
    fontsize_row = 10,
    fontsize_col = 10,
    cellwidth = 80,
    cellheight = 15,
    main = paste0(
      "Top Genetic Factors by Cluster for ", disease_name,
      "\nSorted by ", ifelse(sort_method == "effect_diff", "Maximum Effect Difference", 
                             ifelse(sort_method == "variance", "Variance Between Clusters", "Statistical Significance")),
      "\n*** p < ", p_value_threshold, " (Significant)"
    ),
    filename = output_pdf,
    width = 12,
    height = min(25, 2 + 0.4 * nrow(heatmap_data))
  )
  
  if (is.null(output_pdf)) {
    return(p)
  } else {
    message(paste0("Saved heatmap to ", output_pdf))
  }
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
    data$Significant_Cluster0=data$Significant_Cluster0=="True"
    data$Significant_Cluster1=data$Significant_Cluster1=="True"
    data$Significant_Cluster2=data$Significant_Cluster2=="True"
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
        Max_Abs_Mean = max(abs(c(Mean_Value_Cluster0, Mean_Value_Cluster1, Mean_Value_Cluster2))),
        
        # Maximum difference between any two clusters across all diseases
        Max_Difference = max(
          abs(Mean_Value_Cluster0 - Mean_Value_Cluster1),
          abs(Mean_Value_Cluster0 - Mean_Value_Cluster2),
          abs(Mean_Value_Cluster1 - Mean_Value_Cluster2)
        ),
        
        # Number of significant findings
        Num_Significant = sum(
          Significant_Cluster0, Significant_Cluster1, Significant_Cluster2
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
    geom_text(
      aes(
        label = sprintf("%.2f%s", Value, ifelse(Is_Significant, "***", "")),
        color = abs(Value) > max_abs_val/2
      ),
      size = 3
    ) +
    scale_fill_gradient2(
      low = "blue", high = "red", mid = "white", 
      midpoint = 0, limit = c(-max_abs_val, max_abs_val)
    ) +
    scale_color_manual(values = c("black", "white"), guide = "none") +
    labs(
      title = "Comparison of Genetic Factors Across Diseases",
      subtitle = "*** p < 0.05 (Significant)",
      fill = "Mean Value"
    ) +
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

# Example usage
# Sorted by maximum effect difference (default)
plot_genetic_cluster_differences(
  "cluster_scores_disease_66.csv",
  "heatmap_disease66_by_diff.pdf",
  sort_method = "effect_diff",
  show_top_n = 20
)

# Sorted by variance between clusters
plot_genetic_cluster_differences(
  "cluster_scores_disease_66.csv",
  "heatmap_disease66_by_variance.pdf",
  sort_method = "variance",
  show_top_n = 20
)

# Sorted by significance
plot_genetic_cluster_differences(
  "cluster_scores_disease_66.csv",
  "heatmap_disease66_by_significance.pdf",
  sort_method = "significance",
  show_top_n = 20
)

# Do the same for the other diseases
plot_genetic_cluster_differences(
  "cluster_scores_disease_17.csv",
  "heatmap_disease17_by_diff.pdf"
)

plot_genetic_cluster_differences(
  "cluster_scores_disease_112.csv",
  "heatmap_disease112_by_diff.pdf"
)

# Compare the top 5 most differential genetic factors across all three diseases
compare_genetic_factors_across_diseases(
  c("cluster_scores_disease_17.csv", 
    "cluster_scores_disease_66.csv", 
    "cluster_scores_disease_112.csv"),
  "comparison_across_diseases.pdf",
  num_top_factors = 25,
  sort_by = "max_diff"
)

# You can also manually specify which factors to compare
compare_genetic_factors_across_diseases(
  c("cluster_scores_disease_17.csv", 
    "cluster_scores_disease_66.csv", 
    "cluster_scores_disease_112.csv"),
  "comparison_specific_factors.pdf",
  selected_factors = c("BMI", "CAD", "PC", "HT", "SCZ")
)



library(torch)

# Load the torch tensor
all_thetas_tensor <- torch$load("all_patient_thetas_alltime.pt",weights_only=FALSE)

# Convert to an R array
all_thetas_array <- tensor_to_r(all_thetas_tensor)

# Save as RDS
saveRDS(all_thetas_array, "all_patient_thetas_alltime.rds")