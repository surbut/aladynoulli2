#!/usr/bin/env Rscript
# Rare Variant Burden Analysis on RAP
# Correlates rare variant carriers with disease outcomes and signature loadings

# =============================================================================
# SETUP AND DATA LOADING
# =============================================================================

library(data.table)
library(dplyr)
library(ggplot2)
library(corrplot)

# Set working directory (adjust for RAP)
setwd("/mnt/project")

# =============================================================================
# 1. LOAD GENOTYPE DATA (Rare Variants)
# =============================================================================

# On RAP, genotype files are typically in:
# - /mnt/project/genotype/ or similar
# - May be in BGEN format or already extracted as PLINK/CSV
# - Check available files first

cat("Looking for genotype files...\n")
genotype_paths <- c(
  "/mnt/project/genotype/",
  "/mnt/project/exports/genotype/",
  "/mnt/project/data/genotype/"
)

genotype_file <- NULL
for (path in genotype_paths) {
  if (dir.exists(path)) {
    cat(sprintf("Found directory: %s\n", path))
    files <- list.files(path, pattern = ".*\\.(raw|txt|csv|bgen)$", full.names = TRUE)
    if (length(files) > 0) {
      cat(sprintf("  Found %d genotype files\n", length(files)))
      # Look for rare variant files specifically
      rare_variant_files <- files[grepl("rare|variant|carrier", files, ignore.case = TRUE)]
      if (length(rare_variant_files) > 0) {
        genotype_file <- rare_variant_files[1]
        cat(sprintf("  Using rare variant file: %s\n", genotype_file))
        break
      } else {
        genotype_file <- files[1]  # Use first file if no specific rare variant file
        cat(sprintf("  Using first available file: %s\n", genotype_file))
        break
      }
    }
  }
}

if (is.null(genotype_file)) {
  cat("⚠️  No genotype file found. Please specify the path manually.\n")
  cat("   Expected format: CSV/TSV with columns: IID (or eid), variant columns\n")
  cat("   Or PLINK .raw format with IID column\n")
  
  # Example: manually specify if known
  # genotype_file <- "/mnt/project/exports/rare_variants.raw"
}

# Load genotype data
if (!is.null(genotype_file) && file.exists(genotype_file)) {
  cat(sprintf("\nLoading genotype data from: %s\n", genotype_file))
  
  # Try different formats
  if (grepl("\\.raw$", genotype_file)) {
    # PLINK .raw format
    genotype_data <- fread(genotype_file, sep = " ")
    cat(sprintf("  Loaded %d samples, %d variants\n", nrow(genotype_data), ncol(genotype_data) - 6))
  } else {
    # CSV/TSV format
    genotype_data <- fread(genotype_file)
    cat(sprintf("  Loaded %d samples, %d columns\n", nrow(genotype_data), ncol(genotype_data)))
  }
  
  # Extract patient IDs (assuming first column or column named IID/eid)
  id_col <- if ("IID" %in% names(genotype_data)) "IID" else names(genotype_data)[1]
  patient_ids_geno <- as.character(genotype_data[[id_col]])
  
  # Extract variant columns (exclude ID columns)
  id_cols <- c("FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE")
  variant_cols <- setdiff(names(genotype_data), id_cols)
  
  cat(sprintf("  Found %d variant columns\n", length(variant_cols)))
  
} else {
  cat("⚠️  Genotype file not found. Creating placeholder structure.\n")
  cat("   You'll need to load your genotype data manually.\n")
  genotype_data <- NULL
  variant_cols <- NULL
  patient_ids_geno <- NULL
}

# =============================================================================
# 2. LOAD Y ARRAY (Disease Outcomes)
# =============================================================================

cat("\nLoading Y array (disease outcomes)...\n")

# Try multiple possible locations
y_paths <- c(
  "/mnt/project/exports/Y_tensor.pt",
  "/mnt/project/exports/Y_tensor.npz",
  "/mnt/project/exports/Y_tensor.rds",
  "/mnt/project/data/Y_tensor.rds"
)

Y <- NULL
Y_loaded <- FALSE

for (y_path in y_paths) {
  if (file.exists(y_path)) {
    cat(sprintf("Found Y array at: %s\n", y_path))
    
    if (grepl("\\.rds$", y_path)) {
      # R RDS format
      Y <- readRDS(y_path)
      Y_loaded <- TRUE
      cat(sprintf("  Loaded Y array: %s\n", paste(dim(Y), collapse = " x ")))
      break
    } else if (grepl("\\.npz$", y_path)) {
      # NumPy compressed format (need RcppCNPy or similar)
      cat("  NumPy format detected - may need Python/R interface\n")
      # Y <- RcppCNPy::npyLoad(y_path, "Y")
    } else if (grepl("\\.pt$", y_path)) {
      cat("  PyTorch format detected - need Python/R interface\n")
    }
  }
}

if (!Y_loaded) {
  cat("⚠️  Y array not found in standard locations.\n")
  cat("   Please load manually or specify path.\n")
  cat("   Expected: 3D array (N x D x T) where:\n")
  cat("     N = patients, D = diseases, T = timepoints\n")
}

# =============================================================================
# 3. CONDENSE Y ARRAY TO DISEASE CARRIERS (Binary)
# =============================================================================

if (!is.null(Y) && Y_loaded) {
  cat("\nCondensing Y array to binary disease carriers...\n")
  
  # Option 1: Any disease occurrence across all timepoints
  # Y_binary: (N x D) - 1 if patient ever had disease, 0 otherwise
  Y_binary <- apply(Y, c(1, 2), function(x) as.integer(any(x > 0)))
  
  # Option 2: Disease by age group (if you want temporal analysis)
  # Early onset: ages 30-50
  # Late onset: ages 51-80
  if (dim(Y)[3] >= 50) {
    early_onset <- apply(Y[, , 1:21], c(1, 2), function(x) as.integer(any(x > 0)))  # ages 30-50
    late_onset <- apply(Y[, , 22:dim(Y)[3]], c(1, 2), function(x) as.integer(any(x > 0)))  # ages 51-80
  }
  
  cat(sprintf("  Created binary disease matrix: %d patients x %d diseases\n", 
              nrow(Y_binary), ncol(Y_binary)))
  cat(sprintf("  Total disease occurrences: %d\n", sum(Y_binary)))
  
  # Get disease names (if available)
  disease_names <- if (!is.null(dimnames(Y)[[2]])) {
    dimnames(Y)[[2]]
  } else {
    paste0("Disease_", 1:ncol(Y_binary))
  }
  
  # Get patient IDs (if available)
  patient_ids_Y <- if (!is.null(dimnames(Y)[[1]])) {
    as.character(dimnames(Y)[[1]])
  } else {
    paste0("Patient_", 1:nrow(Y_binary))
  }
  
  colnames(Y_binary) <- disease_names
  rownames(Y_binary) <- patient_ids_Y
  
} else {
  cat("⚠️  Cannot condense Y array - not loaded.\n")
  Y_binary <- NULL
  patient_ids_Y <- NULL
  disease_names <- NULL
}

# =============================================================================
# 4. LOAD SIGNATURE LOADINGS
# =============================================================================

cat("\nLoading signature loadings...\n")

theta_paths <- c(
  "/mnt/project/exports/thetas.pt",
  "/mnt/project/exports/thetas.npy",
  "/mnt/project/exports/thetas.rds",
  "/mnt/project/data/thetas.rds"
)

thetas <- NULL
for (theta_path in theta_paths) {
  if (file.exists(theta_path)) {
    cat(sprintf("Found thetas at: %s\n", theta_path))
    if (grepl("\\.rds$", theta_path)) {
      thetas <- readRDS(theta_path)
      cat(sprintf("  Loaded thetas: %s\n", paste(dim(thetas), collapse = " x ")))
      break
    }
  }
}

if (is.null(thetas)) {
  cat("⚠️  Signature loadings not found. Will skip signature-disease plots.\n")
}

# =============================================================================
# 5. ALIGN PATIENT IDs ACROSS DATASETS
# =============================================================================

cat("\nAligning patient IDs across datasets...\n")

if (!is.null(patient_ids_geno) && !is.null(patient_ids_Y)) {
  # Find common patients
  common_patients <- intersect(patient_ids_geno, patient_ids_Y)
  cat(sprintf("  Found %d common patients\n", length(common_patients)))
  
  if (length(common_patients) == 0) {
    cat("⚠️  No common patients found! Check ID formats.\n")
    cat(sprintf("    Genotype IDs sample: %s\n", paste(head(patient_ids_geno), collapse = ", ")))
    cat(sprintf("    Y array IDs sample: %s\n", paste(head(patient_ids_Y), collapse = ", ")))
  } else {
    # Subset all datasets to common patients
    geno_idx <- match(common_patients, patient_ids_geno)
    Y_idx <- match(common_patients, patient_ids_Y)
    
    genotype_data_aligned <- genotype_data[geno_idx, ]
    Y_binary_aligned <- Y_binary[Y_idx, ]
    
    if (!is.null(thetas)) {
      theta_idx <- if (!is.null(rownames(thetas))) {
        match(common_patients, rownames(thetas))
      } else {
        Y_idx  # Assume same order as Y
      }
      thetas_aligned <- thetas[theta_idx, ]
    } else {
      thetas_aligned <- NULL
    }
    
    cat(sprintf("  Aligned datasets: %d patients\n", length(common_patients)))
  }
} else {
  cat("⚠️  Cannot align - missing patient IDs from one or both datasets.\n")
  genotype_data_aligned <- genotype_data
  Y_binary_aligned <- Y_binary
  thetas_aligned <- thetas
}

# =============================================================================
# 6. CORRELATE RARE VARIANT CARRIERS WITH DISEASE
# =============================================================================

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("CORRELATING RARE VARIANT CARRIERS WITH DISEASE\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

if (!is.null(genotype_data_aligned) && !is.null(Y_binary_aligned) && 
    nrow(genotype_data_aligned) == nrow(Y_binary_aligned)) {
  
  # Convert variant columns to binary carrier status
  # Assuming dosage format: 0 = non-carrier, >0 = carrier
  variant_carrier_matrix <- as.matrix(genotype_data_aligned[, variant_cols, with = FALSE])
  variant_carrier_matrix[variant_carrier_matrix > 0] <- 1  # Binary: carrier vs non-carrier
  variant_carrier_matrix[variant_carrier_matrix <= 0] <- 0
  
  # Calculate correlations
  correlations <- cor(variant_carrier_matrix, Y_binary_aligned, use = "pairwise.complete.obs")
  
  cat(sprintf("Calculated correlations: %d variants x %d diseases\n", 
              nrow(correlations), ncol(correlations)))
  
  # Find strongest associations
  max_corrs <- apply(correlations, 1, function(x) max(abs(x), na.rm = TRUE))
  top_variants <- order(max_corrs, decreasing = TRUE)[1:min(10, length(max_corrs))]
  
  cat("\nTop 10 variants by maximum correlation:\n")
  for (i in top_variants) {
    variant_name <- variant_cols[i]
    max_corr <- max_corrs[i]
    best_disease_idx <- which.max(abs(correlations[i, ]))
    best_disease <- disease_names[best_disease_idx]
    cat(sprintf("  %s: r=%.3f with %s\n", variant_name, max_corr, best_disease))
  }
  
  # Statistical tests (Fisher's exact test for binary outcomes)
  cat("\nPerforming Fisher's exact tests...\n")
  fisher_results <- data.frame(
    variant = character(),
    disease = character(),
    odds_ratio = numeric(),
    p_value = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Test top variants
  for (var_idx in top_variants[1:min(5, length(top_variants))]) {
    variant_carriers <- variant_carrier_matrix[, var_idx]
    
    for (dis_idx in 1:min(10, ncol(Y_binary_aligned))) {  # Test top 10 diseases
      disease_status <- Y_binary_aligned[, dis_idx]
      
      # Create contingency table
      contingency <- table(variant_carriers, disease_status)
      
      if (all(dim(contingency) == c(2, 2)) && sum(contingency) > 10) {
        fisher_test <- fisher.test(contingency)
        fisher_results <- rbind(fisher_results, data.frame(
          variant = variant_cols[var_idx],
          disease = disease_names[dis_idx],
          odds_ratio = fisher_test$estimate,
          p_value = fisher_test$p.value
        ))
      }
    }
  }
  
  # Adjust for multiple testing
  fisher_results$p_adj <- p.adjust(fisher_results$p_value, method = "fdr")
  fisher_results <- fisher_results[order(fisher_results$p_value), ]
  
  cat(sprintf("\nFound %d significant associations (FDR < 0.05):\n", 
              sum(fisher_results$p_adj < 0.05)))
  print(head(fisher_results[fisher_results$p_adj < 0.05, ], 10))
  
} else {
  cat("⚠️  Cannot calculate correlations - datasets not aligned.\n")
  correlations <- NULL
  fisher_results <- NULL
}

# =============================================================================
# 7. PLOT SIGNATURE LOADINGS VS DISEASE
# =============================================================================

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("PLOTTING SIGNATURE LOADINGS VS DISEASE\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

if (!is.null(thetas_aligned) && !is.null(Y_binary_aligned) && 
    nrow(thetas_aligned) == nrow(Y_binary_aligned)) {
  
  n_signatures <- ncol(thetas_aligned)
  n_diseases <- ncol(Y_binary_aligned)
  
  cat(sprintf("Plotting %d signatures vs %d diseases\n", n_signatures, n_diseases))
  
  # Create output directory
  output_dir <- "/mnt/project/exports/plots"
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Plot 1: Signature loadings by disease status (boxplots)
  for (sig_idx in 1:min(5, n_signatures)) {  # Plot first 5 signatures
    sig_name <- paste0("Signature_", sig_idx)
    sig_loadings <- thetas_aligned[, sig_idx]
    
    # Find diseases with sufficient cases
    disease_counts <- colSums(Y_binary_aligned)
    diseases_to_plot <- which(disease_counts >= 50 & disease_counts <= nrow(Y_binary_aligned) * 0.5)
    
    if (length(diseases_to_plot) > 0) {
      # Create data frame for plotting
      plot_data <- data.frame(
        signature_loading = rep(sig_loadings, length(diseases_to_plot)),
        disease_status = factor(rep(NA, length(sig_loadings) * length(diseases_to_plot)), 
                               levels = c("No Disease", "Has Disease")),
        disease = rep(disease_names[diseases_to_plot], each = length(sig_loadings))
      )
      
      idx <- 1
      for (dis_idx in diseases_to_plot) {
        disease_status_vec <- ifelse(Y_binary_aligned[, dis_idx] == 1, "Has Disease", "No Disease")
        plot_data$disease_status[idx:(idx + length(sig_loadings) - 1)] <- disease_status_vec
        idx <- idx + length(sig_loadings)
      }
      
      # Create boxplot
      p <- ggplot(plot_data, aes(x = disease_status, y = signature_loading, fill = disease_status)) +
        geom_boxplot(alpha = 0.7) +
        facet_wrap(~ disease, scales = "free_y", ncol = 3) +
        labs(
          title = sprintf("Signature %d Loadings by Disease Status", sig_idx),
          x = "Disease Status",
          y = "Signature Loading"
        ) +
        theme_minimal() +
        theme(legend.position = "none")
      
      ggsave(sprintf("%s/signature_%d_by_disease.png", output_dir, sig_idx), 
             p, width = 12, height = 8, dpi = 300)
      cat(sprintf("  Saved: signature_%d_by_disease.png\n", sig_idx))
    }
  }
  
  # Plot 2: Correlation heatmap (signatures x diseases)
  sig_disease_cor <- cor(thetas_aligned, Y_binary_aligned, use = "pairwise.complete.obs")
  
  png(sprintf("%s/signature_disease_correlation.png", output_dir), 
      width = 12, height = 8, units = "in", res = 300)
  corrplot(sig_disease_cor, method = "color", type = "full", 
           order = "hclust", tl.cex = 0.6, tl.col = "black",
           title = "Signature-Disease Correlations")
  dev.off()
  cat("  Saved: signature_disease_correlation.png\n")
  
} else {
  cat("⚠️  Cannot create plots - signature loadings or disease data missing.\n")
}

# =============================================================================
# 8. SAVE RESULTS
# =============================================================================

cat("\nSaving results...\n")

output_file <- "/mnt/project/exports/rare_variant_analysis_results.rds"
results <- list(
  correlations = correlations,
  fisher_results = fisher_results,
  variant_cols = variant_cols,
  disease_names = disease_names,
  n_patients = if (!is.null(genotype_data_aligned)) nrow(genotype_data_aligned) else NULL
)

saveRDS(results, output_file)
cat(sprintf("  Saved results to: %s\n", output_file))

cat("\n✅ Analysis complete!\n")

