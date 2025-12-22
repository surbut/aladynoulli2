#!/usr/bin/env Rscript
# Rare Variant Burden Analysis on RAP (DNAnexus)
# Correlates rare variant carriers with disease outcomes and signature loadings

# =============================================================================
# SETUP AND DNAnexus ACCESS
# =============================================================================

library(data.table)
library(dplyr)
library(ggplot2)
library(corrplot)

# Install DNAnexus R package if needed
if (!require("dnanexus", quietly = TRUE)) {
  cat("Installing DNAnexus R package...\n")
  # On RAP, this should already be available, but if not:
  # install.packages("dnanexus")
}

# DNAnexus project path (for reference - files already downloaded)
project_id <- "project-GJQXvjjJ1JqJ43PZ6y1XPqG9"
genotype_path <- "/project/to_Sarah/Signature/genotypes/"

# Local directories (matching your RAP setup)
genotype_dir <- path.expand("~/genogz")  # Genotype files location
pheno_dir <- path.expand("~/pheno")     # Phenotype/metadata location

cat("Genotype directory:", genotype_dir, "\n")
cat("Phenotype directory:", pheno_dir, "\n")

# =============================================================================
# LOAD LOCAL GENOTYPE FILES (already downloaded)
# =============================================================================

genotype_file <- NULL
genotype_files <- NULL

if (dir.exists(genotype_dir)) {
  cat("✓ Found genotype directory\n")
  
  # List all files (matching your code: list.files("~/genogz/"))
  all_files <- list.files(genotype_dir, full.names = TRUE)
  genotype_files <- list.files(genotype_dir)
  
  cat(sprintf("Found %d files in genotype directory:\n", length(genotype_files)))
  print(head(genotype_files, 10))
  if (length(genotype_files) > 10) cat("  ...\n")
  
  # Find .raw.gz files (PLINK format, compressed)
  raw_files <- all_files[grepl("\\.raw\\.gz$", all_files, ignore.case = TRUE)]
  
  if (length(raw_files) > 0) {
    cat(sprintf("\nFound %d .raw.gz files:\n", length(raw_files)))
    print(basename(raw_files))
    
    # Use first file (or you can specify which gene you want)
    # TET2 or TTN
    genotype_file <- raw_files[1]
    cat(sprintf("\nUsing: %s\n", basename(genotype_file)))
  } else {
    cat("⚠️  No .raw.gz files found\n")
  }
} else {
  cat("⚠️  Genotype directory not found:", genotype_dir, "\n")
  cat("   Trying DNAnexus download methods...\n")
}

# =============================================================================
# METHOD 1: Using DNAnexus R SDK (if files not found locally)
# =============================================================================

# Try to list files in the genotype directory
tryCatch({
  library(dnanexus)
  dx_set_workspace(project_id)
  
  cat("\nListing files in genotype directory...\n")
  files <- dx_ls(genotype_path)
  print(files)
  
  # Find genotype files
  genotype_files <- files[grepl(".*\\.(raw|txt|csv|bgen|vcf)$", files$name, ignore.case = TRUE), ]
  
  if (nrow(genotype_files) > 0) {
    cat(sprintf("\nFound %d genotype files:\n", nrow(genotype_files)))
    print(genotype_files)
    
    # Download first file (or specify which one you want)
    genotype_file_id <- genotype_files$id[1]
    local_genotype_file <- tempfile(fileext = paste0(".", tools::file_ext(genotype_files$name[1])))
    
    cat(sprintf("\nDownloading %s to %s...\n", genotype_files$name[1], local_genotype_file))
    dx_download(genotype_file_id, local_genotype_file)
    
    genotype_file <- local_genotype_file
    cat("✓ Downloaded genotype file\n")
  } else {
    cat("⚠️  No genotype files found in that directory\n")
    genotype_file <- NULL
  }
  
}, error = function(e) {
  cat("⚠️  DNAnexus R SDK not available or error:", e$message, "\n")
  cat("   Trying alternative method...\n")
  genotype_file <- NULL
})

# =============================================================================
# METHOD 2: Using dx command-line tool (if DNAnexus R SDK not available)
# =============================================================================

if (is.null(genotype_file) || !file.exists(genotype_file)) {
  cat("\nTrying dx command-line tool...\n")
  
  # List files using dx command
  list_cmd <- sprintf("dx ls %s:%s", project_id, genotype_path)
  cat("Running:", list_cmd, "\n")
  
  files_output <- tryCatch({
    system(list_cmd, intern = TRUE)
  }, error = function(e) {
    cat("⚠️  dx command failed:", e$message, "\n")
    NULL
  })
  
  if (!is.null(files_output) && length(files_output) > 0) {
    cat("Files found:\n")
    print(files_output)
    
    # Find genotype files
    genotype_file_names <- files_output[grepl(".*\\.(raw|txt|csv|bgen|vcf)$", files_output, ignore.case = TRUE)]
    
    if (length(genotype_file_names) > 0) {
      # Download first file
      file_to_download <- genotype_file_names[1]
      local_genotype_file <- tempfile(fileext = paste0(".", tools::file_ext(file_to_download)))
      
      download_cmd <- sprintf("dx download %s:%s%s -o %s", 
                              project_id, genotype_path, file_to_download, local_genotype_file)
      cat("\nRunning:", download_cmd, "\n")
      
      download_result <- tryCatch({
        system(download_cmd, intern = TRUE)
      }, error = function(e) {
        cat("⚠️  Download failed:", e$message, "\n")
        NULL
      })
      
      if (file.exists(local_genotype_file)) {
        genotype_file <- local_genotype_file
        cat("✓ Downloaded genotype file:", genotype_file, "\n")
      }
    }
  }
}

# =============================================================================
# METHOD 3: If files are already mounted/accessible locally
# =============================================================================

if (is.null(genotype_file) || !file.exists(genotype_file)) {
  cat("\nTrying local filesystem paths (if project is mounted)...\n")
  
  # Common mount points on RAP
  local_paths <- c(
    sprintf("/mnt/project/%s%s", project_id, genotype_path),
    sprintf("/project/%s%s", project_id, genotype_path),
    "/mnt/project/genotypes/",
    "/mnt/project/exports/genotypes/"
  )
  
  for (local_path in local_paths) {
    if (dir.exists(local_path)) {
      cat(sprintf("Found directory: %s\n", local_path))
      files <- list.files(local_path, pattern = ".*\\.(raw|txt|csv|bgen|vcf)$", 
                          full.names = TRUE, ignore.case = TRUE)
      if (length(files) > 0) {
        genotype_file <- files[1]
        cat(sprintf("  Using: %s\n", genotype_file))
        break
      }
    }
  }
}

# =============================================================================
# LOAD GENOTYPE DATA
# =============================================================================

if (!is.null(genotype_file) && file.exists(genotype_file)) {
  cat(sprintf("\nLoading genotype data from: %s\n", basename(genotype_file)))
  
  # Handle compressed files
  if (grepl("\\.gz$", genotype_file)) {
    cat("  File is compressed (.gz), decompressing...\n")
    # fread can handle .gz files directly
    temp_file <- genotype_file
  } else {
    temp_file <- genotype_file
  }
  
  # Try different formats
  if (grepl("\\.raw", genotype_file, ignore.case = TRUE)) {
    # PLINK .raw format (can be .raw or .raw.gz)
    cat("  Reading PLINK .raw format...\n")
    genotype_data <- fread(temp_file, sep = " ")
    cat(sprintf("  ✓ Loaded %d samples, %d columns\n", nrow(genotype_data), ncol(genotype_data)))
    
    # Show column names
    cat("  Column names (first 10):\n")
    print(head(names(genotype_data), 10))
    
  } else if (grepl("\\.csv", genotype_file, ignore.case = TRUE)) {
    # CSV format
    genotype_data <- fread(temp_file)
    cat(sprintf("  ✓ Loaded %d samples, %d columns\n", nrow(genotype_data), ncol(genotype_data)))
  } else if (grepl("\\.txt", genotype_file, ignore.case = TRUE)) {
    # TSV/TXT format
    genotype_data <- fread(temp_file, sep = "\t")
    cat(sprintf("  ✓ Loaded %d samples, %d columns\n", nrow(genotype_data), ncol(genotype_data)))
  } else {
    cat("⚠️  Unknown file format. Trying default fread...\n")
    genotype_data <- tryCatch({
      fread(temp_file)
    }, error = function(e) {
      cat("  Error:", e$message, "\n")
      NULL
    })
  }
  
  if (!is.null(genotype_data)) {
    # Extract patient IDs
    id_col <- if ("IID" %in% names(genotype_data)) "IID" else names(genotype_data)[1]
    patient_ids_geno <- as.character(genotype_data[[id_col]])
    cat(sprintf("  Patient ID column: %s\n", id_col))
    cat(sprintf("  Sample patient IDs: %s\n", paste(head(patient_ids_geno, 5), collapse = ", ")))
    
    # Extract variant columns (exclude ID columns)
    id_cols <- c("FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE")
    variant_cols <- setdiff(names(genotype_data), id_cols)
    
    cat(sprintf("  Found %d variant columns\n", length(variant_cols)))
    if (length(variant_cols) > 0) {
      cat("  Variant columns (first 5):\n")
      print(head(variant_cols, 5))
    }
    
    # Store gene name from filename
    gene_name <- if (grepl("TET2", basename(genotype_file))) "TET2" 
                 else if (grepl("TTN", basename(genotype_file))) "TTN"
                 else "Unknown"
    cat(sprintf("  Gene: %s\n", gene_name))
  }
  
} else {
  cat("\n⚠️  Could not find or download genotype file.\n")
  cat("   Please check:\n")
  cat("   1. DNAnexus project ID is correct\n")
  cat("   2. Path to genotypes is correct\n")
  cat("   3. You have access to the project\n")
  cat("   4. dx command-line tool is installed and configured\n")
  genotype_data <- NULL
  variant_cols <- NULL
  patient_ids_geno <- NULL
}

# =============================================================================
# LOAD Y ARRAY (Disease Outcomes)
# =============================================================================

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("LOADING Y ARRAY (Disease Outcomes)\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

# =============================================================================
# LOAD Y ARRAY (Disease Outcomes) - Matching your RAP setup
# =============================================================================

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("LOADING Y ARRAY (Disease Outcomes)\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

Y_binary <- NULL
Y_loaded <- FALSE

# Load Y_binary.rds from ~/pheno/ (matching your code)
y_path <- file.path(pheno_dir, "Y_binary.rds")
if (file.exists(y_path)) {
  cat(sprintf("Loading Y_binary.rds from: %s\n", y_path))
  Y <- readRDS(y_path)
  
  # Check dimensions
  if (length(dim(Y)) == 2) {
    cat(sprintf("  ✓ Loaded condensed binary Y matrix: %s (N x D)\n", paste(dim(Y), collapse = " x ")))
    
    # Subset to first 400k patients (matching your code: Y_ten=Y[1:400000,])
    if (nrow(Y) > 400000) {
      cat("  Subsetting to first 400,000 patients...\n")
      Y_ten <- Y[1:400000, ]
    } else {
      Y_ten <- Y
    }
    
    # Load patient IDs and set as rownames (matching your code)
    pid_path <- file.path(pheno_dir, "processed_ids.csv")
    if (file.exists(pid_path)) {
      cat("  Loading patient IDs...\n")
      pid <- fread(pid_path)
      if ("eid" %in% names(pid)) {
        rownames(Y_ten) <- pid[1:nrow(Y_ten), eid]
        cat(sprintf("  ✓ Set rownames from processed_ids.csv\n"))
      } else {
        cat("⚠️  'eid' column not found in processed_ids.csv\n")
      }
    }
    
    # Load disease names and set as colnames (matching your code)
    dxnames_path <- file.path(pheno_dir, "disease_names.csv")
    if (file.exists(dxnames_path)) {
      cat("  Loading disease names...\n")
      dxnames_df <- fread(dxnames_path, header = TRUE)
      if ("x" %in% names(dxnames_df)) {
        dxnames <- dxnames_df[, x]
        colnames(Y_ten) <- dxnames[1:ncol(Y_ten)]
        cat(sprintf("  ✓ Set colnames from disease_names.csv (%d diseases)\n", length(dxnames)))
      } else {
        cat("⚠️  'x' column not found in disease_names.csv\n")
        cat("    Available columns:", paste(names(dxnames_df), collapse = ", "), "\n")
      }
    }
    
    Y_binary <- Y_ten
    Y_loaded <- TRUE
    cat(sprintf("\n✓ Final Y_binary matrix: %s\n", paste(dim(Y_binary), collapse = " x ")))
    
  } else if (length(dim(Y)) == 3) {
    cat(sprintf("  Loaded full Y tensor: %s (N x D x T)\n", paste(dim(Y), collapse = " x ")))
    cat("    Condensing to binary matrix...\n")
    Y_binary <- apply(Y, c(1, 2), function(x) as.integer(any(x > 0)))
    cat(sprintf("    Condensed to: %s\n", paste(dim(Y_binary), collapse = " x ")))
    Y_loaded <- TRUE
  } else {
    cat("⚠️  Unexpected Y array dimensions:", dim(Y), "\n")
  }
} else {
  cat("⚠️  Y_binary.rds not found at:", y_path, "\n")
  cat("   Please ensure the file exists in ~/pheno/\n")
}

# =============================================================================
# 3. GET DISEASE NAMES AND PATIENT IDs FROM Y (already set above)
# =============================================================================

if (!is.null(Y_binary)) {
  # Extract disease names and patient IDs (already set as rownames/colnames above)
  disease_names <- colnames(Y_binary)
  patient_ids_Y <- rownames(Y_binary)
  
  cat(sprintf("\nY binary matrix summary:\n"))
  cat(sprintf("  Patients: %d\n", nrow(Y_binary)))
  cat(sprintf("  Diseases: %d\n", ncol(Y_binary)))
  cat(sprintf("  Sample diseases: %s\n", paste(head(disease_names, 5), collapse = ", ")))
  cat(sprintf("  Sample patient IDs: %s\n", paste(head(patient_ids_Y, 5), collapse = ", ")))
} else {
  disease_names <- NULL
  patient_ids_Y <- NULL
}

# =============================================================================
# 4. LOAD SIGNATURE LOADINGS (if available)
# =============================================================================

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("LOADING SIGNATURE LOADINGS\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

theta_paths <- c(
  "/mnt/project/exports/thetas.rds",
  "/mnt/project/exports/thetas.pt",
  "/mnt/project/exports/thetas.npy",
  file.path(Sys.getenv("HOME"), "thetas.rds"),
  file.path(getwd(), "thetas.rds"),
  "thetas.rds"
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
  cat("   You can continue with variant-disease correlations without signatures.\n")
}

# =============================================================================
# 5. ALIGN PATIENT IDs ACROSS DATASETS
# =============================================================================

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("ALIGNING PATIENT IDs ACROSS DATASETS\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

if (!is.null(patient_ids_geno) && !is.null(patient_ids_Y)) {
  # Find common patients
  common_patients <- intersect(patient_ids_geno, patient_ids_Y)
  cat(sprintf("  Found %d common patients\n", length(common_patients)))
  
  if (length(common_patients) == 0) {
    cat("⚠️  No common patients found! Check ID formats.\n")
    cat(sprintf("    Genotype IDs sample: %s\n", paste(head(patient_ids_geno, 5), collapse = ", ")))
    cat(sprintf("    Y array IDs sample: %s\n", paste(head(patient_ids_Y, 5), collapse = ", ")))
    cat("\n   Trying to match by converting to numeric...\n")
    # Try numeric matching
    geno_numeric <- as.numeric(patient_ids_geno)
    Y_numeric <- as.numeric(patient_ids_Y)
    common_numeric <- intersect(geno_numeric, Y_numeric)
    if (length(common_numeric) > 0) {
      cat(sprintf("  Found %d common patients (numeric match)\n", length(common_numeric)))
      geno_idx <- match(common_numeric, geno_numeric)
      Y_idx <- match(common_numeric, Y_numeric)
      common_patients <- common_numeric
    } else {
      geno_idx <- NULL
      Y_idx <- NULL
    }
  } else {
    geno_idx <- match(common_patients, patient_ids_geno)
    Y_idx <- match(common_patients, patient_ids_Y)
  }
  
  if (!is.null(geno_idx) && !is.null(Y_idx)) {
    # Subset all datasets to common patients
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
    
    cat(sprintf("  ✓ Aligned datasets: %d patients\n", length(common_patients)))
  } else {
    cat("⚠️  Could not align datasets.\n")
    genotype_data_aligned <- NULL
    Y_binary_aligned <- NULL
    thetas_aligned <- NULL
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
  if (sum(fisher_results$p_adj < 0.05) > 0) {
    print(head(fisher_results[fisher_results$p_adj < 0.05, ], 10))
  }
  
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
  output_dir <- file.path(getwd(), "plots")
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

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("SAVING RESULTS\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

output_file <- file.path(getwd(), "rare_variant_analysis_results.rds")
results <- list(
  correlations = correlations,
  fisher_results = fisher_results,
  variant_cols = variant_cols,
  disease_names = disease_names,
  n_patients = if (!is.null(genotype_data_aligned)) nrow(genotype_data_aligned) else NULL,
  gene_name = if (exists("gene_name")) gene_name else "Unknown"
)

saveRDS(results, output_file)
cat(sprintf("  ✓ Saved results to: %s\n", output_file))

cat("\n✅ Analysis complete!\n")
cat(sprintf("   Results saved to: %s\n", output_file))
cat(sprintf("   Plots saved to: %s\n", output_dir))

