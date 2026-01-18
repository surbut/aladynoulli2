# Mediation Analysis: Gene -> Signature -> Disease
# Runs on secure system with individual-level data access
# Exports only summary statistics (no individual-level data)

# This script tests if genetic effects on disease are mediated through signatures
# Uses Baron-Kenny approach:
# 1. Gene -> Disease (total effect)
# 2. Gene -> Signature (path a)
# 3. Signature -> Disease (path b, controlling for Gene)
# 4. Gene -> Disease | Signature (direct effect, controlling for Signature)

library(data.table)
library(parallel)

# Configuration
GENOTYPE_DIR <- "~/genogz/"
PHENO_DIR <- "~/pheno/"
SIGNATURE_AUC_FILE <- "~/pheno/signature_auc_phenotypes.txt"  # Signature AUC phenotypes file
OUTPUT_FILE <- "mediation_analysis_results.csv"

# Signature configuration
N_SIGNATURES <- 21  # Signatures 0-20
# Note: signature_auc_phenotypes.txt has one AUC value per signature (no age groups)

# Genes of interest (from RVAS results)
GENES_OF_INTEREST <- c("TTN", "LDLR", "BRCA2", "MIP")

# Load phenotype data
cat("Loading phenotype data...\n")
Y_ten <- readRDS(file.path(PHENO_DIR, "Y_binary.rds"))[1:400000,]
pid <- fread(file.path(PHENO_DIR, "processed_ids.csv"))
rownames(Y_ten) <- pid[, eid]
dxnames <- fread(file.path(PHENO_DIR, "disease_names.csv"), header = TRUE)[, x]
colnames(Y_ten) <- dxnames

# Load signature AUC phenotypes
cat("Loading signature AUC phenotypes...\n")
signature_data <- fread(SIGNATURE_AUC_FILE)

# Extract patient IDs (IID column)
cat(sprintf("  Loaded %d patients\n", nrow(signature_data)))
signature_pids <- as.character(signature_data[, IID])

# Extract signature AUC columns (SIG0_AUC, SIG1_AUC, ..., SIG20_AUC)
signature_cols <- grep("^SIG[0-9]+_AUC$", names(signature_data), value = TRUE)
cat(sprintf("  Found %d signature AUC columns\n", length(signature_cols)))

# Diagnostic: Show first few column names
cat("  Sample column names: ", paste(head(signature_cols, 5), collapse = ", "), "\n")

# Create signature matrix: (N, K) where K = number of signatures
signature_matrix <- matrix(NA, nrow = nrow(signature_data), ncol = N_SIGNATURES)

for (sig_idx in 0:(N_SIGNATURES-1)) {
  # Find column for this signature (e.g., SIG0_AUC, SIG1_AUC, etc.)
  sig_col <- paste0("SIG", sig_idx, "_AUC")
  
  if (sig_col %in% names(signature_data)) {
    signature_matrix[, sig_idx + 1] <- as.numeric(signature_data[[sig_col]])
  } else {
    cat(sprintf("  Warning: Column %s not found\n", sig_col))
  }
}

cat(sprintf("  Created signature matrix: %d patients × %d signatures\n", 
            nrow(signature_matrix), ncol(signature_matrix)))
cat(sprintf("  Signature matrix missing %%: %.2f%%\n", 
            mean(is.na(signature_matrix)) * 100))

# Store all mediation results
all_mediation_results <- data.table()

# Function to run mediation analysis for one gene-disease-signature combination
run_mediation <- function(gene, disease_idx, signature_idx, burden_vec, disease_vec, signature_vec, verbose = FALSE) {
  # Filter complete cases
  complete <- !is.na(burden_vec) & !is.na(disease_vec) & !is.na(signature_vec)
  n_complete <- sum(complete)
  
  if (n_complete < 100) {
    if (verbose) cat(sprintf("      Skipping: only %d complete cases (< 100)\n", n_complete))
    return(NULL)  # Too few observations
  }
  
  # Check for sufficient cases
  n_cases <- sum(disease_vec[complete] == 1)
  n_controls <- sum(disease_vec[complete] == 0)
  
  if (n_cases < 10 || n_controls < 10) {
    if (verbose) cat(sprintf("      Skipping: insufficient cases (%d) or controls (%d)\n", n_cases, n_controls))
    return(NULL)  # Too few cases/controls
  }
  
  # Prepare data
  dat <- data.frame(
    disease = disease_vec[complete],
    gene_burden = burden_vec[complete],
    signature = signature_vec[complete]
  )
  
  # 1. Total effect: Disease ~ Gene
  model_total <- tryCatch({
    glm(disease ~ gene_burden, 
        family = binomial(link = "logit"), 
        data = dat,
        control = list(maxit = 50))
  }, error = function(e) {
    if (verbose) cat(sprintf("      Model 1 (total) failed: %s\n", e$message))
    return(NULL)
  })
  
  if (is.null(model_total) || !model_total$converged) {
    if (verbose) cat("      Model 1 (total) did not converge\n")
    return(NULL)
  }
  
  coef_total <- coef(model_total)[2]
  se_total <- summary(model_total)$coefficients[2, 2]
  p_total <- summary(model_total)$coefficients[2, 4]
  
  # 2. Path a: Signature ~ Gene
  model_a <- tryCatch({
    lm(signature ~ gene_burden, data = dat)
  }, error = function(e) {
    if (verbose) cat(sprintf("      Model 2 (path a) failed: %s\n", e$message))
    return(NULL)
  })
  
  if (is.null(model_a)) {
    if (verbose) cat("      Model 2 (path a) failed\n")
    return(NULL)
  }
  
  coef_a <- coef(model_a)[2]
  se_a <- summary(model_a)$coefficients[2, 2]
  p_a <- summary(model_a)$coefficients[2, 4]
  
  # 3. Path b: Disease ~ Signature (controlling for Gene)
  model_b <- tryCatch({
    glm(disease ~ signature + gene_burden,
        family = binomial(link = "logit"),
        data = dat,
        control = list(maxit = 50))
  }, error = function(e) {
    if (verbose) cat(sprintf("      Model 3 (path b) failed: %s\n", e$message))
    return(NULL)
  })
  
  if (is.null(model_b) || !model_b$converged) {
    if (verbose) cat("      Model 3 (path b) did not converge\n")
    return(NULL)
  }
  
  coef_b <- coef(model_b)[2]  # Signature coefficient
  se_b <- summary(model_b)$coefficients[2, 2]
  p_b <- summary(model_b)$coefficients[2, 4]
  
  # 4. Direct effect: Disease ~ Gene | Signature
  coef_direct <- coef(model_b)[3]  # Gene coefficient controlling for Signature
  se_direct <- summary(model_b)$coefficients[3, 2]
  p_direct <- summary(model_b)$coefficients[3, 4]
  
  # Mediation statistics
  indirect_effect <- coef_a * coef_b
  total_effect <- coef_total
  proportion_mediated <- if (abs(total_effect) > 1e-10) indirect_effect / total_effect else NA
  
  # Sobel test for mediation (approximate)
  # SE of indirect effect: sqrt(b^2*SE_a^2 + a^2*SE_b^2)
  se_indirect <- sqrt(coef_b^2 * se_a^2 + coef_a^2 * se_b^2)
  
  if (se_indirect < 1e-10) {
    if (verbose) cat("      Sobel test failed: SE(indirect) too small\n")
    z_sobel <- NA
    p_sobel <- NA
  } else {
    z_sobel <- indirect_effect / se_indirect
    p_sobel <- 2 * (1 - pnorm(abs(z_sobel)))
  }
  
  return(data.table(
    gene = gene,
    disease = dxnames[disease_idx],
    signature = paste0("Signature_", signature_idx),
    n_complete = n_complete,
    n_cases = n_cases,
    n_controls = n_controls,
    # Total effect (Gene -> Disease)
    total_effect = coef_total,
    total_effect_se = se_total,
    total_effect_p = p_total,
    # Path a (Gene -> Signature)
    path_a = coef_a,
    path_a_se = se_a,
    path_a_p = p_a,
    # Path b (Signature -> Disease | Gene)
    path_b = coef_b,
    path_b_se = se_b,
    path_b_p = p_b,
    # Direct effect (Gene -> Disease | Signature)
    direct_effect = coef_direct,
    direct_effect_se = se_direct,
    direct_effect_p = p_direct,
    # Mediation statistics
    indirect_effect = indirect_effect,
    indirect_effect_se = se_indirect,
    proportion_mediated = proportion_mediated,
    sobel_z = z_sobel,
    sobel_p = p_sobel
  ))
}

# Process each gene
genotype_files <- list.files(GENOTYPE_DIR, pattern = "\\.raw\\.gz$", full.names = TRUE)

for (geno_file in genotype_files) {
  gene <- gsub(".*QCed\\.([A-Z0-9]+)\\..*", "\\1", basename(geno_file))
  
  # Only analyze genes of interest
  if (!gene %in% GENES_OF_INTEREST) {
    next
  }
  
  cat("Processing gene:", gene, "\n")
  
  # Load genotype data
  geno <- fread(geno_file)
  variant_cols <- setdiff(names(geno), c("FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"))
  X <- as.matrix(geno[, ..variant_cols])
  
  # Calculate rare variant burden
  burden <- rowSums(2 - X, na.rm = TRUE)
  geno_ids <- as.character(geno[, IID])
  Y_eids <- rownames(Y_ten)
  
  # Diagnostic: Check ID matching
  cat(sprintf("  Genotype IDs: %d patients\n", length(geno_ids)))
  cat(sprintf("  Phenotype IDs: %d patients\n", length(Y_eids)))
  cat(sprintf("  Signature IDs: %d patients\n", length(signature_pids)))
  
  # Try different ID matching strategies
  # Strategy 1: Direct character match
  common_char <- intersect(intersect(geno_ids, Y_eids), signature_pids)
  cat(sprintf("  Common patients (character match): %d\n", length(common_char)))
  
  # Strategy 2: Convert all to numeric (in case of leading zeros or formatting issues)
  # Only convert IDs that can be converted to numeric without losing information
  geno_ids_num <- tryCatch({
    as.character(as.numeric(geno_ids))
  }, warning = function(w) {
    geno_ids  # If conversion fails, use original
  })
  Y_eids_num <- tryCatch({
    as.character(as.numeric(Y_eids))
  }, warning = function(w) {
    Y_eids
  })
  sig_pids_num <- tryCatch({
    as.character(as.numeric(signature_pids))
  }, warning = function(w) {
    signature_pids
  })
  
  # Check for NAs (from failed conversion)
  if (any(is.na(geno_ids_num)) || any(is.na(Y_eids_num)) || any(is.na(sig_pids_num))) {
    cat("  ⚠️  Warning: Some IDs couldn't be converted to numeric (may have non-numeric characters)\n")
    common_num <- character(0)  # Don't use numeric matching if conversion fails
  } else {
    common_num <- intersect(intersect(geno_ids_num, Y_eids_num), sig_pids_num)
    cat(sprintf("  Common patients (numeric match): %d\n", length(common_num)))
  }
  
  # Strategy 3: Try matching original IID column if different from IID
  if ("FID" %in% names(geno) && length(common_char) < 100) {
    geno_fids <- as.character(geno[, FID])
    common_fid <- intersect(intersect(geno_fids, Y_eids), signature_pids)
    cat(sprintf("  Common patients (FID match): %d\n", length(common_fid)))
  }
  
  # Use the match strategy that gives the most common patients
  use_numeric <- length(common_num) > length(common_char) && length(common_num) > 0
  if (use_numeric) {
    cat("  Using numeric ID matching\n")
    common <- common_num
  } else {
    cat("  Using character ID matching\n")
    common <- common_char
  }
  
  if (length(common) < 100) {
    cat(sprintf("  ⚠️  Only %d common patients, skipping\n", length(common)))
    next
  }
  
  # Align data
  if (use_numeric) {
    # We're using numeric matching - match numeric IDs
    geno_idx <- match(common, geno_ids_num)
    Y_idx <- match(common, Y_eids_num)
    sig_idx_match <- match(common, sig_pids_num)
  } else {
    # Use character matching
    geno_idx <- match(common, geno_ids)
    Y_idx <- match(common, Y_eids)
    sig_idx_match <- match(common, signature_pids)
  }
  
  # Check for any missing indices
  if (any(is.na(geno_idx)) || any(is.na(Y_idx)) || any(is.na(sig_idx_match))) {
    cat(sprintf("  ⚠️  Warning: %d geno, %d Y, %d sig indices missing\n",
                sum(is.na(geno_idx)), sum(is.na(Y_idx)), sum(is.na(sig_idx_match))))
    # Remove NA indices
    valid_idx <- !is.na(geno_idx) & !is.na(Y_idx) & !is.na(sig_idx_match)
    geno_idx <- geno_idx[valid_idx]
    Y_idx <- Y_idx[valid_idx]
    sig_idx_match <- sig_idx_match[valid_idx]
    common <- common[valid_idx]
    cat(sprintf("  After removing NA indices: %d patients\n", length(common)))
  }
  
  burden_aligned <- burden[geno_idx]
  signature_aligned <- signature_matrix[sig_idx_match, ]  # (N, K) matrix
  
  cat(sprintf("  Matched %d patients across genotype, phenotype, and signature data\n", length(common)))
  
  # Diagnostic: Check diseases and signatures before looping
  cat("  Checking diseases...\n")
  disease_cases <- colSums(Y_ten[Y_idx, ], na.rm = TRUE)
  diseases_with_sufficient_cases <- sum(disease_cases >= 10)
  cat(sprintf("    Diseases with >= 10 cases: %d / %d\n", diseases_with_sufficient_cases, ncol(Y_ten)))
  
  cat("  Checking signatures...\n")
  sig_missing_pct <- colMeans(is.na(signature_aligned)) * 100
  signatures_with_few_missing <- sum(sig_missing_pct <= 10)
  cat(sprintf("    Signatures with <= 10%% missing: %d / %d\n", signatures_with_few_missing, N_SIGNATURES))
  cat(sprintf("    Signature missing %%: min=%.2f%%, max=%.2f%%, mean=%.2f%%\n", 
              min(sig_missing_pct), max(sig_missing_pct), mean(sig_missing_pct)))
  
  # For each disease and signature combination
  # (This is a nested loop - may want to parallelize or filter by significance first)
  n_analyzed <- 0
  n_successful <- 0
  n_skipped_disease <- 0
  n_skipped_signature <- 0
  n_skipped_mediation <- 0
  
  for (d_idx in 1:ncol(Y_ten)) {
    disease_vec <- Y_ten[Y_idx, d_idx]
    
    # Skip if disease has too few cases
    n_cases_disease <- sum(disease_vec, na.rm = TRUE)
    if (n_cases_disease < 10) {
      n_skipped_disease <- n_skipped_disease + 1
      next
    }
    
    # For each signature (0-20)
    for (sig_idx in 1:N_SIGNATURES) {
      n_analyzed <- n_analyzed + 1
      
      signature_vec <- signature_aligned[, sig_idx]
      
      # Skip if signature has too many missing values
      n_missing <- sum(is.na(signature_vec))
      if (n_missing > length(signature_vec) * 0.1) {
        n_skipped_signature <- n_skipped_signature + 1
        next  # More than 10% missing
      }
      
      # Run mediation analysis
      # Use verbose for first few combinations to debug
      verbose_debug <- (n_analyzed <= 5)
      result <- tryCatch({
        run_mediation(gene, d_idx, sig_idx - 1,  # sig_idx - 1 because signatures are 0-indexed
                      burden_aligned, disease_vec, signature_vec, verbose = verbose_debug)
      }, error = function(e) {
        cat(sprintf("    Error for gene=%s, disease=%s, signature=%d: %s\n", 
                    gene, dxnames[d_idx], sig_idx - 1, e$message))
        return(NULL)
      })
      
      if (!is.null(result)) {
        all_mediation_results <- rbind(all_mediation_results, result)
        n_successful <- n_successful + 1
      } else {
        n_skipped_mediation <- n_skipped_mediation + 1
      }
      
      # Progress update every 100 combinations
      if (n_analyzed %% 100 == 0) {
        cat(sprintf("  Processed %d combinations (%d successful, %d skipped: %d disease, %d signature, %d mediation)\n", 
                    n_analyzed, n_successful, n_skipped_disease + n_skipped_signature + n_skipped_mediation,
                    n_skipped_disease, n_skipped_signature, n_skipped_mediation))
      }
    }
  }
  
  cat(sprintf("  Gene %s: Total analyzed=%d, Successful=%d\n", 
              gene, n_analyzed, n_successful))
  cat(sprintf("    Skipped: %d (disease cases < 10) + %d (signature > 10%% missing) + %d (mediation returned NULL) = %d total\n",
              n_skipped_disease, n_skipped_signature, n_skipped_mediation,
              n_skipped_disease + n_skipped_signature + n_skipped_mediation))
}

# Save summary results (no individual-level data)
cat(sprintf("\nSaving mediation results to: %s\n", OUTPUT_FILE))
fwrite(all_mediation_results, OUTPUT_FILE)

cat(sprintf("✓ Saved %d mediation results\n", nrow(all_mediation_results)))

# Summary statistics
cat("\nTop 10 by proportion mediated:\n")
print(all_mediation_results[order(abs(proportion_mediated), decreasing = TRUE, na.last = TRUE)][1:10])

cat("\nTop 10 by Sobel test significance:\n")
print(all_mediation_results[order(sobel_p, decreasing = FALSE, na.last = TRUE)][1:10])

