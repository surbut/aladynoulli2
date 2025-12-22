# Simple Rare Variant Burden - Disease Association (OPTIMIZED VERSION)
# One analysis per gene, associations with all diseases
# Features:
# - Saves results after each gene (checkpointing)
# - Parallel processing for diseases
# - Resume capability if interrupted
# - Better error handling

library(data.table)
library(parallel)

# ============================================================================
# Configuration
# ============================================================================
CHECKPOINT_DIR <- "~/rare_variant_checkpoints"  # Directory for checkpoint files
FINAL_OUTPUT <- "~/rare_variant_burden_associations.csv"
NUM_CORES <- detectCores() - 1  # Leave one core free

# Create checkpoint directory
dir.create(CHECKPOINT_DIR, showWarnings = FALSE, recursive = TRUE)

# ============================================================================
# Load data (only once at the start)
# ============================================================================
cat("Loading data...\n")
genotype_files <- list.files("~/genogz/", pattern = "\\.raw\\.gz$", full.names = TRUE)
Y_ten <- readRDS("~/pheno/Y_binary.rds")[1:400000,]
pid <- fread("~/pheno/processed_ids.csv")
rownames(Y_ten) <- pid[, eid]
dxnames <- fread("~/pheno/disease_names.csv", header = TRUE)[, x]
colnames(Y_ten) <- dxnames
cov <- fread("~/pheno/baselinagefamh.csv")

cat(sprintf("Loaded: %d genes, %d diseases, %d patients\n", 
            length(genotype_files), length(dxnames), nrow(Y_ten)))

# ============================================================================
# Checkpoint: Find already processed genes
# ============================================================================
processed_genes <- character(0)
if (dir.exists(CHECKPOINT_DIR)) {
  checkpoint_files <- list.files(CHECKPOINT_DIR, pattern = "^gene_.*\\.csv$", full.names = TRUE)
  processed_genes <- gsub(".*gene_([A-Z0-9]+)\\.csv$", "\\1", basename(checkpoint_files))
  cat(sprintf("Found %d already processed genes: %s\n", 
              length(processed_genes), paste(head(processed_genes, 5), collapse = ", ")))
}

# Load existing results if resuming
all_results <- data.table()
if (length(processed_genes) > 0) {
  for (gene_file in checkpoint_files) {
    tryCatch({
      gene_data <- fread(gene_file)
      all_results <- rbind(all_results, gene_data)
    }, error = function(e) {
      cat(sprintf("  Warning: Could not load %s: %s\n", gene_file, e$message))
    })
  }
  cat(sprintf("Loaded %d existing associations from checkpoints\n", nrow(all_results)))
}

# ============================================================================
# Function to analyze one gene-disease pair
# ============================================================================
analyze_gene_disease <- function(gene, disease_idx, disease_name, 
                                  burden_aligned, y_aligned, cov_matrix) {
  y <- y_aligned[, disease_idx]
  
  # Skip if too few cases
  n_cases <- sum(y == 1, na.rm = TRUE)
  if (n_cases < 10) {
    return(data.table(
      gene = gene,
      disease = disease_name,
      beta = NA_real_,
      se = NA_real_,
      or = NA_real_,
      z = NA_real_,
      p_value = NA_real_,
      n_cases = n_cases,
      n_total = sum(!is.na(y))
    ))
  }
  
  # Logistic regression: disease ~ burden + covariates
  # Create data frame for glm
  glm_data <- data.frame(
    y = y,
    burden = burden_aligned,
    cov_matrix
  )
  
  fit <- tryCatch({
    glm(y ~ burden + ., data = glm_data, 
        family = binomial(link = "logit"))
  }, error = function(e) {
    return(NULL)
  })
  
  if (is.null(fit) || !fit$converged) {
    return(data.table(
      gene = gene,
      disease = disease_name,
      beta = NA_real_,
      se = NA_real_,
      or = NA_real_,
      z = NA_real_,
      p_value = NA_real_,
      n_cases = n_cases,
      n_total = sum(!is.na(y))
    ))
  }
  
  # Extract coefficients for burden
  coef_summary <- summary(fit)$coefficients
  burden_idx <- which(rownames(coef_summary) == "burden")
  
  if (length(burden_idx) == 0 || burden_idx > nrow(coef_summary)) {
    return(data.table(
      gene = gene,
      disease = disease_name,
      beta = NA_real_,
      se = NA_real_,
      or = NA_real_,
      z = NA_real_,
      p_value = NA_real_,
      n_cases = n_cases,
      n_total = sum(!is.na(y))
    ))
  }
  
  beta <- coef_summary[burden_idx, "Estimate"]
  se <- coef_summary[burden_idx, "Std. Error"]
  z <- coef_summary[burden_idx, "z value"]
  p_value <- coef_summary[burden_idx, "Pr(>|z|)"]
  or <- exp(beta)
  
  data.table(
    gene = gene,
    disease = disease_name,
    beta = beta,
    se = se,
    or = or,
    z = z,
    p_value = p_value,
    n_cases = n_cases,
    n_total = sum(!is.na(y))
  )
}

# ============================================================================
# Main processing loop
# ============================================================================
cat("\n", "=", rep("=", 70), "\n", sep = "")
cat("Starting gene-by-gene analysis\n")
cat("=", rep("=", 70), "\n\n", sep = "")

total_genes <- length(genotype_files)
processed_count <- 0
skipped_count <- 0

for (gene_idx in seq_along(genotype_files)) {
  geno_file <- genotype_files[gene_idx]
  gene <- gsub(".*QCed\\.([A-Z0-9]+)\\..*", "\\1", basename(geno_file))
  
  # Skip if already processed
  if (gene %in% processed_genes) {
    cat(sprintf("[%d/%d] Skipping %s (already processed)\n", 
                gene_idx, total_genes, gene))
    skipped_count <- skipped_count + 1
    next
  }
  
  cat(sprintf("\n[%d/%d] Analyzing: %s\n", gene_idx, total_genes, gene))
  start_time <- Sys.time()
  
  # Checkpoint file for this gene
  checkpoint_file <- file.path(CHECKPOINT_DIR, paste0("gene_", gene, ".csv"))
  
  tryCatch({
    # Load genotype data
    geno <- fread(geno_file)
    
    # Get variant columns (PLINK encoding: 2=ref hom, 1=het, 0=alt hom)
    variant_cols <- setdiff(names(geno), c("FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"))
    X <- as.matrix(geno[, ..variant_cols])
    
    # Calculate rare variant burden: rowSums(2 - X)
    burden <- rowSums(2 - X, na.rm = TRUE)
    
    # Align patients
    geno_ids <- as.character(geno[, IID])
    Y_eids <- rownames(Y_ten)
    cov_eids <- as.character(cov$identifier)
    
    # Find common patients (intersect all three sets)
    common <- intersect(intersect(geno_ids, Y_eids), cov_eids)
    if (length(common) < 100) {
      cat(sprintf("  ⚠️  Only %d common patients, skipping\n", length(common)))
      skipped_count <- skipped_count + 1
      next
    }
    
    # Get indices for alignment
    geno_idx_aligned <- match(common, geno_ids)
    Y_idx_aligned <- match(common, Y_eids)
    cov_idx_aligned <- match(common, cov_eids)
    
    # Get aligned data
    burden_aligned <- burden[geno_idx_aligned]
    Y_aligned <- Y_ten[Y_idx_aligned, , drop = FALSE]
    
    # Get covariate matrix (age, sex, PCs)
    cov_cols <- c("age", "sex", grep("PC", names(cov), value = TRUE))
    cov_matrix <- as.matrix(cov[cov_idx_aligned, ..cov_cols])
    
    cat(sprintf("  Matched %d patients\n", length(common)))
    
    # Process diseases in parallel chunks (to avoid memory issues)
    chunk_size <- 50  # Process 50 diseases at a time
    num_diseases <- ncol(Y_aligned)
    num_chunks <- ceiling(num_diseases / chunk_size)
    
    gene_results <- list()
    
    for (chunk in 1:num_chunks) {
      start_d <- (chunk - 1) * chunk_size + 1
      end_d <- min(chunk * chunk_size, num_diseases)
      disease_indices <- start_d:end_d
      
      # Process chunk in parallel
      chunk_results <- mclapply(disease_indices, function(d) {
        analyze_gene_disease(gene, d, dxnames[d], 
                            burden_aligned, Y_aligned, cov_matrix)
      }, mc.cores = NUM_CORES)
      
      gene_results <- c(gene_results, chunk_results)
    }
    
    # Combine results for this gene
    gene_results_dt <- rbindlist(gene_results)
    
    # Save checkpoint immediately
    fwrite(gene_results_dt, checkpoint_file)
    cat(sprintf("  ✓ Saved checkpoint: %d associations\n", nrow(gene_results_dt)))
    
    # Add to running total
    all_results <- rbind(all_results, gene_results_dt)
    
    # Save combined results periodically (every 5 genes)
    if (processed_count %% 5 == 0 && processed_count > 0) {
      fwrite(all_results, FINAL_OUTPUT)
      cat(sprintf("  ✓ Updated combined output file\n"))
    }
    
    processed_count <- processed_count + 1
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    cat(sprintf("  ✓ Completed in %.1f seconds\n", elapsed))
    
  }, error = function(e) {
    cat(sprintf("  ✗ ERROR processing %s: %s\n", gene, e$message))
    # Continue with next gene instead of crashing
  })
}

# ============================================================================
# Final save and summary
# ============================================================================
cat("\n", "=", rep("=", 70), "\n", sep = "")
cat("Analysis complete!\n")
cat("=", rep("=", 70), "\n\n", sep = "")

# Final save of combined results
fwrite(all_results, FINAL_OUTPUT)
cat(sprintf("✓ Final results saved to: %s\n", FINAL_OUTPUT))
cat(sprintf("\nSummary:\n"))
cat(sprintf("  Processed: %d genes\n", processed_count))
cat(sprintf("  Skipped: %d genes\n", skipped_count))
cat(sprintf("  Total associations: %d\n", nrow(all_results)))
cat(sprintf("  Unique genes: %d\n", length(unique(all_results$gene))))
cat(sprintf("  Unique diseases: %d\n", length(unique(all_results$disease))))

# Show top associations
if (nrow(all_results[!is.na(beta)]) > 0) {
  cat("\nTop 10 associations (by absolute beta):\n")
  top_results <- all_results[!is.na(beta)][order(abs(beta), decreasing = TRUE)][1:min(10, nrow(all_results[!is.na(beta)]))]
  print(top_results)
} else {
  cat("\nNo associations with valid beta values found.\n")
}

cat("\n✓ Done!\n")

