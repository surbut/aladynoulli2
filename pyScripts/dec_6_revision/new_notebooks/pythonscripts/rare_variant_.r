# Simple Rare Variant Burden - Disease Association
# One analysis per gene, associations with all diseases
# UPDATED: Now computes p-values for correlations AND log OR via logistic regression
# Output includes: correlation, p_value, log_or, log_or_p_value
library(data.table)

# Load data
genotype_files <- list.files("~/genogz/", pattern = "\\.raw\\.gz$", full.names = TRUE)
Y_ten <- readRDS("~/pheno/Y_binary.rds")[1:400000,]
pid <- fread("~/pheno/processed_ids.csv")
rownames(Y_ten) <- pid[, eid]
dxnames <- fread("~/pheno/disease_names.csv", header = TRUE)[, x]
colnames(Y_ten) <- dxnames

# Store all results
all_results <- data.table()

# Analyze each gene file
for (geno_file in genotype_files) {
  gene <- gsub(".*QCed\\.([A-Z0-9]+)\\..*", "\\1", basename(geno_file))
  cat("Analyzing:", gene, "\n")
  
  geno <- fread(geno_file)
  
  # Get variant columns (PLINK encoding: 2=ref hom, 1=het, 0=alt hom)
  variant_cols <- setdiff(names(geno), c("FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"))
  X <- as.matrix(geno[, ..variant_cols])
  
  # Calculate rare variant burden: rowSums(2 - X)
  burden <- rowSums(2 - X, na.rm = TRUE)
  
  # Align patients: geno has IDs in IID (or FID, same), Y_ten has eids as rownames
  geno_ids <- as.character(geno[, IID])  # Patient IDs from genotype file
  Y_eids <- rownames(Y_ten)              # eids from Y_ten (the 400K analyzed individuals)
  
  # Find common patients (matching geno IID with Y_ten rownames)
  common <- intersect(geno_ids, Y_eids)
  if (length(common) < 100) {
    cat(sprintf("  ⚠️  Only %d common patients, skipping\n", length(common)))
    next
  }
  
  # Get indices for alignment
  geno_idx <- match(common, geno_ids)
  Y_idx <- match(common, Y_eids)
  
  # Only compute GLM for these specific genes (others will have NA for log_or)
  genes_for_glm <- c("TTN", "LDLR", "BRCA2", "MIP")
  compute_glm_for_this_gene <- gene %in% genes_for_glm
  
  cat(sprintf("  Matched %d patients\n", length(common)))
  if (compute_glm_for_this_gene) {
    cat(sprintf("  ✓ Computing log OR via GLM for %s\n", gene))
  }
  
  # Correlate burden with ALL diseases and compute p-values
  # Also compute log OR via logistic regression (to match psi scale)
  # OPTIMIZATION: Only compute GLM for specific genes of interest to speed up
  corrs <- numeric(length(dxnames))
  p_values <- numeric(length(dxnames))
  log_ors <- numeric(length(dxnames))
  log_or_p_values <- numeric(length(dxnames))
  
  # Compute correlation, p-value, and log OR for each disease
  for (j in seq_along(dxnames)) {
    # Get non-missing pairs
    burden_vec <- burden[geno_idx]
    disease_vec <- Y_ten[Y_idx, j]
    complete_cases <- !is.na(burden_vec) & !is.na(disease_vec)
    
    if (sum(complete_cases) < 10) {
      corrs[j] <- NA
      p_values[j] <- NA
      log_ors[j] <- NA
      log_or_p_values[j] <- NA
      next
    }
    
    # Compute correlation test first (fast)
    test_result <- cor.test(burden_vec[complete_cases], 
                            disease_vec[complete_cases])
    corrs[j] <- test_result$estimate
    p_values[j] <- test_result$p.value
    
    # Only compute log OR via logistic regression for specific genes of interest
    # This skips GLM for most genes, dramatically speeding up
    if (!compute_glm_for_this_gene) {
      log_ors[j] <- NA
      log_or_p_values[j] <- NA
      # Don't skip - we still want to store correlation results for all genes
    } else {
    
    # Check for sufficient cases (need both cases and controls)
    n_cases <- sum(disease_vec[complete_cases] == 1)
    n_controls <- sum(disease_vec[complete_cases] == 0)
    
    if (n_cases < 5 || n_controls < 5) {
      # Too few cases or controls for reliable logistic regression
      log_ors[j] <- NA
      log_or_p_values[j] <- NA
    } else {
      # Fit logistic regression: disease ~ burden
      # Use faster method: control max iterations and use simpler optimizer
      tryCatch({
        # Use method="quick" if available, or set maxit to limit iterations
        glm_fit <- glm(disease_vec[complete_cases] ~ burden_vec[complete_cases], 
                      family = binomial(link = "logit"),
                      control = list(maxit = 25))  # Limit iterations for speed
        log_ors[j] <- coef(glm_fit)[2]  # Coefficient for burden (log OR)
        log_or_p_values[j] <- summary(glm_fit)$coefficients[2, 4]  # p-value
      }, error = function(e) {
        log_ors[j] <<- NA
        log_or_p_values[j] <<- NA
      })
    }
    }  # Close else block for compute_glm
  }
  
  # Store results
  gene_results <- data.table(
    gene = gene,
    disease = dxnames,
    correlation = corrs,
    p_value = p_values,
    log_or = log_ors,
    log_or_p_value = log_or_p_values
  )
  all_results <- rbind(all_results, gene_results)
}

# Save results
output_file <- "rare_variant_burden_associations.csv"  # With log OR columns
fwrite(all_results, output_file)
cat(sprintf("\n✓ Saved %d associations (%d genes × %d diseases) to %s\n", 
            nrow(all_results), length(unique(all_results$gene)), length(unique(all_results$disease)), output_file))
cat(sprintf("  Columns: gene, disease, correlation, p_value, log_or, log_or_p_value\n"))

# Show top associations
cat("\nTop 10 associations by absolute correlation:\n")
print(all_results[order(abs(correlation), decreasing = TRUE)][1:10])

# Show top associations by significance
cat("\nTop 10 associations by significance (lowest p-value):\n")
print(all_results[order(p_value, decreasing = FALSE)][1:10])
