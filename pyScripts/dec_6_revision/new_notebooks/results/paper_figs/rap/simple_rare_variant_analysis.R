# Simple Rare Variant Burden - Disease Association
# One analysis per gene, associations with all diseases
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
  
  cat(sprintf("  Matched %d patients\n", length(common)))
  
  # Correlate burden with ALL diseases
  corrs <- cor(burden[geno_idx], Y_ten[Y_idx, ], use = "pairwise.complete.obs")
  
  # Store results
  gene_results <- data.table(
    gene = gene,
    disease = dxnames,
    correlation = as.numeric(corrs)
  )
  all_results <- rbind(all_results, gene_results)
}

# Save results
fwrite(all_results, "rare_variant_burden_associations.csv")
cat(sprintf("\n✓ Saved %d associations (%d genes × %d diseases)\n", 
            nrow(all_results), length(unique(all_results$gene)), length(unique(all_results$disease))))

# Show top associations
cat("\nTop 10 associations:\n")
print(all_results[order(abs(correlation), decreasing = TRUE)][1:10])

