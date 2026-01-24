#!/usr/bin/env Rscript

# Check how many existing results have significant total effects

library(data.table)

cat("===== CHECKING EXISTING MEDIATION RESULTS =====\n\n")

# Load the full results
results <- fread("mediation_analysis_results.csv")
cat(sprintf("Total results: %d\n\n", nrow(results)))

# Check total effect significance
results[, total_effect_sig := total_effect_p < 0.05]

cat("Results with significant total effects (p < 0.05):\n")
cat(sprintf("  Count: %d (%.1f%%)\n", sum(results$total_effect_sig), mean(results$total_effect_sig) * 100))
cat(sprintf("  Count with non-significant: %d (%.1f%%)\n\n", 
            sum(!results$total_effect_sig), mean(!results$total_effect_sig) * 100))

# Filter to significant total effects
results_sig <- results[total_effect_sig == TRUE]
cat(sprintf("Filtered results (significant total effects): %d\n\n", nrow(results_sig)))

# Check by gene
cat("By gene (significant total effects only):\n")
for (gene in sort(unique(results_sig$gene))) {
  gene_data <- results_sig[results_sig$gene == gene]
  cat(sprintf("  %s: %d mediations\n", gene, nrow(gene_data)))
}

# Check top results
cat("\nTop 20 by Sobel p-value (with significant total effects):\n")
top_results <- results_sig[order(sobel_p)][1:min(20, nrow(results_sig))]
print(top_results[, .(gene, disease, signature, total_effect_p, path_a_p, path_b_p, sobel_p, proportion_mediated)])

# Save filtered results
fwrite(results_sig, "mediation_results_significant_total_effects.csv")
cat("\n✓ Saved filtered results to: mediation_results_significant_total_effects.csv\n")






