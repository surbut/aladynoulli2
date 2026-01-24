#!/usr/bin/env Rscript

library(data.table)

cat("===== COMPARING MEDIATION RESULTS =====\n\n")

# Load both datasets
prev <- fread("mediation_analysis_results.csv")
new <- fread("/Users/sarahurbut/Downloads/mediation_analysis_results_filtered.csv")

cat(sprintf("Previous (unfiltered): %d results\n", nrow(prev)))
cat(sprintf("New (filtered): %d results\n", nrow(new)))
cat(sprintf("Removed: %d results (%.1f%% reduction)\n\n", 
            nrow(prev) - nrow(new), 
            (1 - nrow(new)/nrow(prev)) * 100))

# Check total effect significance
prev_sig_total <- sum(prev$total_effect_p < 0.05, na.rm = TRUE)
cat(sprintf("Previous: %d with significant total effects (%.1f%%)\n", 
            prev_sig_total, prev_sig_total/nrow(prev)*100))
cat(sprintf("New: %d with significant total effects (100%%)\n\n", nrow(new)))

# By gene
cat("Results by gene:\n")
for (gene in sort(unique(prev$gene))) {
  prev_gene <- prev[prev$gene == gene]
  new_gene <- new[new$gene == gene]
  
  cat(sprintf("  %s:\n", gene))
  cat(sprintf("    Previous: %d\n", nrow(prev_gene)))
  cat(sprintf("    New: %d\n", nrow(new_gene)))
  if (nrow(prev_gene) > 0) {
    cat(sprintf("    Reduction: %d (%.1f%%)\n", 
                nrow(prev_gene) - nrow(new_gene),
                (1 - nrow(new_gene)/nrow(prev_gene)) * 100))
  }
  cat("\n")
}

# Check enhancing mediations
cat("\n===== ENHANCING MEDIATIONS (Path A > 0, Path B > 0) =====\n")
prev_enhancing <- prev[path_a > 0 & path_b > 0]
new_enhancing <- new[path_a > 0 & path_b > 0]

cat(sprintf("Previous: %d enhancing mediations\n", nrow(prev_enhancing)))
cat(sprintf("New: %d enhancing mediations\n\n", nrow(new_enhancing)))

# Check if they have significant total effects
if (nrow(prev_enhancing) > 0) {
  prev_enhancing_sig <- sum(prev_enhancing$total_effect_p < 0.05, na.rm = TRUE)
  cat(sprintf("Previous enhancing with sig total: %d (%.1f%%)\n", 
              prev_enhancing_sig, prev_enhancing_sig/nrow(prev_enhancing)*100))
}

# Top results comparison
cat("\n===== TOP 10 BY SOBEL P-VALUE =====\n")
cat("\nPrevious (unfiltered):\n")
prev_top <- prev[order(sobel_p)][1:min(10, nrow(prev))]
print(prev_top[, .(gene, disease, signature, total_effect_p, sobel_p, proportion_mediated)])

cat("\nNew (filtered):\n")
new_top <- new[order(sobel_p)][1:min(10, nrow(new))]
print(new_top[, .(gene, disease, signature, total_effect_p, sobel_p, proportion_mediated)])






