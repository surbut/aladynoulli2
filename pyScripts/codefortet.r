
# Load required libraries
library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stats)
library(logistf)
library(pheatmap)
library(UpSetR)
library(readr)

# Function to read in the data files with correct column names
read_gwas_data <- function(file_path) {
  # Read the first line to get column names
  con <- file(file_path, "r")
  header_line <- readLines(con, n = 1)
  close(con)
  
  # Parse column names
  col_names <- strsplit(header_line, "\t")[[1]]
  
  # Read the data with specified column names
  data <- read_delim(
    file_path, 
    delim = "\t", 
    skip = 1,  # Skip the header row since we're manually specifying column names
    col_names = col_names
  ) %>%
    select(`#CHR`, POS, UID, EA, OA, BETA, SE, LOG10P, rsid) %>%
    rename(CHR = `#CHR`) %>%
    mutate(CHR = as.numeric(CHR))
  
  return(data)
}

# Function to find SNPs within 1MB window
find_overlapping_snps <- function(snp_data, target_snps, window_size = 1000000) {
  overlapping <- logical(nrow(target_snps))
  
  for(i in 1:nrow(target_snps)) {
    target_chr <- target_snps$CHR[i]
    target_pos <- target_snps$POS[i]
    
    # Find SNPs in same chromosome within window
    same_chr <- snp_data$CHR == target_chr
    within_window <- abs(snp_data$POS - target_pos) <= window_size
    
    overlapping[i] <- any(same_chr & within_window)
  }
  
  return(overlapping)
}

# Read in all the data files
angina <- read_gwas_data("/Users/sarahurbut/Library/CloudStorage/DB_backup_5132025941p/tetgwas/result/10_loci/Angina_pectoris_ukb_eur_regenie_af1.sig.lead.sumstats.txt")
cor_athero <- read_gwas_data("/Users/sarahurbut/Library/CloudStorage/DB_backup_5132025941p/tetgwas/result/10_loci/Coronary_atherosclerosis_ukb_eur_regenie_af1.sig.lead.sumstats.txt")
hyperchol <- read_gwas_data("/Users/sarahurbut/Library/CloudStorage/DB_backup_5132025941p/tetgwas/result/10_loci/Hypercholesterolemia_ukb_eur_regenie_af1.sig.lead.sumstats.txt")
mi <- read_gwas_data("/Users/sarahurbut/Library/CloudStorage/DB_backup_5132025941p/tetgwas/result/10_loci/Myocardial_infarction_ukb_eur_regenie_af1.sig.lead.sumstats.txt")
acute_ihd <- read_gwas_data("/Users/sarahurbut/Library/CloudStorage/DB_backup_5132025941p/tetgwas/result/10_loci/Other_acute_and_subacute_forms_of_ischemic_heart_disease_ukb_eur_regenie_af1.sig.lead.sumstats.txt")
chronic_ihd <- read_gwas_data("/Users/sarahurbut/Library/CloudStorage/DB_backup_5132025941p/tetgwas/result/10_loci/Other_chronic_ischemic_heart_disease,_unspecified_ukb_eur_regenie_af1.sig.lead.sumstats.txt")
sig5 <- read_gwas_data("/Users/sarahurbut/Library/CloudStorage/Dropbox/result326/10_loci/SIG5_AUC_ukb_eur_regenie_af1.sig.lead.sumstats.txt")

# Add source information
angina$source <- "Angina"
cor_athero$source <- "Cor_Athero"
hyperchol$source <- "Hypercholest"
mi$source <- "MI"
acute_ihd$source <- "Acute_IHD"
chronic_ihd$source <- "Chronic_IHD"
sig5$source <- "SIG5_AUC"

# Get all unique variants with genomic coordinates
all_variants <- bind_rows(angina, cor_athero, hyperchol, mi, acute_ihd, chronic_ihd, sig5) %>%
  select(rsid, CHR, POS) %>%
  distinct() %>%
  arrange(CHR, POS)

# Create presence/absence matrix using 1MB windows
presence_matrix_1mb <- data.frame(
  UID = all_variants$rsid,
  CHR = all_variants$CHR,
  POS = all_variants$POS,
  Angina = find_overlapping_snps(angina, all_variants),
  Cor_Athero = find_overlapping_snps(cor_athero, all_variants),
  Hypercholest = find_overlapping_snps(hyperchol, all_variants),
  MI = find_overlapping_snps(mi, all_variants),
  Acute_IHD = find_overlapping_snps(acute_ihd, all_variants),
  Chronic_IHD = find_overlapping_snps(chronic_ihd, all_variants),
  SIG5_AUC = find_overlapping_snps(sig5, all_variants)
)

# Convert to binary matrix for UpSetR
upset_matrix_1mb <- presence_matrix_1mb %>%
  select(-UID, -CHR, -POS) %>%
  mutate_all(as.numeric)

# Get set sizes for labeling
set_sizes_1mb <- colSums(upset_matrix_1mb)
set_labels_1mb <- paste0(
  names(set_sizes_1mb), 
  " (", set_sizes_1mb, ")"
)

# Create UpSet plot with 1MB windows
upset_plot_1mb <- upset(
  upset_matrix_1mb,
  nsets = 7,
  order.by = "freq",
  sets = names(upset_matrix_1mb),
  keep.order = TRUE,
  set.metadata = list(
    data = data.frame(
      sets = names(set_sizes_1mb),
      size = set_sizes_1mb
    ),
    plots = list(
      list(
        type = "hist",
        column = "size",
        assign = 20
      )
    )
  ),
  mainbar.y.label = "Intersection Size (1MB Windows)",
  sets.x.label = "Variants Per Phenotype",
  text.scale = 1.2,
  point.size = 3,
  line.size = 1,
  mb.ratio = c(0.6, 0.4),
  set_size.show = TRUE
)

# Print the upset plot
print(upset_plot_1mb)

# Save the plot
pdf("cardiovascular_variants_upset_plot_1mb_windows.pdf", width = 12, height = 8)
print(upset_plot_1mb)
dev.off()

# Compare with original exact matching approach
# Create original presence matrix for comparison
angina_variants <- angina$rsid
cor_athero_variants <- cor_athero$rsid
hyperchol_variants <- hyperchol$rsid
mi_variants <- mi$rsid
acute_ihd_variants <- acute_ihd$rsid
chronic_ihd_variants <- chronic_ihd$rsid
sig5_variants <- sig5$rsid

all_variants_exact <- unique(c(
  angina_variants, cor_athero_variants, hyperchol_variants, 
  mi_variants, acute_ihd_variants, chronic_ihd_variants, 
  sig5_variants
))

presence_matrix_exact <- data.frame(
  UID = all_variants_exact,
  Angina = all_variants_exact %in% angina_variants,
  Cor_Athero = all_variants_exact %in% cor_athero_variants,
  Hypercholest = all_variants_exact %in% hyperchol_variants,
  MI = all_variants_exact %in% mi_variants,
  Acute_IHD = all_variants_exact %in% acute_ihd_variants,
  Chronic_IHD = all_variants_exact %in% chronic_ihd_variants,
  SIG5_AUC = all_variants_exact %in% sig5_variants
)

upset_matrix_exact <- presence_matrix_exact %>%
  select(-UID) %>%
  mutate_all(as.numeric)

set_sizes_exact <- colSums(upset_matrix_exact)

# Print comparison
cat("Comparison of overlap methods:\n")
cat("Exact SNP matching:\n")
print(set_sizes_exact)
cat("\n1MB window approach:\n")
print(set_sizes_1mb)
cat("\nIncrease in overlap with 1MB windows:\n")
print(set_sizes_1mb - set_sizes_exact)