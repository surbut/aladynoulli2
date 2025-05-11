
  genotype_file_default="/Users/sarahurbut/Dropbox/genotype_raw/genotype_dosage_20250416.raw";
  library(data.table)
  
  genotype_file_sig5="/Users/sarahurbut/Dropbox/genotype_raw/genotype_dosage_20250415.raw";
  covariate_file="/Users/sarahurbut/Dropbox/for_regenie/ukbb_covariates_400k.txt";
  snp_list_dir="/Users/sarahurbut/Dropbox/snp_lists";
  phenotype_dir="/Users/sarahurbut/Dropbox/for_regenie/case_control_phenotypes";
  sig_stats_dir="~/Dropbox/result326/10_loci";



library(broom)

signatures_to_analyze=c(0,5,7,14,15,18)
sigs_list=vector("list", length(signatures_to_analyze))
names(sigs_list)=as.character(signatures_to_analyze)

for (sig in signatures_to_analyze){
print(paste("analysing sig",sig))
# Read signature summary statistics
sig_stats_file = paste0(sig_stats_dir,"/SIG",sig,"_AUC_ukb_eur_regenie_af1.sig.lead.sumstats.txt")



# Read the file with headers
sig_stats =fread(sig_stats_file)

colnames(sig_stats)
# Create mapping of SNP ID to signature z-stat


# Select appropriate genotype file based on signature
genotype_file = ifelse(sig==5,genotype_file_sig5,genotype_file_default)

# Read significant SNPs for this signature
snp_file = fread(paste0("~/Dropbox/snp_lists/snp_list_sig",sig,".txt"))
sig_snps_vec <- as.character(snp_file$V2)
# Read case-control status
phenotypes = fread(paste0("~/Dropbox/for_regenie/case_control_phenotypes/case_control_sig",sig,".tsv"))
# Read genotypes (only for significant SNPs)
genotypes = fread(genotype_file)

# Create mapping from column names to base SNP IDs
geno_cols = colnames(genotypes)
col_to_snp = geno_cols[-c(1:6)]

# Extract base SNP ID by removing allele suffix
base_snp = stringr::str_split_fixed(col_to_snp,"_",n=2)[,1] # Split on underscore for rsIDs
colnames(genotypes)[-c(1:6)]=base_snp
snps_present <- intersect(sig_snps_vec, colnames(genotypes))
genotypes_subset <- genotypes[, .SD, .SDcols = c("FID", snps_present)]
# Find matching columns for our SNPs of interest



covariates = fread("~/Dropbox/for_regenie/ukbb_covariates_400k.txt")


# Keep only needed columns and rename 'identifier' to match merge
covariates=data.frame(covariates)
covariates = covariates[,c(1,2,5:24)]
colnames(covariates)[1]="FID"


merged_data = merge(genotypes_subset,covariates,by="FID")
merged_data=merge(merged_data,phenotypes)


pheno_cols <- setdiff(colnames(merged_data), c("FID", snps_present, "sex", paste0("PC", 1:20)))
pc_cols <- paste0("PC", 1:20)

# Initialize results storage


results_list <- list()

for (snp in snps_present) {
  sum_data=sig_stats[sig_stats$rsid==snp,]
  for (pheno in pheno_cols) {
    # Prepare data
    model_data <- merged_data[, .SD,.SDcols=c(snp, pheno, "sex", pc_cols)]
    model_data <- na.omit(model_data)
    if (nrow(model_data) < 10) next
    
    # Fit logistic regression
    formula_str <- paste(pheno, "~", snp, "+ sex +", paste(pc_cols, collapse = " + "))
    fit <- try(glm(as.formula(formula_str), data = model_data, family = binomial()), silent = TRUE)
    if (inherits(fit, "try-error")) next
    
    # Extract results for the SNP
    tidy_fit <- tidy(fit)
    snp_row <- tidy_fit[tidy_fit$term == snp, ]
    if (nrow(snp_row) == 1) {
      results_list[[length(results_list) + 1]] <- data.frame(
        SNP = snp,
        Phenotype = pheno,
        Beta = snp_row$estimate,
        Z_statistic = snp_row$statistic,
        P_value = snp_row$p.value,
        N = nrow(model_data),
        sig_snp_beta=sum_data$BETA,
        sig_snp_se=sum_data$SE,
        sig_snp_p=sum_data$LOG10P
      )
    }
  }
}
sigs_list[[as.character(sig)]]=results_list
}


list_final=lapply(1:length(sigs_list),function(x){
  r=do.call(rbind,sigs_list[[x]])
  r$sig=names(sigs_list)[x]
  return(r)})



  r=do.call(rbind,list_final)
  

  
  library(data.table)
  
  # Directory containing your case_control files
  phenotype_dir <- "~/Dropbox/for_regenie/case_control_phenotypes"
  
  # List all files
  case_files <- list.files(phenotype_dir, pattern = "^case_control_sig\\d+\\.tsv$", full.names = TRUE)
  
  # Create a named list mapping signature number to phenotype names
  sig_to_phenos <- setNames(vector("list", length(case_files)), nm = gsub(".*sig(\\d+)\\.tsv$", "\\1", case_files))
  
  for (f in case_files) {
    sig_num <- gsub(".*sig(\\d+)\\.tsv$", "\\1", f)
    dt <- fread(f, nrows = 0) # Only read header
    phenos <- setdiff(colnames(dt), "FID")
    sig_to_phenos[[sig_num]] <- phenos
  }
  
  library(dplyr)
  
  
  library(ggplot2)
  library(dplyr)
  
  # Add signature log10P column
  r$Signature_log10P <- r$sig_snp_p
  

  # Shorten phenotype names (optional)
  r$Phenotype_short <- gsub("_", " ", r$Phenotype) # or use a mapping for even shorter names
  
  
  # For each signature, add a row for the signature p-value
  signature_rows <- r %>%
    group_by(sig, SNP) %>%
    summarise(
      Phenotype = "Signature",
      P_value = 10^(-unique(sig_snp_p)),  # Convert log10P to P
      sig = unique(sig)
    ) %>%
    ungroup()
  

  
  # Combine with the original data
  r_augmented <- bind_rows(r, signature_rows)
  
  r_augmented <- r_augmented %>%
    mutate(Significant = P_value < 5e-8)
  
  library(ggplot2)
  
  
  saveRDS(r_augmented,"r_augmented.rds")
p=ggplot(r_augmented, aes(x = Phenotype, y = SNP, fill = -log10(P_value))) +
    geom_tile(color = "white") +
    geom_tile(data = subset(r_augmented, Significant), fill = NA, color = "black", size = 1.2) + # bold border for sig
    geom_text(aes(label = ifelse(Significant, "*", "")), color = "black", size = 5, vjust = 0.5) +
    facet_wrap(~as.factor(sig), scales = "free", ncol = 2) +
    scale_fill_gradient(low = "#eeeeee", high = "#2c3e50", name = "-log10(p)") +
    theme_minimal(base_size = 5) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
      axis.text.y = element_text(size = 5)
    ) +
    labs(
      title = "SNP-Phenotype Association Heatmap by Signature",
      x = "Phenotype",
      y = "SNP"
    )


p = ggplot(r_augmented, aes(x = Phenotype, y = SNP, fill = -log10(P_value))) +
  geom_tile(color = "white") +
  geom_tile(data = subset(r_augmented, Significant), fill = NA, color = "black", size = 1.2) + # bold border for sig
  geom_text(aes(label = ifelse(Significant, "*", "")), color = "black", size = 5, vjust = 0.5) +
  facet_grid(rows = vars(sig), scales = "free_y", space = "free_y") +
  scale_fill_gradient(low = "#eeeeee", high = "#2c3e50", name = "-log10(p)") +
  theme_minimal(base_size = 5) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
    axis.text.y = element_text(size = 5),
    strip.text.y = element_text(angle = 0, size = 7)
  ) +
  labs(
    title = "SNP-Phenotype Association Heatmap by Signature",
    x = "Phenotype",
    y = "SNP"
  )

ggsave(plot = p,filename = "SNP-Phenotype Association Heatmap by Signature.pdf",dpi=300,width = 10,height = 10)
  
  
p = ggplot(r_augmented, aes(x = Phenotype, y = SNP, fill = as.factor(sig))) +
  geom_tile(color = "white") +
  geom_tile(
    data = subset(r_augmented, Significant),
    color = "black", fill = NA, size = 0.8
  ) +
  geom_text(aes(label = ifelse(Significant, "*", "")), color = "black", size = 5, vjust = 0.5) +
  facet_grid(rows = vars(sig), scales = "free_y", space = "free_y") +
  scale_fill_brewer(palette = "Set1", name = "Signature") +
  theme_minimal(base_size = 5) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
    axis.text.y = element_text(size = 5),
    strip.text.y = element_text(angle = 0, size = 7),
    legend.position = "right"
  ) +
  labs(
    title = "SNP-Phenotype Association Heatmap by Signature",
    x = "Phenotype",
    y = "SNP"
  )
