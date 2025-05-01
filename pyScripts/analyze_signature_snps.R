library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stats)
library(logistf)  # For more stable logistic regression
library(pheatmap)

analyze_signature_snp_associations <- function(
  genotype_file_default = "/Users/sarahurbut/Dropbox/genotype_raw/genotype_dosage_20250416.raw",
  genotype_file_sig5 = "/Users/sarahurbut/Dropbox/genotype_raw/genotype_dosage_20250415.raw",
  covariate_file = "/Users/sarahurbut/Dropbox/for_regenie/ukbb_covariates_400k.txt",
  snp_list_dir = "/Users/sarahurbut/Dropbox/snp_lists",
  phenotype_dir = "/Users/sarahurbut/Dropbox/for_regenie/case_control_phenotypes",
  sig_stats_dir = "/Users/sarahurbut/Dropbox/result326/10_loci",
  signatures_to_analyze = c(4),
  output_dir = NULL
) {
  
  # Initialize results list
  results_list <- list()
  
  for (sig in signatures_to_analyze) {
    message(sprintf("\nAnalyzing Signature %d", sig))
    
    # Read signature summary statistics
    sig_stats_file <- file.path(sig_stats_dir, 
                               sprintf("SIG%d_AUC_ukb_eur_regenie_af1.sig.lead.sumstats.txt", sig))
    
    tryCatch({
      # Read and process signature statistics
      sig_stats <- fread(sig_stats_file)
      message(sprintf("Loaded signature statistics with %d rows", nrow(sig_stats)))
      
      # Create SNP to Z-stat mapping
      sig_z_stats <- setNames(sig_stats[[17]], sig_stats[[13]])  # Adjust column indices as needed
      message(sprintf("Created Z-stat mapping for %d SNPs", length(sig_z_stats)))
      
    }, error = function(e) {
      message(sprintf("Error reading signature statistics: %s", e$message))
      return(NULL)
    })
    
    # Select appropriate genotype file
    genotype_file <- if(sig == 5) genotype_file_sig5 else genotype_file_default
    message(sprintf("Using genotype file: %s", genotype_file))
    
    # Read significant SNPs
    snp_file <- file.path(snp_list_dir, sprintf("snp_list_sig%d.txt", sig))
    tryCatch({
      sig_snps <- fread(snp_file, col.names = c("idx", "rsid"))$rsid
      message(sprintf("Found %d significant SNPs", length(sig_snps)))
    }, error = function(e) {
      message(sprintf("Error reading SNP file: %s", e$message))
      return(NULL)
    })
    
    # Read phenotypes
    pheno_file <- file.path(phenotype_dir, sprintf("case_control_sig%d.tsv", sig))
    tryCatch({
      phenotypes <- fread(pheno_file)
      message(sprintf("Loaded phenotypes with dimensions %d x %d", 
                     nrow(phenotypes), ncol(phenotypes)))
    }, error = function(e) {
      message(sprintf("Error reading phenotype file: %s", e$message))
      return(NULL)
    })
    
    # Read genotypes
    tryCatch({
      genotypes <- fread(genotype_file)
      
      # Process column names to match SNPs
      geno_cols <- colnames(genotypes)
      snp_cols <- geno_cols[!geno_cols %in% c("FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE")]
      
      # Create column mapping
      col_to_snp <- sapply(snp_cols, function(x) strsplit(x, "_")[[1]][1])
      
      # Find matching columns for significant SNPs
      snp_to_col <- sapply(sig_snps, function(snp) {
        matching_cols <- names(col_to_snp)[col_to_snp == snp]
        if(length(matching_cols) > 0) matching_cols[1] else NA
      })
      
      # Remove NA mappings
      snp_to_col <- snp_to_col[!is.na(snp_to_col)]
      
      if(length(snp_to_col) == 0) {
        message("No matching SNPs found in genotype file")
        return(NULL)
      }
      
      # Select and rename columns
      selected_cols <- c("FID", unname(snp_to_col))
      genotypes <- genotypes[, ..selected_cols]  # Using data.table syntax
      setnames(genotypes, unname(snp_to_col), names(snp_to_col))
      
    }, error = function(e) {
      message(sprintf("Error reading genotype file: %s", e$message))
      return(NULL)
    })
    
    # Read covariates
    tryCatch({
      covariates <- fread(covariate_file)
      pc_cols <- grep("^PC", colnames(covariates), value = TRUE)
      covariates <- covariates[, c("identifier", "sex", pc_cols), with = FALSE]
      setnames(covariates, "identifier", "FID")
    }, error = function(e) {
      message(sprintf("Error processing covariates: %s", e$message))
      return(NULL)
    })
    
    # Merge data
    merged_data <- Reduce(function(x, y) merge(x, y, by = "FID"), 
                         list(genotypes, covariates, phenotypes))
    
    # Initialize results storage
    snp_results <- data.frame()
    
    # Analyze each SNP
    for(snp in names(snp_to_col)) {
      message(sprintf("Analyzing %s", snp))
      
      pheno_cols <- setdiff(colnames(phenotypes), "FID")
      
      # Test SNP against each phenotype
      for(pheno in pheno_cols) {
        # Prepare data for model
        check_cols <- c(snp, pheno, "sex", pc_cols)
        model_data <- merged_data[complete.cases(merged_data[, ..check_cols]), ]
        
        if(nrow(model_data) < 10) next
        
        tryCatch({
          # Fit logistic regression using logistf for better stability
          formula_str <- sprintf("%s ~ %s + sex + %s", 
                               pheno, snp, paste(pc_cols, collapse = " + "))
          
          model <- glm(formula = as.formula(formula_str),data = model_data,family="binomial")
          
          # Extract results
          beta <- coef(model)[snp]
          z_stat <- beta / sqrt(diag(vcov(model)))[snp]
          p_val <- model$prob[snp]
          
          # Add to results
          snp_results <- rbind(snp_results, data.frame(
            SNP = snp,
            Phenotype = pheno,
            Beta = beta,
            Z_statistic = z_stat,
            P_value = p_val,
            N = nrow(model_data)
          ))
          
        }, error = function(e) {
          message(sprintf("Error in regression for %s-%s: %s", snp, pheno, e$message))
        })
      }
    }
    
    if(nrow(snp_results) == 0) {
      message(sprintf("No results generated for signature %d", sig))
      next
    }
    
    # Add signature Z-statistics
    snp_results$Signature_Z <- sig_z_stats[snp_results$SNP]
    
    # Find interesting SNPs
    sig_threshold <- 5e-8
    interesting_snps <- snp_results %>%
      group_by(SNP) %>%
      summarise(
        all_nonsig = all(P_value > sig_threshold),
        mean_abs_z = mean(abs(Z_statistic)),
        Signature_Z = first(Signature_Z),
        N = mean(N)
      ) %>%
      filter(all_nonsig) %>%
      arrange(desc(mean_abs_z))
    
    # Store results
    results_list[[as.character(sig)]] <- list(
      full_results = snp_results,
      interesting_snps = interesting_snps
    )
    
    # Create visualization
    if(nrow(interesting_snps) > 0) {
      # Get top SNPs
      top_snps <- head(interesting_snps$SNP, 10)
      plot_data <- snp_results %>%
        filter(SNP %in% top_snps) %>%
        select(SNP, Phenotype, Z_statistic, Signature_Z)
      
      # Create matrix for heatmap
      heatmap_data <- plot_data %>%
        pivot_wider(
          names_from = Phenotype,
          values_from = Z_statistic,
          id_cols = c(SNP, Signature_Z)
        ) %>%
        as.data.frame()
      
      rownames(heatmap_data) <- heatmap_data$SNP
      heatmap_data$SNP <- NULL
      sig_z <- heatmap_data$Signature_Z
      heatmap_data$Signature_Z <- NULL
      
      # Create heatmap
      if(!is.null(output_dir)) {
        pdf(file.path(output_dir, sprintf("signature_%d_snp_heatmap.pdf", sig)),
            width = 12, height = 8)
      }
      
      pheatmap(
        heatmap_data,
        main = sprintf("Signature %d: SNPs with No Individual Disease Associations", sig),
        scale = "none",
        cluster_rows = FALSE,
        cluster_cols = FALSE,
        display_numbers = TRUE,
        number_format = "%.2f",
        fontsize_number = 8
      )
      
      if(!is.null(output_dir)) dev.off()
      
      # Print summary
      message("\nTop SNPs summary:")
      print(head(interesting_snps, 10))
    }
  }
  
  return(results_list)
} 



# Load required packages
library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stats)
library(logistf)
library(pheatmap)

# Run the analysis
results <- analyze_signature_snp_associations(
  signatures_to_analyze = c(5),
  output_dir = "output"
)


# Load required libraries
library(UpSetR)
library(dplyr)
library(tidyr)
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

# Read in all the data files
angina <- read_gwas_data("Angina_pectoris_ukb_eur_regenie_af1.sig.lead.sumstats.txt")
cor_athero <- read_gwas_data("Coronary_atherosclerosis_ukb_eur_regenie_af1.sig.lead.sumstats.txt")
hyperchol <- read_gwas_data("Hypercholesterolemia_ukb_eur_regenie_af1.sig.lead.sumstats.txt")
mi <- read_gwas_data("Myocardial_infarction_ukb_eur_regenie_af1.sig.lead.sumstats.txt")
acute_ihd <- read_gwas_data("Other_acute_and_subacute_forms_of_ischemic_heart_disease_ukb_eur_regenie_af1.sig.lead.sumstats.txt")
chronic_ihd <- read_gwas_data("Other_chronic_ischemic_heart_disease,_unspecified_ukb_eur_regenie_af1.sig.lead.sumstats.txt")
sig5 <- read_gwas_data("~/Dropbox/result326/10_loci/SIG5_AUC_ukb_eur_regenie_af1.sig.lead.sumstats.txt")

# Add source information
angina$source <- "Angina"
cor_athero$source <- "Cor_Athero"
hyperchol$source <- "Hypercholest"
mi$source <- "MI"
acute_ihd$source <- "Acute_IHD"
chronic_ihd$source <- "Chronic_IHD"
sig5$source <- "SIG5_AUC"

# Create a list of variant UIDs for each phenotype
angina_variants <- angina$rsid
cor_athero_variants <- cor_athero$rsid
hyperchol_variants <- hyperchol$rsid
mi_variants <- mi$rsid
acute_ihd_variants <- acute_ihd$rsid
chronic_ihd_variants <- chronic_ihd$rsid
sig5_variants <- sig5$rsid

# Get all unique variants
all_variants <- unique(c(
  angina_variants, cor_athero_variants, hyperchol_variants, 
  mi_variants, acute_ihd_variants, chronic_ihd_variants, 
  sig5_variants
))

# Create a presence/absence matrix
presence_matrix <- data.frame(
  UID = all_variants,
  Angina = all_variants %in% angina_variants,
  Cor_Athero = all_variants %in% cor_athero_variants,
  Hypercholest = all_variants %in% hyperchol_variants,
  MI = all_variants %in% mi_variants,
  Acute_IHD = all_variants %in% acute_ihd_variants,
  Chronic_IHD = all_variants %in% chronic_ihd_variants,
  SIG5_AUC = all_variants %in% sig5_variants
)

# Convert to binary matrix for UpSetR
upset_matrix <- presence_matrix %>%
  select(-UID) %>%
  mutate_all(as.numeric)

# Get set sizes for labeling
set_sizes <- colSums(upset_matrix)
set_labels <- paste0(
  names(set_sizes), 
  " (", set_sizes, ")"
)

# Plot upset plot with UpSetR
upset_plot <- upset(
  upset_matrix,
  nsets = 7,
  order.by = "freq",
  sets = names(upset_matrix),
  keep.order = TRUE,
  set.metadata = list(
    data = data.frame(
      sets = names(set_sizes),
      size = set_sizes
    ),
    plots = list(
      list(
        type = "hist",
        column = "size",
        assign = 20
      )
    )
  ),
  mainbar.y.label = "Intersection Size",
  sets.x.label = "Variants Per Phenotype",
  text.scale = 1.2,
  point.size = 3,
  line.size = 1,
  mb.ratio = c(0.6, 0.4),
  set_size.show = TRUE
)

# Print the upset plot
print(upset_plot)

# Save the plot
pdf("cardiovascular_variants_upset_plot.pdf", width = 12, height = 8)
print(upset_plot)
dev.off()

# Additional analysis: Count of variants in each category
# Define categories
categorized_variants <- presence_matrix %>%
  mutate(
    category = case_when(
      SIG5_AUC & (Angina | Cor_Athero | Hypercholest | MI | Acute_IHD | Chronic_IHD) ~ "shared",
      SIG5_AUC ~ "sig5_specific",
      TRUE ~ "trait_specific"
    )
  )

# Count variants in each category
category_counts <- categorized_variants %>%
  count(category)


# Approach using text color intensity for significance
plot_data <- all_data %>%
  select(rsid, source, BETA, log10P) %>%
  mutate(
    # Normalize log10P for text color intensity
    # Higher values = more significant = darker text
    p_intensity = pmin(log10P / 15, 1)  # Cap at 1 (P ~ 1e-15)
  )

# Create the plot
ggplot(plot_data, aes(x = source, y = rsid, fill = BETA)) +
  geom_tile(color = "white", size = 0.2) +
  # Use p_intensity to control text color darkness
  geom_text(aes(label = "",
                color = p_intensity), 
            size = 3) +
  scale_fill_gradient2(
    low = "navy", 
    mid = "white", 
    high = "firebrick3",
    midpoint = 0, 
    name = "Effect Size (Beta)",
    na.value = "gray95"
  ) +
  # Scale for text color (from light gray to black based on significance)
  scale_color_gradient(
    low = "gray70",
    high = "black",
    name = "-log10(P)",
    guide = "legend"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 8),
    axis.title = element_text(size = 12),
    panel.grid = element_blank()
  ) +
  labs(
    title = "Effect Sizes Across Traits by Variant: GWAS significant per trait",
    subtitle = "Darker text indicates stronger statistical significance",
    x = "Trait",
    y = "Variant"
  )

# Print category counts
print(category_counts)

# Calculate percentages
category_counts <- category_counts %>%
  mutate(percentage = n / sum(n) * 100)

# Print category counts with percentages
print(category_counts)

# Variant overlap between individual phenotypes
# Define a function to calculate Jaccard similarity
jaccard_similarity <- function(set1, set2) {
  intersection_size <- sum(set1 & set2)
  union_size <- sum(set1 | set2)
  return(intersection_size / union_size)
}

# Create a matrix to store the similarities
phenotypes <- names(upset_matrix)
similarity_matrix <- matrix(0, nrow = length(phenotypes), ncol = length(phenotypes))
rownames(similarity_matrix) <- phenotypes
colnames(similarity_matrix) <- phenotypes

# Fill the matrix with Jaccard similarities
for (i in 1:length(phenotypes)) {
  for (j in 1:length(phenotypes)) {
    set1 <- upset_matrix[, i]
    set2 <- upset_matrix[, j]
    similarity_matrix[i, j] <- jaccard_similarity(set1, set2)
  }
}

# Print the similarity matrix
print(similarity_matrix)

# For nicer output, convert to a dataframe
similarity_df <- as.data.frame(similarity_matrix)
similarity_df$phenotype1 <- rownames(similarity_df)
similarity_df_long <- similarity_df %>%
  pivot_longer(
    cols = -phenotype1,
    names_to = "phenotype2",
    values_to = "jaccard_similarity"
  ) %>%
  filter(phenotype1 != phenotype2) %>%
  arrange(desc(jaccard_similarity))

# Print the sorted similarities
print(similarity_df_long)

# Analyze effect consistency for shared variants
# Combine all data
all_data <- bind_rows(angina, cor_athero, hyperchol, mi, acute_ihd, chronic_ihd, sig5)

# Filter for shared variants
shared_variants <- categorized_variants %>%
  filter(category == "shared") %>%
  pull(UID)

# Get effect directions for shared variants
effect_directions <- all_data %>%
  filter(rsid %in% shared_variants) %>%
  select(rsid, source, BETA) %>%
  mutate(direction = ifelse(BETA > 0, "positive", "negative"))

# Check consistency for each variant
effect_consistency <- effect_directions %>%
  group_by(rsid) %>%
  summarize(
    consistent = length(unique(direction)) == 1,
    direction = first(direction[source == "SIG5_AUC"]),
    n_phenotypes = n_distinct(source)
  )

# Count consistent vs inconsistent
consistency_summary <- effect_consistency %>%
  count(consistent) %>%
  mutate(percentage = n / sum(n) * 100)

# Print consistency summary
print(consistency_summary)

# Load required libraries
library(tidyverse)
library(pheatmap)

# Assuming you have already run your code to create all_data and shared_variants

# Create a heatmap of Z-statistics with significance markers
# -----------------------------------------------------------


# Load required libraries
library(tidyverse)
library(pheatmap)

# Prepare data with proper handling of NA values
# ---------------------------------------------

all_data$log10P=all_data$LOG10P


# Super simple ggplot2 approach - most robust solution
# ---------------------------------------------------
library(tidyverse)

# Start with your data in long format (which should work without pivoting issues)
plot_data <- all_data %>%
  select(rsid, source, BETA, log10P) %>%
  mutate(
    significant = log10P > 7.3,
    # Create the label with value and asterisk
    label = ifelse(is.na(BETA) | is.infinite(BETA), 
                   "", 
                   ifelse(significant, 
                          "*", 
                          "")
  ))

# Create the plot
ggplot(plot_data, aes(x = source, y = rsid, fill = BETA)) +
  geom_tile(color = "white", size = 0.2) +
  # Add the pre-formatted labels
  geom_text(aes(label = label), size = 3) +
  # Use diverging color scale
  scale_fill_gradient2(
    low = "navy", 
    mid = "white", 
    high = "firebrick3",
    midpoint = 0, 
    name = "Effect Size (Beta)",
    na.value = "gray95"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 8),
    axis.title = element_text(size = 12),
    panel.grid = element_blank(),
    legend.position = "right"
  ) +
  labs(
    title = "Effect Sizes Across Traits by Variant",
    x = "Trait",
    y = "Variant"
  )

# Approach using text color intensity for significance
plot_data <- all_data %>%
  select(rsid, source, BETA,SE, log10P) %>%
  mutate(
    # Normalize log10P for text color intensity
    # Higher values = more significant = darker text
    p_intensity = pmin(log10P / 15, 1)  # Cap at 1 (P ~ 1e-15)
  )

# Create the plot
ggsave(ggplot(plot_data, aes(x = source, y = rsid, fill = BETA/SE)) +
  geom_tile(color = "white", size = 0.2) +
  # Use p_intensity to control text color darkness
  geom_text(aes(label = "",
                color = p_intensity), 
            size = 3) +
  scale_fill_gradient2(
    low = "navy", 
    mid = "white", 
    high = "firebrick3",
    midpoint = 0, 
    name = "Effect Size (Beta)",
    na.value = "gray95"
  ) +
  # Scale for text color (from light gray to black based on significance)
  scale_color_gradient(
    low = "gray70",
    high = "black",
    name = "-log10(P)",
    guide = "legend"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 8),
    axis.title = element_text(size = 12),
    panel.grid = element_blank()
  ) +
  labs(
    title = "Effect Sizes Across Traits by Variant",
    subtitle = "Darker text indicates stronger statistical significance",
    x = "Trait",
    y = "Variant"
  ),file="~/aladynoulli2/gwassigtraits.pdf",dpi = 300)