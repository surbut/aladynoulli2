
library(reshape2)
library(dplyr)
library(ggplot2)
# Load required libraries
library(ggplot2)
library(reshape2)
library(dplyr)
library(tidyr)
library(forcats)
library(cowplot)

# Read data
ukb_params <- readRDS("ukb_params.rds")
prs_names <- read.csv("prs_names.csv")[,1]

# Create main gamma dataframe
gamma_df <- data.frame(ukb_params$gamma)
names(gamma_df) <- as.character(0:20)
rownames(gamma_df) <- prs_names
gamma_df$prs <- prs_names

# Get PRS-signature significance data if available
# If not available, we'll assume a threshold based on effect size
if(exists("gamma_pvals")) {
  significance <- gamma_pvals < 0.05/nrow(gamma_df)/ncol(gamma_df) # Bonferroni correction
} else {
  # If no p-values, use an effect size threshold (e.g., absolute value > 0.1)
  significance <- abs(as.matrix(gamma_df[,1:21])) > 0.1
}

# Find top associations
gamma_melted <- melt(gamma_df, id.vars="prs", variable.name="signature", value.name="effect")
gamma_melted$signature <- paste0("Sig ", gamma_melted$signature)

# Add disease category grouping to PRSs if known
# This is a placeholder - you should modify based on your knowledge of the disease categories
disease_categories <- data.frame(
  prs = prs_names,
  category = case_when(
    prs_names %in% c("CAD", "AF", "HT", "LDL_SF") ~ "Cardiovascular",
    prs_names %in% c("T1D", "T2D", "BMI", "HBA1C_DF") ~ "Metabolic",
    prs_names %in% c("RA", "PSO", "SLE", "CD", "UC") ~ "Autoimmune",
    prs_names %in% c("AD", "PD", "MS", "BD", "SCZ") ~ "Neurological",
    prs_names %in% c("BC", "PC", "CRC", "MEL") ~ "Cancer",
    TRUE ~ "Other"
  )
)

gamma_melted <- merge(gamma_melted, disease_categories, by="prs")

# Create a version focusing on top effects
top_associations <- gamma_melted %>%
  arrange(desc(abs(effect))) %>%
  head(30)

# Category colors
category_colors <- c(
  "Cardiovascular" = "#E74C3C",
  "Metabolic" = "#2ECC71",
  "Autoimmune" = "#3498DB",
  "Neurological" = "#9B59B6",
  "Cancer" = "#F39C12",
  "Other" = "#95A5A6"
)

# 1. Create bar plot of top associations
p1 <- ggplot(top_associations, aes(x=reorder(paste(prs, "-", signature), abs(effect)), 
                                   y=effect, fill=category)) +
  geom_bar(stat="identity") +
  scale_fill_manual(values=category_colors) +
  coord_flip() +
  labs(title="Top PRS-Signature Associations", 
       x="", 
       y="Effect Size") +
  theme_minimal() +
  theme(
    legend.position="right",
    axis.text.y=element_text(size=10),
    plot.title=element_text(size=14, face="bold")
  )

# 2. Create filtered heatmap of significant associations only
# Filter for more significant associations to make heatmap clearer
significance_threshold <- 0.1
significant_effects <- gamma_melted %>%
  filter(abs(effect) > significance_threshold)

# Convert PRS to factor and reorder based on disease category
significant_effects$prs <- factor(
  significant_effects$prs,
  levels = disease_categories %>% 
    arrange(category) %>% 
    pull(prs)
)

p2 <- ggplot(significant_effects, aes(x=signature, y=prs, fill=effect)) +
  geom_tile() +
  scale_fill_gradient2(low="blue", mid="white", high="red", 
                       midpoint=0, limits=c(-0.3, 0.3)) +
  labs(title="Significant PRS-Signature Associations",
       x="Signature", 
       y="Polygenic Risk Score",
       fill="Effect Size") +
  theme_minimal() +
  theme(
    axis.text.x=element_text(angle=45, hjust=1, size=9),
    axis.text.y=element_text(size=9),
    plot.title=element_text(size=14, face="bold")
  )

# Add category color bar to heatmap
# Create data for category annotation
category_data <- disease_categories %>%
  filter(prs %in% significant_effects$prs) %>%
  distinct(prs, category)

category_data <- category_data[match(levels(significant_effects$prs), category_data$prs),]

# Create category annotation plot
p_category <- ggplot(category_data, aes(x=1, y=prs, fill=category)) +
  geom_tile() +
  scale_fill_manual(values=category_colors) +
  theme_void() +
  theme(legend.position="none")

# 3. Full heatmap with all data (for supplementary)
p_full <- ggplot(gamma_melted, aes(x=signature, y=prs, fill=effect)) +
  geom_tile() +
  scale_fill_gradient2(low="blue", mid="white", high="red", 
                       midpoint=0, limits=c(-0.3, 0.3)) +
  labs(title="Complete PRS-Signature Association Matrix",
       x="Signature", 
       y="Polygenic Risk Score",
       fill="Effect Size") +
  theme_minimal() +
  theme(
    axis.text.x=element_text(angle=45, hjust=1),
    plot.title=element_text(size=14, face="bold")
  )

# Combine plots
# Main figure for paper
main_figure <- plot_grid(
  p1, p2, 
  labels = c("A", "B"),
  ncol = 1,
  rel_heights = c(1, 1.5)
)

# Save plots
ggsave("top_prs_associations.pdf", p1, width=10, height=8)
ggsave("significant_prs_heatmap.pdf", p2, width=12, height=10)
ggsave("complete_prs_heatmap.pdf", p_full, width=12, height=10)
ggsave("main_figure_prs_signatures.pdf", main_figure, width=12, height=14)

# Display plots
print(p1)
print(p2)
print(p_full)
print(main_figure)