library(data.table)
setwd("~/aladynoulli2/pyScripts/")


cox_aladyn=fread("model_comparison_results_cox_aladyn.csv")
#static_auc=fread("model_comparison_results_bootstatic_auc.csv")
static_c=fread("model_comparison_results_cindex.csv")
dynamic=fread("model_comparison_results_dynami.csv")

m=merge(cox_aladyn,static_auc[,c("Disease","Aladynoulli_AUC")],by="Disease")
m=merge(m,static_c[,c("Disease","Cox_Concordance")],by="Disease")
m=merge(m,dynamic[,c("Disease","Aladynoulli_AUC")],by="Disease")
names(m)[3]="Cox_AUC"
names(m)[5]="Aladynoulli_static"
names(m)[9]="Aladynoulli_dynamic"


library(data.table)
library(ggplot2)
library(tidyr)
library(dplyr)
# If not already, convert to data.table
setDT(m)

# Extract numeric AUCs from the character columns (e.g., "0.712 (0.693-0.724)")
extract_auc <- function(x) as.numeric(sub(" .*", "", x))

m_long <- m %>%
  mutate(
    Cox_AUC_num = extract_auc(Cox_AUC),
    Static_Aladynoulli_AUC = extract_auc(Aladynoulli_AUC.x),
    Dynamic_Aladynoulli_AUC = extract_auc(Aladynoulli_AUC.y)
  ) %>%
  select(Disease, Cox_AUC_num, Static_Aladynoulli_AUC, Dynamic_Aladynoulli_AUC) %>%
  pivot_longer(
    cols = c(Cox_AUC_num, Static_Aladynoulli_AUC, Dynamic_Aladynoulli_AUC),
    names_to = "Model",
    values_to = "AUC"
  )

# Optional: order diseases by dynamic Aladynoulli AUC
disease_order <- m_long %>%
  filter(Model == "Dynamic_Aladynoulli_AUC") %>%
  arrange(desc(AUC)) %>%
  pull(Disease)

m_long$Disease <- factor(m_long$Disease, levels = disease_order)

# Plot
ggplot(m_long, aes(x = Disease, y = AUC, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  coord_flip() +
  scale_fill_manual(
    values = c(
      "Cox_AUC_num" = "#1b9e77",
      "Static_Aladynoulli_AUC" = "#7570b3",
      "Dynamic_Aladynoulli_AUC" = "#d95f02"
    ),
    labels = c(
      "Cox_AUC_num" = "Cox (10-year AUC)",
      "Static_Aladynoulli_AUC" = "Static Aladynoulli",
      "Dynamic_Aladynoulli_AUC" = "Dynamic Aladynoulli"
    )
  ) +
  labs(
    title = "10-year AUC by Disease and Model",
    x = "Disease",
    y = "AUC",
    fill = "Model"
  ) +
  theme_minimal(base_size = 14)


# Extract numeric C-index
m$Cox_Concordance_num <- extract_auc(m$Cox_Concordance)

# Add to long format for plotting
m_cindex <- m %>%
  select(Disease, Cox_Concordance_num) %>%
  mutate(Model = "Cox_Concordance", AUC = Cox_Concordance_num)

# Combine with AUC data
m_long2 <- bind_rows(
  m_long,
  m_cindex %>% select(Disease, Model, AUC)
)

# Plot: bars for AUC, points for C-index
ggplot(m_long2, aes(x = Disease, y = AUC, fill = Model)) +
  geom_bar(
    data = filter(m_long2, Model != "Cox_Concordance"),
    stat = "identity", position = position_dodge(width = 0.8)
  ) +
  geom_point(
    data = filter(m_long2, Model == "Cox_Concordance"),
    aes(color = Model), size = 3, position = position_nudge(x = 0.3)
  ) +
  coord_flip() +
  scale_fill_manual(
    values = c(
      "Cox_AUC_num" = "#1b9e77",
      "Static_Aladynoulli_AUC" = "#7570b3",
      "Dynamic_Aladynoulli_AUC" = "#d95f02"
    ),
    labels = c(
      "Cox_AUC_num" = "Cox (10-year AUC)",
      "Static_Aladynoulli_AUC" = "Static Aladynoulli",
      "Dynamic_Aladynoulli_AUC" = "Dynamic Aladynoulli"
    )
  ) +
  scale_color_manual(
    values = c("Cox_Concordance" = "black"),
    labels = c("Cox_Concordance" = "Cox C-index")
  ) +
  labs(
    title = "10-year AUC and C-index by Disease and Model",
    x = "Disease",
    y = "Metric Value",
    fill = "Model",
    color = "Metric"
  ) +
  theme_minimal(base_size = 14)