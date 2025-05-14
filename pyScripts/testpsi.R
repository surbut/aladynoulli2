# psi_matrix: K x D
ukb_params=readRDS("big_stuff/ukb_params.rds")
ukb_checkpoint=readRDS("ukb_model.rds")
psi_mat=ukb_params$psi
E_full=read.csv("E_full_first10k.csv")
pce_data=readRDS("~/Dropbox/pce_df_prevent.rds")
n_signatures=21
library(microViz)
brewerPlus <- distinct_palette()
# Color palette for signatures
signature_colors <- brewerPlus
names(signature_colors) <- as.character(0:20)
primary_signature <- apply(psi_mat, 2, which.max) - 1  # length D, 0-based

#### to do for all times with model fir using all times 

all_thetas_array <- readRDS("all_thetas_array_time.rds")
E_full=readRDS("E_full_tensor.rds")
E_mi=E_full[,113]
num_diag <- rowSums(E_full != 51)
table(num_diag)  # See distribution

for(i in 1:10){
# Pick a patient with 5-15 diagnoses
candidate_patients <- which(num_diag >= 10 & num_diag <= 20&E_mi!=51)
#set.seed(1)
chosen_patient <- sample(candidate_patients, 1)
#chosen_patient=176919 #someone with high loading on having nothing 
#chosen_patient=74021
#chosen_patien=357674
patient_theta <- all_thetas_array[chosen_patient,,]
patient_diagnoses <- E_full[chosen_patient,]
diagnosed <- which(patient_diagnoses != 51)
diagnosis_times <- as.numeric(patient_diagnoses[diagnosed]+29)

####

#######
disease_names=ukb_checkpoint$disease_names[,1]
diagnosis_names <- disease_names[diagnosed]
diagnosis_sigs <- primary_signature[diagnosed]

library(ggplot2)
library(tidyr)
library(dplyr)
library(patchwork)

# --- Data Preparation ---
# (Assume you have: patient_theta, diagnosis_times, diagnosis_names, diagnosis_sigs, time_points, signature_colors)
time_points=30:81
n_signatures <- 21
signature_labels <- as.character(0:20) # 0-based
names(signature_colors) <- signature_labels

# Prepare theta_long for the top plot
theta_df <- as.data.frame(patient_theta)
#pop_refs=apply(all_thetas_array,c(2,3),mean)
#theta_df=theta_df/pop_refs
colnames(theta_df) <- time_points
theta_df$signature <- factor(0:(nrow(patient_theta)-1), levels = signature_labels)  # 0-based
theta_long <- pivot_longer(theta_df, cols = -signature, names_to = "age", values_to = "loading")
theta_long$age <- as.numeric(theta_long$age)

# Calculate average loadings across all times
avg_loadings <- theta_long %>%
  group_by(signature) %>%
  summarise(avg_loading = mean(loading))

# Prepare timeline_df for the bottom plot, sorted by diagnosis time
timeline_df <- data.frame(
  disease = diagnosis_names,
  diagnosis_time = diagnosis_times,
  sig = factor(diagnosis_sigs, levels = signature_labels)  # 0-based
) %>%
  arrange(diagnosis_time) %>%
  mutate(y_pos = seq_along(diagnosis_time))

# Set x-axis limits for both plots
x_min <- min(time_points)
x_max <- max(time_points)
label_offset <- 5

# Create a layout with three panels
layout <- "
AAB
CCB
"

enrollment_time=pce_data[chosen_patient,age]

# Top plot: Signature loadings
p1 <- ggplot(theta_long, aes(x = age, y = loading, color = signature)) +
  geom_line(size = 1) +
  scale_color_manual(values = signature_colors, drop = TRUE) +
  geom_vline(xintercept = diagnosis_times, linetype = "dashed", alpha = 0.5) +
  geom_vline(xintercept = enrollment_time, linetype = "solid", alpha = 0.5) +
  theme_minimal(base_size = 14) +
  labs(title = paste0("Signature Loadings Over Time for pt",chosen_patient),
       x = NULL,
       y = "Signature Loading (θ)",
       color = "Signature") +
  theme(legend.position = "right",
        axis.title.x = element_blank(),
        plot.margin = margin(5, 5, 0, 5)) +
  xlim(x_min, x_max + label_offset + 5)

# Timeline plot: lollipops with right-aligned labels
p2 <- ggplot(timeline_df, aes(x = diagnosis_time, y = y_pos)) +
  geom_segment(aes(x = x_min, xend = diagnosis_time, yend = y_pos, color = sig), alpha = 0.5) +
  geom_point(aes(color = sig), size = 4) +
  geom_text(aes(x = x_max + label_offset, label = disease), hjust = 0, size = 2) +
  scale_color_manual(values = signature_colors, drop = TRUE) +
  scale_y_reverse(NULL, breaks = NULL) +
  theme_minimal(base_size = 14) +
  labs(title = "Disease Timeline",
       x = "Age",
       y = NULL) +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.position = "none",
        plot.margin = margin(0, 120, 5, 5)) +
  xlim(x_min, x_max + label_offset + 5)

# Static summary plot (right side)
p3 <- ggplot(avg_loadings, aes(x = 0, y = avg_loading, fill = signature)) +
  geom_bar(stat = "identity", width = 0.3) +
  scale_fill_manual(values = signature_colors, drop = TRUE) +
  theme_minimal(base_size = 14) +
  labs(title = "Static Model\nSummary",
       x = NULL,
       y = "Average Loading (θ)") +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        legend.position = "none",
        plot.margin = margin(5, 5, 5, 5)) +
  coord_cartesian(xlim = c(-0.3, 0.3))

# Combine with patchwork for desired layout
a=(p1 / p2) | p3 +
  plot_layout(widths = c(5, 1), heights = c(2, 1), guides = "collect")

ggsave(plot = a,filename = paste0("longpattraj_withenrollment_",chosen_patient,".pdf"),dpi = 300,width = 25,height = 10)
}


#### do with enrollment info
ukb_params_enroll=readRDS("big_stuff/ukb_params_enrollment.rds")
E_enroll=read.csv("E_enrollment_first10k.csv")
ukb_checkpoint <- readRDS("~/Dropbox/ukb_model.rds")
pce_data = readRDS('/Users/sarahurbut/Dropbox/pce_df_prevent.rds')
brewerPlus <- distinct_palette()
scales::show_col(brewerPlus)

softmax_by_k <- function(x) {
  # Apply softmax along K dimension (dimension 2)
  exp_x <- exp(x)
  sweep(exp_x, c(1,3), apply(exp_x, c(1,3), sum), "/")
}

all_thetas <- softmax_by_k(ukb_params_enroll$lambda)


num_diag=sapply(seq(1:nrow(E_enroll)),function(x){
  sum(E_enroll[x,]!=pce_data[x,age-30])
  }
  )

num_diag_after=sapply(seq(1:nrow(E_full)),function(x){
  sum(E_full[x,]>pce_data[x,age-30]&E_full[x,]!=51)
}
)

table(num_diag)  # See distribution
candidate_patients <- which(num_diag >= 5 & abs(num_diag_after-num_diag)/num_diag<2)
#set.seed(1)

chosen_patient <- sample(candidate_patients, 1)
print(num_diag[chosen_patient])
print(num_diag_after[chosen_patient])
patient_theta <- all_thetas[chosen_patient,,]

patient_diagnoses <- E_full[chosen_patient,]
diagnosed <- which(patient_diagnoses != 51)
diagnosis_times <- as.numeric(patient_diagnoses[diagnosed]+29)
pce_data[chosen_patient,age]
sum(diagnosis_times<pce_data[chosen_patient,age])
sum(diagnosis_times>pce_data[chosen_patient,age]&diagnosis_times!=51)
sum(diagnosis_times!=51)
#######



disease_names=ukb_checkpoint$disease_names[,1]
diagnosis_names <- disease_names[diagnosed]
diagnosis_sigs <- primary_signature[diagnosed]

library(ggplot2)
library(tidyr)
library(dplyr)
library(patchwork)

# --- Data Preparation ---
# (Assume you have: patient_theta, diagnosis_times, diagnosis_names, diagnosis_sigs, time_points, signature_colors)
time_points=30:81
n_signatures <- 21
signature_labels <- as.character(0:20) # 0-based
names(signature_colors) <- signature_labels

# Prepare theta_long for the top plot
theta_df <- as.data.frame(patient_theta)
colnames(theta_df) <- time_points
theta_df$signature <- factor(0:(nrow(patient_theta)-1), levels = signature_labels)  # 0-based
theta_long <- pivot_longer(theta_df, cols = -signature, names_to = "age", values_to = "loading")
theta_long$age <- as.numeric(theta_long$age)

# Calculate average loadings across all times
avg_loadings <- theta_long %>%
  group_by(signature) %>%
  summarise(avg_loading = mean(loading))

# Prepare timeline_df for the bottom plot, sorted by diagnosis time
timeline_df <- data.frame(
  disease = diagnosis_names,
  diagnosis_time = diagnosis_times,
  sig = factor(diagnosis_sigs, levels = signature_labels)  # 0-based
) %>%
  arrange(diagnosis_time) %>%
  mutate(y_pos = seq_along(diagnosis_time))

# Set x-axis limits for both plots
x_min <- min(time_points)
x_max <- max(time_points)
label_offset <- 5

# Create a layout with three panels
layout <- "
AAB
CCB
"

enrollment_time=pce_data[chosen_patient,age]

# Top plot: Signature loadings
p1 <- ggplot(theta_long, aes(x = age, y = loading, color = signature)) +
  geom_line(size = 1) +
  scale_color_manual(values = signature_colors, drop = TRUE) +
  geom_vline(xintercept = diagnosis_times, linetype = "dashed", alpha = 0.5) +
  geom_vline(xintercept = enrollment_time, linetype = "solid", alpha = 0.5) +
  theme_minimal(base_size = 14) +
  labs(title = paste0("Signature Loadings Over Time for pt",chosen_patient),
       x = NULL,
       y = "Signature Loading (θ)",
       color = "Signature") +
  theme(legend.position = "right",
        axis.title.x = element_blank(),
        plot.margin = margin(5, 5, 0, 5)) +
  xlim(x_min, x_max + label_offset + 5)

# Timeline plot: lollipops with right-aligned labels
p2 <- ggplot(timeline_df, aes(x = diagnosis_time, y = y_pos)) +
  geom_segment(aes(x = x_min, xend = diagnosis_time, yend = y_pos, color = sig), alpha = 0.5) +
  geom_point(aes(color = sig), size = 4) +
  geom_text(aes(x = x_max + label_offset, label = disease), hjust = 0, size = 2) +
  scale_color_manual(values = signature_colors, drop = TRUE) +
  scale_y_reverse(NULL, breaks = NULL) +
  theme_minimal(base_size = 14) +
  labs(title = "Disease Timeline",
       x = "Age",
       y = NULL) +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.position = "none",
        plot.margin = margin(0, 120, 5, 5)) +
  xlim(x_min, x_max + label_offset + 5)

# Static summary plot (right side)
p3 <- ggplot(avg_loadings, aes(x = 0, y = avg_loading, fill = signature)) +
  geom_bar(stat = "identity", width = 0.3) +
  scale_fill_manual(values = signature_colors, drop = TRUE) +
  theme_minimal(base_size = 14) +
  labs(title = "Static Model\nSummary",
       x = NULL,
       y = "Average Loading (θ)") +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        legend.position = "none",
        plot.margin = margin(5, 5, 5, 5)) +
  coord_cartesian(xlim = c(-0.3, 0.3))

# Combine with patchwork for desired layout
a=(p1 / p2) | p3 +
  plot_layout(widths = c(5, 1), heights = c(2, 1), guides = "collect")

ggsave(plot = a,filename = "longpattraj_withenrollment_censor9243.pdf",dpi = 300,width = 25,height = 10)