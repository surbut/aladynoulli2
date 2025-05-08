# psi_matrix: K x D
#ukb_params=readRDS("big_stuff/ukb_params.rds")
psi_mat=ukb_params$psi

n_signatures=21
names(signature_colors) <- as.character(1:n_signatures)

primary_signature <- apply(psi_mat, 2, which.max)  # length D, 1-based
num_diag <- rowSums(E != 51)
table(num_diag)  # See distribution

# Pick a patient with 5-15 diagnoses
candidate_patients <- which(num_diag >= 10 & num_diag <= 20)
set.seed(1)
chosen_patient <- sample(candidate_patients, 1)
patient_theta <- all_thetas_array[chosen_patient,,]
patient_diagnoses <- E[chosen_patient,]
diagnosed <- which(patient_diagnoses != 51)
diagnosis_times <- patient_diagnoses[diagnosed]+29
diagnosis_names <- disease_names[diagnosed]
diagnosis_sigs <- primary_signature[diagnosed]


library(ggplot2)
library(tidyr)
library(dplyr)
library(gridExtra)

# Color palette for signatures
signature_colors <- brewerPlus

# Prepare theta data
time_points <- 30:81
theta_df <- as.data.frame(patient_theta)
colnames(theta_df) <- time_points
theta_df$signature <- 1:nrow(patient_theta)
theta_long <- pivot_longer(theta_df, cols = -signature, names_to = "age", values_to = "loading")
theta_long$age <- as.numeric(theta_long$age)

# Main trajectory plot
p1 <- ggplot(theta_long, aes(x = age, y = loading, color = factor(signature))) +
  geom_line(size = 1) +
  scale_color_manual(values = signature_colors) +
  geom_vline(xintercept = diagnosis_times, linetype = "dashed", alpha = 0.5) +
  theme_minimal() +
  labs(title = "Signature Loadings Over Time",
       x = "Age",
       y = "Signature Loading (θ)",
       color = "Signature") +
  theme(legend.position = "right")

# Timeline plot with colored points and disease names
timeline_df <- data.frame(
  disease = diagnosis_names,
  diagnosis_time = diagnosis_times,
  y_pos = seq_along(diagnosis_names),
  sig = diagnosis_sigs
)

p2 <- ggplot(timeline_df, aes(x = diagnosis_time, y = y_pos)) +
  geom_point(aes(color = factor(sig)), size = 4) +
  geom_segment(aes(x = min(time_points), xend = diagnosis_time, yend = y_pos, color = factor(sig)), alpha = 0.5) +
  scale_y_reverse(breaks = seq_along(diagnosis_names), labels = diagnosis_names) +
  scale_color_manual(values = signature_colors, name = "Primary Signature",drop=FALSE) +
  theme_minimal() +
  labs(title = "Disease Timeline",
       x = "Age",
       y = "Diagnosed Condition") +
  theme(axis.text.y = element_text(size = 8), legend.position = "right")

# Combine plots
grid.arrange(p1, p2, ncol = 1, heights = c(2, 1))