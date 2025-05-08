## disease contribution

ukb_params=readRDS("big_stuff/ukb_params.rds")
ukb_checkpoint <- readRDS("~/Dropbox/ukb_model.rds")
brewerPlus <- distinct_palette()
scales::show_col(brewerPlus)

softmax_by_k <- function(x) {
  # Apply softmax along K dimension (dimension 2)
  exp_x <- exp(x)
  sweep(exp_x, c(1,3), apply(exp_x, c(1,3), sum), "/")
}
model_params=ukb_params
all_thetas <- softmax_by_k(model_params$lambda)

sigmoid <- function(x) {
  1/(1 + exp(-x))
}



phi_prob <- sigmoid(model_params$phi)
phi=model_params$phi
pi_pred <- array(0, dim=c(dim(all_thetas)[1], dim(phi_prob)[2], dim(all_thetas)[3]))
N=dim(all_thetas)[1]
T=dim(all_thetas)[3]
for(i in 1:N) {
  #for (d in 1:D) {
  for (t in 1:T)
  {
    pi_pred[i, , t] <- all_thetas[i, , t] %*% phi_prob[, , t]*as.numeric(ukb_params$kappa)
  }
}


matplot(t(all_thetas[101,,]))
pop_mean=apply(all_thetas,c(2,3),mean)
matplot(t(pop_mean))

image(t(phi_prob[,113,]))

contributions=as.matrix(all_thetas[101,,])*(as.matrix(phi_prob[,113,]))*as.numeric(ukb_params$kappa)
matplot(t(contributions))




library(ggplot2)
library(reshape2)
library(patchwork)   # For combining plots

# 1. Individual signature trajectory
theta_indiv <- as.data.frame(t(all_thetas[101, , ]))
colnames(theta_indiv) <- paste0("Sig", 1:ncol(theta_indiv))
theta_indiv$Time <- 1:nrow(theta_indiv)
theta_indiv_long <- melt(theta_indiv, id.vars = "Time", variable.name = "Signature", value.name = "Theta")

# 2. Population mean signature trajectory
pop_mean <- apply(all_thetas, c(2,3), mean)
pop_mean_df <- as.data.frame(t(pop_mean))
colnames(pop_mean_df) <- paste0("Sig", 1:ncol(pop_mean_df))
pop_mean_df$Time <- 1:nrow(pop_mean_df)
pop_mean_long <- melt(pop_mean_df, id.vars = "Time", variable.name = "Signature", value.name = "Theta")

# 3. Signature contributions for individual and disease
phi_vec <- phi_prob[, 113, ]  # K x T
contributions <- t(all_thetas[101, , ]) * t(phi_vec) * as.numeric(ukb_params$kappa)  # T x K
contrib_df <- as.data.frame(contributions)
colnames(contrib_df) <- paste0("Sig", 1:ncol(contrib_df))
contrib_df$Time <- 1:nrow(contrib_df)
contrib_long <- melt(contrib_df, id.vars = "Time", variable.name = "Signature", value.name = "Contribution")



library(ggplot2)
library(reshape2)
library(patchwork)

# Assume all_thetas, phi_prob, ukb_params$kappa, and brewerPlus are loaded

# Individual and population theta (K x T)
theta_indiv <- t(all_thetas[101, , ])
pop_mean <- t(apply(all_thetas, c(2,3), mean))

sig_names <- c(0:20)

colnames(theta_indiv) <- sig_names
colnames(pop_mean) <- sig_names

theta_df <- rbind(
  data.frame(Time = 30:81, theta_indiv, Type = "Individual"),
  data.frame(Time = 30:81, pop_mean, Type = "Population")
)
colnames(theta_df)[c(2:22)]=c(0:20)
theta_long <- melt(theta_df, id.vars = c("Time", "Type"), variable.name = "Signature", value.name = "Theta")
# Combine for plotting


phi_df <- as.data.frame(phi[, 113, ])
colnames(phi_df) <- 30:81
phi_df$Signature=c(0:20)
phi_long <- melt(phi_df, id.vars = "Signature", variable.name = "Age", value.name = "LogOdds")

colnames(contributions) <- sig_names
rownames(contributions) = 30:81
contrib_df <- data.frame(Time = 1:nrow(contributions), data.frame(contributions))
colnames(contrib_df)[2:22]=sig_names
contrib_long <- melt(contrib_df, id.vars = "Time", variable.name = "Signature", value.name = "Contribution")

# Use your custom palette
n_sigs <- length(unique(theta_long$Signature))
my_colors <- brewerPlus[1:21]

# Top: Stacked area plot for theta
# Top: Line plot for theta (remove both color and fill legends)
p1 <- ggplot(theta_long, aes(x = Time, y = Theta, linetype = Type, color = Signature)) +
  geom_line(size = 1) +
  scale_color_manual(values = my_colors) +
  theme_classic() +
  labs(title = "Signature Proportions Over Time", y = "Theta", x = "Time") +
  theme(legend.position = "none")  # Remove all legends

# Middle: Heatmap of phi_prob (remove legend)
p2 <- ggplot(phi_long, aes(x = Age, y = Signature, fill = LogOdds)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red",mid="white",midpoint=-10) +
  theme_classic() +
  labs(title = "Temporal Signature Associations for Disease", y = "Signature", x = "Age") 




# Bottom: Stacked area plot for contributions (keep legend)
p3 <- ggplot(contrib_long, aes(x = Time, y = Contribution, fill = Signature)) +
  geom_area(position = "stack", alpha = 0.7) +
  scale_fill_manual(values = my_colors) +
  theme_classic() +
  labs(title = "Signature Contributions to Risk", y = "Risk", x = "Time") +
  theme(legend.position = "bottom")

# Combine plots (no guides = "collect" needed)
final_plot <- p1 / p2 / p3 + plot_layout(heights = c(1, 1, 1))
print(final_plot)

ggsave(plot = final_plot,filename ="contributions.pdf",width = 10,height = 20)
