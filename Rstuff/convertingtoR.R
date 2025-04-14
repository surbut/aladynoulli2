# Install if needed
## restart R
library(reticulate)
## convert ro R
#use_condaenv("r-tensornoulli")
use_condaenv("/opt/miniconda3/envs/new_env_pyro2", required = TRUE)
torch <- import("torch")

# 1. Load the model and get phi

#model=torch$load("/Users/sarahurbut/Dropbox (Personal)/resultstraj_genetic_scale1/results/output_0_10000/model.pt",weights_only=FALSE)
model=torch$load("/Users/sarahurbut/Dropbox/resultshighamp//results/output_0_10000/model.pt",weights_only=FALSE)
sig_refs=torch$load("/Users/sarahurbut/Dropbox/data_for_running/reference_trajectories.pt",weights_only=FALSE)

sigs=tensor_to_r(sig_refs$signature_refs)
# Function to convert torch tensor to R array
tensor_to_r <- function(tensor) {
  as.array(tensor$detach()$cpu()$numpy())
}



sigs=rbind(sigs,rep(-5,52))
softmax=function(x){
  exp(x)/sum(exp(x))
}




# 3. Calculate population-level theta using softmax
softmax <- function(x) {
  exp(x) / sum(exp(x))
}
pop_theta <- apply(sigs, 2, softmax)  # [K+1, T]
# 5. Calculate pi predictions using tensor multiplication
# This is equivalent to the Python einsum('nkt,kdt->ndt')
pi_pred <- array(0, dim=c(dim(all_thetas)[1], dim(phi_prob)[2], dim(all_thetas)[3]))
N=dim(all_thetas)[1]
for(i in 1:N) {
  #for (d in 1:D) {
  for (t in 1:T)
  {
    pi_pred[i, , t] <- all_thetas[i, , t] %*% phi_prob[, , t]
  }
}

image(t(d))

matplot(t(phi_prob[6,,]))

model_apu=torch$load("/Users/sarahurbut/Dropbox/model_with_kappa_bigam_aou.pt",weights_only=FALSE)





# 4. Calculate average disease probabilities
T <- dim(phi_prob)[3]
D <- dim(phi_prob)[2]
avg_pi <- matrix(0, nrow=D, ncol=T)

# For each timepoint

for(t in 1:T) {
  # Get phi at this timepoint
  phi_t <- phi_prob[,,t]  # [K, D]
  
  for(n in 1:N){
  
  # Get population theta at this timepoint and ensure it's a column vector
  theta_t <- matrix(pop_theta[,t], ncol=1)  # [K,1]
  
  # Calculate disease probabilities: [D] = ([D,K] %*% [K,1]) * scalar
  avg_pi[,t] <- as.vector(t(phi_t) %*% theta_t) * kappa
}

pop_theta=apply(sigs,2,softmax)
matplot(t(pop_theta))
theta=apply(model_params$lambda,c(1,3),softmax)
# Convert model parameters
model_params <- list(
  phi = tensor_to_r(model$model_state_dict$phi),
  psi = tensor_to_r(model$model_state_dict$psi),
  lambda = tensor_to_r(model$model_state_dict$lambda),
  kappa=tensor_to_r(model$model_state_dict$kappa)
)

mu_d=tensor_to_r(model$logit_prevalence_t)



image(cor(model_params$phi[,,10]))


library(corrplot)

# Assuming disease names are stored in your model
library(corrplot)

# Create a clearer correlation plot
plot_disease_clusters <- function(phi, time_point) {
  cor_matrix <- cor(phi[,,time_point])
  
  corrplot(cor_matrix,
           method = "color",
           col = colorRampPalette(c("#4477AA", "white", "#EE6677"))(100),
           type = "full",           # Show full matrix instead of just upper triangle
           tl.pos = "n",            # No text labels
           cl.pos = "r",            # Color legend on right
           addCoef.col = NA,        # Don't show correlation coefficients
           title = paste("Disease Clustering at Time", time_point))
}

# Create the plot
plot_disease_clusters(model_params$phi, time_point = 10)

# If you want to see the structure more clearly, we could also try:
library(pheatmap)
pheatmap(cor(model_params$phi[,,10]),
         show_rownames = FALSE,
         show_colnames = FALSE,
         clustering_method = "ward.D2")


plot_disease_clusters(model$Y, time_point = 10)

phi_kd <- tensor_to_r(model$model_state_dict$phi)  # Convert to R array
np <- import("numpy")

# 2. Load the lambdas
#all_lambdas <- np$load('/Users/sarahurbut/aladynoulli2/pyScripts/oldstuff/all_lambdas_combined_smallg.npy')
#all_lambdas <- as.array(all_lambdas)  # Convert to R array

all_lambdas <- as.array(model_params$lambda)  # Convert to R array

# Print dimensions to understand the structure
print("Dimensions:")
print(paste("all_lambdas:", paste(dim(all_lambdas), collapse=" x ")))
print(paste("phi_kd:", paste(dim(phi_kd), collapse=" x ")))

# Corrected softmax function
softmax_by_k <- function(x) {
  # Apply softmax along K dimension (dimension 2)
  exp_x <- exp(x)
  sweep(exp_x, c(1,3), apply(exp_x, c(1,3), sum), "/")
}

all_thetas <- softmax_by_k(model_params$lambda)
phi_prob <- 1/(1 + exp(-phi_kd))

# 4. Convert phi to probabilities using sigmoid
sigmoid <- function(x) {
  1/(1 + exp(-x))
}
phi_prob <- sigmoid(model_params$phi)





y_observed_mean=apply(ya,c(2,3),mean)
cal=mean(y_observed_mean)/mean(apply(pi_pred,c(2,3),mean))
# Verify dimensions and values
print(paste("Pi predictions shape:", paste(dim(pi_pred), collapse=" x ")))
print(paste("Range of values:", min(pi_pred), "to", max(pi_pred)))
print(paste("Mean value:", mean(pi_pred)))


plot(log(model$prevalence_t),log(means*3.03))
abline(c(0,1))


pce_df=readRDS("~/Dropbox (Personal)/first10kukb_pce.rds")
pce_goff=pce_df$pce_goff
pce_goff[is.na(pce_goff)]=mean(pce_df$pce_goff,na.rm=TRUE)
## probabilities of ascvd indices
ascvd_probs=pi_pred[,c(112:117),]*cal
## probability of surviving each
ascvd_survival=1-ascvd_probs # Nx6xT
## probabilty of failing one per time interval, NxT
ascvd_at_least_one=apply(ascvd_survival,c(1,3),function(x){1-prod(x)}) #nxT
pro_surv_all=1-ascvd_at_least_one ##nxT
### Probaility of failing ten years
ascvd_risk=matrix(0,nrow=dim(ya)[1],ncol=dim(ya)[3]-10)

for(i in 1:N){
  for(t in 1:(dim(ya)[3]-10)){
    ascvd_risk[i,t]=1-prod(pro_surv_all[i,t:(t+10)],scientific=TRUE)
  }
}


ascvd_risk=data.frame(ascvd_risk)
enroll_index=pce_df$age-30
ascvd_risk[,eval(enroll_index)]

ten_year_risks = ascvd_risk[cbind(1:nrow(ascvd_risk), enroll_index)]

all.equal(as.character(rownames(biga[[1]]))[1:10000],as.character(pce_df$id))

prevent=fread("~/Dropbox (Personal)/for_akl/ukb_pce_prevent_scores.csv")
m=merge(pce_df,prevent,by.x="id",by.y="eid",all.x = T)
m$prevent_impute=m$prevent_base_ascvd_risk
m$prevent_impute[is.na(m$prevent_impute)]=mean(na.omit(m$prevent_base_ascvd_risk))
saveRDS(m,"~/Dropbox (Personal)/pce_df_prevent.rds")

original_psi=torch$load('/Users/sarahurbut/Dropbox/data_for_running/initial_psi_400k.pt')
original_psi=tensor_to_r(psi)
original_psi=tensor_to_r(original_psi)
image(original_psi)
image(model_params$psi)

sm=apply(original_psi,2,function(x){exp(x)/sum(exp(x))})
smo=apply(model_params$psi[c(1:20),],2,function(x){exp(x)/sum(exp(x))})
smo2=apply(model_params$psi[c(1:20),],2,function(x){(x)/sum((x))})

# Assuming original_psi_softmax_r is your softmax-transformed matrix
# Calculate the variance for each signature
variance_explained <- apply(smo, 2, var)

# Calculate the proportion of variance explained
total_variance <- sum(variance_explained)
proportion_variance_explained <- variance_explained / total_variance



library(ggplot2)
library(gridExtra)
library(viridis)

## signatures across diseases 
matplot(t(phi_prob[5,,]))
matplot(t(phi_prob[1,,]))
matplot(t(phi_prob[18,,]))

# 1. Population-level signature proportions
sigs <- rbind(sigs, rep(-5, 52))
pop_theta <- apply(sigs, 2, softmax)
rownames(pop_theta)=paste("Sig",seq(1:21))
p1 <- ggplot(data = as.data.frame(t(pop_theta))) +
  geom_line(aes(x = 1:52, y = V1, color = "Sig 1")) +
  geom_line(aes(x = 1:52, y = V2, color = "Sig 2")) +
  # ... add other signatures
  labs(title = "Population-Level Signature Proportions",
       x = "Age", y = "Proportion") +
  theme_minimal()

# 2. Disease probabilities
# Calculate individual-averaged probabilities
N <- dim(all_thetas)[1]
D <- dim(phi_prob)[2]
T <- dim(phi_prob)[3]

pi_individual_avg <- matrix(0, nrow=D, ncol=T)
for(t in 1:T) {
  pi_t <- matrix(0, nrow=N, ncol=D)
  for(i in 1:N) {
    pi_t[i,] <- all_thetas[i,,t] %*% phi_prob[,,t]
  }
  pi_individual_avg[,t] <- colMeans(pi_t)
}
pi_individual_avg <- pi_individual_avg * kappa

# Population-level probabilities
pi_pop <- matrix(0, nrow=D, ncol=T)
for(t in 1:T) {
  pi_pop[,t] <- as.vector(t(phi_prob[,,t]) %*% matrix(pop_theta[,t], ncol=1)) * kappa
}

# Create heatmap of differences
diff_matrix <- pi_individual_avg - pi_pop

p2 <- ggplot(data = reshape2::melt(diff_matrix)) +
  geom_tile(aes(x = Var2, y = Var1, fill = value)) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red") +
  labs(title = "Individual vs Population Probability Differences",
       x = "Age", y = "Disease") +
  theme_minimal()

# Arrange plots
grid.arrange(p1, p2, ncol=1)


library(ggplot2)
library(gridExtra)
library(reshape2)
library(viridis)

# 1. Signatures across diseases (for selected signatures)
selected_sigs <- c(6, 7, 16)  # Your selected signatures
sig_names <- c("Cardiovascular", "Cancer", "Metabolic")  # Example names

# Create data frames for signature-specific disease trajectories
plot_data_list <- list()
for(i in seq_along(selected_sigs)) {
  sig_idx <- selected_sigs[i]
  df <- data.frame(
    Time = rep(1:52, ncol(phi_prob)),
    Disease = rep(model$disease_names,each=52),
    Probability = as.vector(t(phi_prob[sig_idx,,]))
  )
  plot_data_list[[i]] <- df
}

# Create signature-specific plots
sig_plots <- lapply(seq_along(selected_sigs), function(i) {
  ggplot(plot_data_list[[i]], aes(x=Time, y=Probability, color=Disease)) +
    geom_line(alpha=0.6) +
    labs(title=paste("Signature", selected_sigs[i], "-", sig_names[i]),
         x="Age", y="Disease Probability") +
    theme_minimal() +
    theme(legend.position="none") +
    scale_color_viridis(discrete=TRUE)
})

# 2. Population-level signature proportions
sigs <- rbind(sigs, rep(-5, 52))
pop_theta <- apply(sigs, 2, softmax)
rownames(pop_theta) <- paste("Sig", 1:nrow(pop_theta))

theta_df <- melt(pop_theta)
colnames(theta_df) <- c("Signature", "Time", "Proportion")

p_theta <- ggplot(theta_df, aes(x=Time, y=Proportion, color=Signature)) +
  geom_line() +
  labs(title="Population-Level Signature Proportions",
       x="Age", y="Proportion") +
  theme_minimal() +
  theme(legend.position="right") +
  scale_color_viridis(discrete=TRUE)

# Arrange all plots
grid.arrange(
  sig_plots[[1]], sig_plots[[2]], sig_plots[[3]], 
  p_theta,
  ncol=2
)


library(ggplot2)
library(gridExtra)
library(reshape2)
library(viridis)

# 1. Signatures across diseases (for selected signatures)
selected_sigs <- c(5, 1, 18)  # Your selected signatures
sig_names <- c("Cardiovascular", "Cancer", "Metabolic")  # Example names

# Create data frames with disease names
plot_data_list <- list()
for(i in seq_along(selected_sigs)) {
  sig_idx <- selected_sigs[i]
  df <- data.frame(
    Time = rep(1:52, ncol(phi_prob)),
    Disease = rep(model$disease_names, each=52),
    Probability = as.vector(t(phi_prob[sig_idx,,]))
  )
  plot_data_list[[i]] <- df
}

# Create signature-specific plots
# For each signature, show only top N diseases for clarity
N_top_diseases <- 10

sig_plots <- lapply(seq_along(selected_sigs), function(i) {
  # Find top diseases for this signature based on maximum probability
  top_diseases <- plot_data_list[[i]] %>%
    group_by(Disease) %>%
    summarize(max_prob = max(Probability)) %>%
    arrange(desc(max_prob)) %>%
    head(N_top_diseases) %>%
    pull(Disease)
  
  # Filter data for top diseases
  plot_data <- plot_data_list[[i]] %>%
    filter(Disease %in% top_diseases)
  
  ggplot(plot_data, aes(x=Time, y=Probability, color=Disease)) +
    geom_line(linewidth=1, alpha=0.8) +
    labs(title=paste("Signature", selected_sigs[i], "-", sig_names[i]),
         x="Age", y="Disease Probability") +
    theme_minimal() +
    theme(legend.position="right",
          legend.text = element_text(size=8),
          plot.title = element_text(size=12)) +
    scale_color_viridis(discrete=TRUE) +
    guides(color=guide_legend(ncol=1))
})

# Population-level signature proportions plot (as before)
sigs <- rbind(sigs, rep(-5, 52))
pop_theta <- apply(sigs, 2, softmax)
rownames(pop_theta) <- paste("Sig", 1:nrow(pop_theta))

theta_df <- melt(pop_theta)
colnames(theta_df) <- c("Signature", "Time", "Proportion")

p_theta <- ggplot(theta_df, aes(x=Time, y=Proportion, color=Signature)) +
  geom_line() +
  labs(title="Population-Level Signature Proportions",
       x="Age", y="Proportion") +
  theme_minimal() +
  theme(legend.position="right") +
  scale_color_viridis(discrete=TRUE)

# Arrange all plots
grid.arrange(
  sig_plots[[1]], sig_plots[[2]], 
  sig_plots[[3]], p_theta,
  ncol=2
)


library(ggplot2)
library(gridExtra)
library(reshape2)
library(viridis)

# 1. Signatures across diseases (for selected signatures)
selected_sigs <- c(5, 1, 18)  # Your selected signatures
sig_names <- c("Cardiovascular", "Cancer", "Metabolic")  # Example names

# Create data frames with disease names - simplified version
plot_data_list <- list()
for(i in seq_along(selected_sigs)) {
  sig_idx <- selected_sigs[i]
  df <- data.frame(
    Time = 1:52,
    Disease_Matrix = t(phi_prob[sig_idx,,])  # Transpose to get diseases as columns
  )
  colnames(df)[-1] <- model$disease_names  # Name columns after diseases
  
  # Melt to long format
  df_long <- melt(df, id.vars = "Time", 
                  variable.name = "Disease", 
                  value.name = "Probability")
  
  plot_data_list[[i]] <- df_long
}

# Create signature-specific plots
N_top_diseases <- 10

sig_plots <- lapply(seq_along(selected_sigs), function(i) {
  # Find top diseases for this signature
  top_diseases <- plot_data_list[[i]] %>%
    group_by(Disease) %>%
    summarize(max_prob = max(Probability)) %>%
    arrange(desc(max_prob)) %>%
    head(N_top_diseases) %>%
    pull(Disease)
  
  # Filter data for top diseases
  plot_data <- plot_data_list[[i]] %>%
    filter(Disease %in% top_diseases)
  
  ggplot(plot_data, aes(x=Time, y=Probability, color=Disease)) +
    geom_line(linewidth=1, alpha=0.8) +
    labs(title=paste("Signature", selected_sigs[i], "-", sig_names[i]),
         x="Age", y="Disease Probability") +
    theme_minimal() +
    theme(legend.position="right",
          legend.text = element_text(size=8),
          plot.title = element_text(size=12)) +
    scale_color_viridis(discrete=TRUE)
})

# Arrange plots
grid.arrange(
  sig_plots[[1]], sig_plots[[2]], 
  sig_plots[[3]], 
  ncol=2
)
