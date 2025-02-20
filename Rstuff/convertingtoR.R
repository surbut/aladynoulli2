# Install if needed
# install.packages("reticulate")


## convert ro R

library(reticulate)
model=torch$load("/Users/sarahurbut/Dropbox (Personal)/resultstraj_genetic_scale1/results/output_0_10000/model.pt",weights_only=FALSE)

# Load torch through reticulate
torch <- import("torch")
# Function to convert torch tensor to R array
tensor_to_r <- function(tensor) {
  as.array(tensor$detach()$cpu()$numpy())
}

# Convert model parameters
model_params <- list(
  phi = tensor_to_r(model$model_state_dict$phi),
  psi = tensor_to_r(model$model_state_dict$psi),
  lambda = tensor_to_r(model$model_state_dict$lambda)
)

mu_d=tensor_to_r(model$logit_prevalence_t)

model=torch$load("/Users/sarahurbut/Dropbox (Personal)/resultstraj_genetic_scale1/results/
                 output_0_10000/model.pt",weights_only=FALSE)


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

library(reticulate)
library(torch)

# 1. Load the model and get phi
model <- torch$load('/Users/sarahurbut/Dropbox (Personal)/resultstraj_genetic_scale1/results/output_0_10000/model.pt',weights_only=FALSE)
phi_kd <- tensor_to_r(model$model_state_dict$phi)  # Convert to R array
np <- import("numpy")

# 2. Load the lambdas
all_lambdas <- np$load('/Users/sarahurbut/aladynoulli2/pyScripts/oldstuff/all_lambdas_combined_smallg.npy')
all_lambdas <- as.array(all_lambdas)  # Convert to R array


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

all_thetas <- softmax_by_k(all_lambdas)
phi_prob <- 1/(1 + exp(-phi_kd))

# 4. Convert phi to probabilities using sigmoid
sigmoid <- function(x) {
  1/(1 + exp(-x))
}
phi_prob <- sigmoid(phi_kd)

# 5. Calculate pi predictions using tensor multiplication
# This is equivalent to the Python einsum('nkt,kdt->ndt')
pi_pred <- array(0, dim=c(dim(all_thetas)[1], dim(phi_prob)[2], dim(all_thetas)[3]))
for(t in 1:dim(all_thetas)[3]) {
  pi_pred[,,t] <- all_thetas[,,t] %*% phi_prob
}

# Verify dimensions and values
print(paste("Pi predictions shape:", paste(dim(pi_pred), collapse=" x ")))
print(paste("Range of values:", min(pi_pred), "to", max(pi_pred)))
print(paste("Mean value:", mean(pi_pred)))
