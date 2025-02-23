# Install if needed
## restart R
install.packages("reticulate")
library(reticulate)
## convert ro R
use_condaenv("r-tensornoulli")

torch <- import("torch")
library(reticulate)
library(torch)
# 1. Load the model and get phi

model=torch$load("/Users/sarahurbut/Dropbox (Personal)/resultstraj_genetic_scale1/results/output_0_10000/model.pt",weights_only=FALSE)


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

all_thetas <- softmax_by_k(all_lambdas)
phi_prob <- 1/(1 + exp(-phi_kd))

# 4. Convert phi to probabilities using sigmoid
sigmoid <- function(x) {
  1/(1 + exp(-x))
}
phi_prob <- sigmoid(phi_kd)
D=348

# 5. Calculate pi predictions using tensor multiplication
# This is equivalent to the Python einsum('nkt,kdt->ndt')
pi_pred <- array(0, dim=c(dim(all_thetas)[1], dim(phi_prob)[2], dim(all_thetas)[3]))
for(i in 1:N) {
  for (d in 1:D) {
    for (t in 1:T)
    {
      pi_pred[i, d, t] <- all_thetas[i, , t] %*% phi_prob[, d, t]
    }
  }
}


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
ascvd_probs=pi_pred[,c(112:115),]*cal
## probability of surviving each
ascvd_survival=1-ascvd_probs
## probabilty of failing one per time interval, NxT
ascvd_at_least_one=apply(ascvd_survival,c(1,3),function(x){1-prod(x)})
### Probaility of failing ten years
ascvd_risk=matrix(0,nrow=dim(ya)[1],ncol=dim(ya)[3]-10)

for(t in 1:(dim(ya)[3]-10)){
  ascvd_risk[,t]=1-prod(1-ascvd_at_least_one[,t:(t+10)])
}

ascvd_risk=data.frame(ascvd_risk)
enroll_index=pce_df$age-30
ascvd_risk[,eval(enroll_index)]

ten_year_risks = ascvd_risk[cbind(1:nrow(ascvd_risk), enroll_index)]

#all.equal(as.character(rownames(biga[[1]]))[1:10000],as.character(pce_df$id))
