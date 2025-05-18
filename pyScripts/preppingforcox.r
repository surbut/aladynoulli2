
library(reticulate)
## convert ro R
#use_condaenv("r-tensornoulli")
use_condaenv("/opt/miniconda3/envs/new_env_pyro2", required = TRUE)
torch <- import("torch")
tensor_to_r <- function(tensor) {
  as.array(tensor$detach()$cpu()$numpy())
}


## the trained data was 20000-30000

ukb_train = torch$load(
  "/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_model_W0.0001_jointphi_sexspecific_20000_30000.pt",
  weights_only = FALSE
)

Y_train_load = tensor_to_r(ukb_train$Y)
E_mat = tensor_to_r(ukb_train$E)


#### 
ukb_test=torch$load(
  "/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_model_W0.0001_jointphi_sexspecific_0_10000.pt",
  weights_only = FALSE
)

Y_test_load = tensor_to_r(ukb_test$Y)
E_mat = tensor_to_r(ukb_train$E)
saveRDS(Y_test_load,"ukb_Y_test.rds")