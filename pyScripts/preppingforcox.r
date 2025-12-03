
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

saveRDS(Y_train_load,"ukb_Y_train.rds")
#### 
ukb_test=torch$load(
  "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_model_W0.0001_jointphi_sexspecific_0_10000.pt",
  weights_only = FALSE
)

Y_test_load = tensor_to_r(ukb_test$Y)
E_enroll = tensor_to_r(ukb_test$E)

saveRDS(Y_test_load,"ukb_Y_test.rds")
saveRDS(E_enroll,"ukb_E_enroll_0_1000k.rds")

E_complete=torch$load(
  "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_matrix.pt",
  weights_only = FALSE
)
E_complete = tensor_to_r(E_complete)


clusters=torch$load(
  "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_400k.pt",
  weights_only = FALSE
)


pi_train=torch$load(
  "/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_enroll_sex_20000_30000.pt",
  weights_only = FALSE
)

pi_train=tensor_to_r(pi_train)

saveRDS(pi_train,file = "/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_enroll_sex_20000_30000.rds")

pi_test=torch$load(
  "/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_enroll_sex_0_10000.pt",
  weights_only = FALSE
)


pi_test=tensor_to_r(pi_test)

saveRDS(pi_test,"/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_enroll_sex_0_10000.rds")

####
####
####
####
clusters=torch$load(
"/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_400k.pt",
weights_only = FALSE
)
disease_names=read.csv("~/Library/CloudStorage/Dropbox-Personal/disease_names.csv")
dn=disease_names['x']

df=data.frame(dn,clusters)
df$r_index=rownames(df)
write.csv(df,"~/aladynoulli2/Disease_names_by_cluster.csv",row.names=FALSE,quote=FALSE)


pi_test=torch$load(
  "/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_full_leakage_free_0_10000.pt",
  weights_only = FALSE
)


pi_test=tensor_to_r(pi_test)

saveRDS(pi_test,"/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_full_leakage_free_0_10000.rds")



#####


pi_train=torch$load(
  "/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_full_leakage_free_20000_30000.pt",
  weights_only = FALSE
)



pi_train=tensor_to_r(pi_train)

saveRDS(pi_train,"/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_full_leakage_free_20000_30000.rds")

###

pi_train=torch$load(
  "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pi_full_leakage_free_0_10000_fixedphi.pt",
  weights_only = FALSE
)



pi_train=tensor_to_r(pi_train)

saveRDS(pi_train,"/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pi_full_leakage_free_0_10000_fixedphi.rds")

###
###
###

pi_train=torch$load(
  "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/pi_enroll_fixedphi_sex_20000_30000.pt",
  weights_only = FALSE
)

pi_train=tensor_to_r(pi_train)

saveRDS(pi_train,file = "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/pi_enroll_fixedphi_sex_20000_30000.rds")

pi_test=torch$load(
  "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/pi_enroll_fixedphi_sex_0_10000.pt",
  weights_only = FALSE
)


pi_test=tensor_to_r(pi_test)

saveRDS(pi_test,"/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/pi_enroll_fixedphi_sex_0_10000.rds")


###
###


pi_train=torch$load(
  "/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_ENROLLMENT_pooled/pi_enroll_fixedphi_sex_20000_30000.pt",
  weights_only = FALSE
)

pi_train=tensor_to_r(pi_train)

saveRDS(pi_train,file = "/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_ENROLLMENT_pooled/pi_enroll_fixedphi_sex_20000_30000.rds")

pi_test=torch$load(
  "/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_ENROLLMENT_pooled/pi_enroll_fixedphi_sex_0_10000.pt",
  weights_only = FALSE
)


pi_test=tensor_to_r(pi_test)

saveRDS(pi_test,"/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_ENROLLMENT_pooled/pi_enroll_fixedphi_sex_0_10000.rds")

#####
#####
pi_train=torch$load(
  "/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_prediction_jointphi_sex_pcs/pi_enroll_sex_20000_30000.pt",
  weights_only = FALSE
)

pi_train=tensor_to_r(pi_train)

saveRDS(pi_train,file = 
          "/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_prediction_jointphi_sex_pcs/pi_enroll_sex_20000_30000.rds")



pi_test=torch$load(
  "/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_prediction_jointphi_sex_pcs/pi_enroll_sex_0_10000.pt",
  weights_only = FALSE
)

pi_test=tensor_to_r(pi_train)

saveRDS(pi_test,file = 
          "/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_prediction_jointphi_sex_pcs/pi_enroll_sex_0_10000.rds")


