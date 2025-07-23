library(pROC)
library(survival)
source("tdcsourcecode.R")


pi_test_full <- readRDS("pi_full_leakage_free_0_10000_fixedphi.rds")
Y_test=readRDS("ukb_Y_test.rds")
disease_mapping=readRDS("disease_mapping.rds")
major_diseases=readRDS("major_diseases.rds")
disease_names=readRDS("disease_names.rds")


# Evaluate time-dependent Cox models
tdc_auc_results <- test_time_dependent_cox(
 Y_test = Y_test,
 FH_processed = FH_processed,
 test_indices = 0:10000,
 disease_mapping = disease_mapping,
 major_diseases = major_diseases,
 disease_names = disease_names,
 follow_up_duration_years = 7,
 fitted_models = NULL,
 pi_test = pi_test_full)
  

tdc_auc_df <- data.frame(
 disease_group = names(tdc_auc_results[[1]]),
 auc = unlist(tdc_auc_results[[1]])
)

tdc_c_df <- data.frame(
 disease_group = names(tdc_auc_results[[2]]),
 c = unlist(tdc_auc_results[[2]])
)