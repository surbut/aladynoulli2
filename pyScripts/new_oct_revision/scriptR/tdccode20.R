# cox_utils.R

### here we load some stuff##

library(survival)
library(broom)
library(dplyr)

FH_processed = read.csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh.csv')

#####

disease_names = readRDS("~/aladynoulli2/pyScripts/ukb_model.rds")$disease_names[, 1]
train_indices = 20001:30000
major_diseases <- list(
  ASCVD = c(
    "Myocardial infarction",
    "Coronary atherosclerosis",
    "Other acute and subacute forms of ischemic heart disease",
    "Unstable angina (intermediate coronary syndrome)",
    "Angina pectoris",
    "Other chronic ischemic heart disease, unspecified"
  ),
  Diabetes = c("Type 2 diabetes"),
  Atrial_Fib = c("Atrial fibrillation and flutter"),
  CKD = c("Chronic renal failure [CKD]", "Chronic Kidney Disease, Stage III"),
  All_Cancers = c(
    "Colon cancer",
    "Malignant neoplasm of rectum, rectosigmoid junction, and anus",
    "Cancer of bronchus; lung",
    "Breast cancer [female]",
    "Malignant neoplasm of female breast",
    "Cancer of prostate",
    "Malignant neoplasm of bladder",
    "Secondary malignant neoplasm",
    "Secondary malignancy of lymph nodes",
    "Secondary malignancy of respiratory organs",
    "Secondary malignant neoplasm of digestive systems",
    "Secondary malignant neoplasm of liver",
    "Secondary malignancy of bone"
  ),
  Stroke = c(
    "Cerebral artery occlusion, with cerebral infarction",
    "Cerebral ischemia"
  ),
  Heart_Failure = c("Congestive heart failure (CHF) NOS", "Heart failure NOS"),
  Pneumonia = c("Pneumonia", "Bacterial pneumonia", "Pneumococcal pneumonia"),
  COPD = c(
    "Chronic airway obstruction",
    "Emphysema",
    "Obstructive chronic bronchitis"
  ),
  Osteoporosis = c("Osteoporosis NOS"),
  Anemia = c(
    "Iron deficiency anemias, unspecified or not due to blood loss",
    "Other anemias"
  ),
  Colorectal_Cancer = c(
    "Colon cancer",
    "Malignant neoplasm of rectum, rectosigmoid junction, and anus"
  ),
  Breast_Cancer = c("Breast cancer [female]", "Malignant neoplasm of female breast"),
  Prostate_Cancer = c("Cancer of prostate"),
  Lung_Cancer = c("Cancer of bronchus; lung"),
  Bladder_Cancer = c("Malignant neoplasm of bladder"),
  Secondary_Cancer = c(
    "Secondary malignant neoplasm",
    "Secondary malignancy of lymph nodes",
    "Secondary malignancy of respiratory organs",
    "Secondary malignant neoplasm of digestive systems"
  ),
  Depression = c("Major depressive disorder"),
  Anxiety = c("Anxiety disorder"),
  Bipolar_Disorder = c("Bipolar"),
  Rheumatoid_Arthritis = c("Rheumatoid arthritis"),
  Psoriasis = c("Psoriasis vulgaris"),
  Ulcerative_Colitis = c("Ulcerative colitis"),
  Crohns_Disease = c("Regional enteritis"),
  Asthma = c("Asthma"),
  Parkinsons = c("Parkinson's disease"),
  Multiple_Sclerosis = c("Multiple sclerosis"),
  Thyroid_Disorders = c(
    "Thyrotoxicosis with or without goiter",
    "Secondary hypothyroidism",
    "Hypothyroidism NOS"
  )
)


disease_mapping = list(
  ASCVD = c('heart_disease', 'heart_disease.1'),
  Stroke = c('stroke', 'stroke.1'),
  Diabetes = c('diabetes', 'diabetes.1'),
  Breast_Cancer = c('breast_cancer', 'breast_cancer.1'),
  Prostate_Cancer = c('prostate_cancer', 'prostate_cancer.1'),
  Lung_Cancer = c('lung_cancer', 'lung_cancer.1'),
  Colorectal_Cancer = c('bowel_cancer', 'bowel_cancer.1'),
  Depression = character(0),
  Osteoporosis = character(0),
  Parkinsons = c('parkinsons', 'parkinsons.1'),
  COPD = character(0),
  Anemia = character(0),
  CKD = character(0),
  Heart_Failure = c('heart_disease', 'heart_disease.1'),
  Pneumonia = character(0),
  Atrial_Fib = character(0),
  Bladder_Cancer = character(0),
  Secondary_Cancer = character(0),
  Anxiety = character(0),
  Bipolar_Disorder = character(0),
  Rheumatoid_Arthritis = character(0),
  Psoriasis = character(0),
  Ulcerative_Colitis = character(0),
  Crohns_Disease = character(0),
  Asthma = character(0),
  Multiple_Sclerosis = character(0),
  Thyroid_Disorders = character(0)
)

fit_cox_baseline_models <- function(Y_train,
                                    FH_processed,
                                    train_indices,
                                    disease_mapping,
                                    major_diseases,
                                    disease_names,
                                    follow_up_duration_years = 10,
                                    pi_train=NULL) {
  fitted_models <- list()
  
  
  FH_train <- FH_processed[train_indices, ]
  
  #disease_group = "ASCVD"
  for (disease_group in names(major_diseases)) {
    ## which fh_cols match
    fh_cols <- disease_mapping[[disease_group]]
    if (is.null(fh_cols))
      fh_cols <- character(0)
    if (length(fh_cols) == 0)
      cat(sprintf(" - %s: No FH columns, fitting Sex only.\n", disease_group))
    cat(sprintf(" - Fitting %s...\n", disease_group))
    
    target_sex_code <- NA
    if (disease_group == "Breast_Cancer")
      target_sex_code <- 0
    if (disease_group == "Prostate_Cancer")
      target_sex_code <- 1
    
    if (!is.na(target_sex_code)) {
      mask_train <- FH_train$sex == target_sex_code
    } else {
      mask_train <- rep(TRUE, nrow(FH_train))
    }
    
    current_FH_train <- FH_train[mask_train, ]
    current_Y_train <- Y_train[mask_train, , , drop = FALSE]
    print("dim(current_FH_train))")
    print(dim(current_FH_train))
    print("dim(current_Y_train)")
    print(dim(current_Y_train))
    
    if (!is.null(pi_train)) {
      
      current_pi_train <- pi_train[mask_train, , , drop = FALSE]
      print("dimPi")
      print(dim(current_pi_train))
    } else {
      current_pi_train <- NULL
    }
    
    if (nrow(current_FH_train) == 0) {
      cat(
        sprintf(
          "   Warning: No individuals for target sex code %s in training slice.\n",
          target_sex_code
        )
      )
      fitted_models[[disease_group]] <- NULL
      next
    }
    
    # Find disease indices
    disease_indices <- unlist(lapply(major_diseases[[disease_group]], function(disease) {
      which(tolower(disease_names) == tolower(disease))
    }))
    print("disease indices are")
    print(disease_indices)
    if (length(disease_indices) == 0) {
      fitted_models[[disease_group]] <- NULL
      next
    }
    print(disease_names[sort(disease_indices)])
    
    
    print(paste("Total events before filtering:", sum(current_Y_train[, disease_indices, ] == 1)))
    # Prepare data for Cox model
    
    n_age_filtered <- 0
    n_prevalent <- 0
    n_no_followup <- 0
    
    cox_data <- data.frame()
    for (i in seq_len(nrow(current_FH_train))) {
      
      age_at_enrollment <- current_FH_train$age[i]
      t_enroll <- as.integer(age_at_enrollment - 29)
      
      
      # Age filtering
      if (t_enroll < 0 || t_enroll >= dim(current_Y_train)[3]) {
        n_age_filtered <- n_age_filtered + 1
        next
      }
      
      # Prevalent disease check
      if (length(disease_indices) == 1 && t_enroll > 0) {
        if (any(current_Y_train[i, disease_indices, 1:(t_enroll-1)] == 1)) {
          n_prevalent <- n_prevalent + 1
          next
        }
      }
      
      
      end_time <- min(t_enroll + follow_up_duration_years,
                      dim(current_Y_train)[3])
      ymat <- current_Y_train[i, disease_indices, t_enroll:end_time, drop = TRUE]
      # Find all event indices (relative to t_enroll)
      if (length(disease_indices) == 1) {
        # ymat is a vector
        event_ages <- which(ymat == 1)
      } else {
        # ymat is a matrix
        event_ages <- which(ymat == 1, arr.ind = TRUE)[, 2]
      }
      
      if (length(event_ages) == 0) {
        age_at_event <- end_time + 29 - 1
        event <- 0
      } else {
        min_event_idx <- min(event_ages)
        age_at_event <- t_enroll + min_event_idx + 29 - 1
        event <- 1
      }
      age_enroll <- t_enroll + 29
      # Skip if no follow-up time
      
      if (age_enroll >= age_at_event) {
        n_no_followup <- n_no_followup + 1
        next
      }
      
      row <- data.frame(
        age_enroll = t_enroll + 29,
        age = age_at_event,
        event = event,
        sex = current_FH_train$sex[i]
      )
      if (length(fh_cols) > 0 &&
          all(fh_cols %in% colnames(current_FH_train))) {
        row$fh <- any(current_FH_train[i, fh_cols])
      }
      # Add noulli prediction for this person
      if (!is.null(current_pi_train)) {
        # Get the noulli prediction for this disease group
        pi_diseases <- current_pi_train[i, disease_indices, t_enroll]
        yearly_risk <- 1 - prod(1 - pi_diseases)  # Convert to yearly risk
        row$noulli_risk <- yearly_risk
      }
      
      cox_data <- rbind(cox_data, row)
    }
    print(paste("Excluded due to age:", n_age_filtered))
    print(paste("Excluded due to prevalent disease:", n_prevalent))
    print(paste("Excluded due to no follow-up:", n_no_followup))
    print(paste("Final nrow(cox_data):", nrow(cox_data)))
    print(paste("Total events before filtering:", sum(current_Y_train[, disease_indices, ] == 1)))
    print(paste("Total events after filtering:", sum(cox_data$event)))
    print("first few rows of cox data")
    print(head(cox_data))
    
    if (nrow(cox_data) == 0 || sum(cox_data$event) < 5) {
      cat(sprintf(
        "   Warning: Too few events (%d) for %s\n",
        sum(cox_data$event),
        disease_group
      ))
      fitted_models[[disease_group]] <- NULL
      next
    }
    # Fit Cox model
    formula_str <- "Surv(age_enroll,age, event) ~ sex"
    if ("fh" %in% colnames(cox_data))
      formula_str <- "Surv(age_enroll,age, event) ~ sex + fh"
    if (!is.na(target_sex_code)) {
      formula_str <- if ("fh" %in% colnames(cox_data))
        "Surv(age_enroll,age, event) ~ fh"
      else
        "Surv(age_enroll,age,event) ~ 1"
    }
    if (!is.null(pi_train) && "noulli_risk" %in% colnames(cox_data))
      formula_str <- paste(formula_str, "+ noulli_risk")
    
    
    print(formula_str)
    fit <- try(coxph(as.formula(formula_str), data = cox_data), silent =
                 TRUE)
    print(summary(fit))
    if (inherits(fit, "try-error")) {
      cat(sprintf("   Error fitting %s: %s\n", disease_group, fit))
      fitted_models[[disease_group]] <- NULL
    } else {
      fitted_models[[disease_group]] <- fit
      cat(sprintf(
        "   Model fitted for %s using %d samples.\n",
        disease_group,
        nrow(cox_data)
      ))
    }
  }
  cat("Finished fitting Cox models.\n")
  return(fitted_models)
}




###

Y_train=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/ukb_Y_train.rds")
FH_processed = read.csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh.csv')

pi_train=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pi_enroll_sex_20000_30000.rds")


cb = fit_cox_baseline_models(
  Y_train = Y_train,
  FH_processed = FH_processed,
  
  train_indices = 20001:30000,
  disease_mapping = disease_mapping,
  major_diseases = major_diseases,
  disease_names = disease_names,
  follow_up_duration_years = 10,
  #pi_train=pi_train
)


### test cox models:

library(survival)
library(pROC)
rm(Y_train)
rm(pi_train)
rm(train_indices)
Y_test=readRDS("/Users/sarahurbut/aladynoulli2/pyScripts/ukb_Y_test.rds")


test_cox_baseline_models <- function(Y_test,
                                     FH_processed,
                                     test_indices,
                                     disease_mapping,
                                     major_diseases,
                                     disease_names,
                                     follow_up_duration_years = 10,
                                     fitted_models,pi_test=NULL) {
  auc_results <- list()
  concordance_results=list()
  FH_test <- FH_processed[test_indices, ]
  
  for (disease_group in names(major_diseases)) {
    fh_cols <- disease_mapping[[disease_group]]
    if (is.null(fh_cols)) fh_cols <- character(0)
    if (length(fh_cols) == 0)
      cat(sprintf(" - %s: No FH columns, fitting Sex only.\n", disease_group))
    cat(sprintf(" - Evaluating %s...\n", disease_group))
    
    target_sex_code <- NA
    if (disease_group == "Breast_Cancer")
      target_sex_code <- 0
    if (disease_group == "Prostate_Cancer")
      target_sex_code <- 1
    
    if (!is.na(target_sex_code)) {
      mask_test <- FH_test$sex == target_sex_code
    } else {
      mask_test <- rep(TRUE, nrow(FH_test))
    }
    
    if (!is.null(pi_test)) {
      current_pi_test <- pi_test[mask_test, , , drop = FALSE]
      print("dimPi")
      print(dim(current_pi_test))
    } else {
      current_pi_test<- NULL
    }
    current_FH_test <- FH_test[mask_test, ]
    current_Y_test <- Y_test[mask_test, , , drop = FALSE]
    print(paste("After sex filtering:", nrow(current_FH_test)))
    
    
    
    
    if (nrow(current_FH_test) == 0) {
      cat(sprintf("   Warning: No individuals for target sex code %s in testing slice.\n", target_sex_code))
      next
    }
    
    disease_indices <- unlist(lapply(major_diseases[[disease_group]], function(disease) {
      which(tolower(disease_names) == tolower(disease))
    }))
    if (length(disease_indices) == 0) next
    
    
    print(paste("Total events before filtering:", sum(current_Y_test[, disease_indices, ] == 1)))
    
    n_age_filtered <- 0
    n_prevalent <- 0
    n_no_followup <- 0
    
    
    cox_data <- data.frame()
    for (i in seq_len(nrow(current_FH_test))) {
      age_at_enrollment <- current_FH_test$age[i]
      t_enroll <- as.integer(age_at_enrollment - 29)
      
      
      if (t_enroll < 0 || t_enroll >= dim(current_Y_test)[3]) {
        n_age_filtered <- n_age_filtered + 1
        next
      }
      
      # Prevalent disease check
      if (length(disease_indices) == 1 && t_enroll > 0) {
        if (any(current_Y_test[i, disease_indices, 1:(t_enroll-1)] == 1)) {
          n_prevalent <- n_prevalent + 1
          next
        }
      }
      
      end_time <- min(t_enroll + follow_up_duration_years, dim(current_Y_test)[3])
      ymat <- current_Y_test[i, disease_indices, t_enroll:end_time, drop = TRUE]
      # Find all event indices (relative to t_enroll)
      if (length(disease_indices) == 1) {
        # ymat is a vector
        event_ages <- which(ymat == 1)
      } else {
        # ymat is a matrix
        event_ages <- which(ymat == 1, arr.ind = TRUE)[, 2]
      }
      
      if (length(event_ages) == 0) {
        age_at_event <- end_time + 29 - 1
        event <- 0
      } else {
        min_event_idx <- min(event_ages)
        age_at_event <- t_enroll + min_event_idx + 29 - 1
        event <- 1
      }
      age_enroll <- t_enroll + 29
      if (age_enroll >= age_at_event) {
        n_no_followup <- n_no_followup + 1
        next
      }
      row <- data.frame(
        age_enroll = age_enroll,
        age = age_at_event,
        event = event,
        sex = current_FH_test$sex[i]
      )
      if (length(fh_cols) > 0 && all(fh_cols %in% colnames(current_FH_test))) {
        row$fh <- any(current_FH_test[i, fh_cols])
      }
      if (!is.null(pi_test)) {
        # Get the noulli prediction for this disease group
        pi_diseases <- current_pi_test[i, disease_indices, t_enroll]
        yearly_risk <- 1 - prod(1 - pi_diseases)  # Convert to yearly risk
        row$noulli_risk <- yearly_risk
      }
      
      cox_data <- rbind(cox_data, row)
    }
    
    print(paste("Excluded due to age:", n_age_filtered))
    print(paste("Excluded due to prevalent disease:", n_prevalent))
    print(paste("Excluded due to no follow-up:", n_no_followup))
    print(paste("Final nrow(cox_data):", nrow(cox_data)))
    
    print(paste("First few rows of data"))
    print(head(cox_data))
    print(paste("Total events before filtering:", sum(current_Y_test[, disease_indices, ] == 1)))
    print(paste("Total events after filtering:", sum(cox_data$event)))
    
    fit <- fitted_models[[disease_group]]
    if (is.null(fit) || nrow(cox_data) == 0) next
    
    # Predict linear predictor (risk score)
    risk_score <- predict(fit, newdata = cox_data, type = "lp")
    
    
    
    # Calculate AUC (using pROC)
    roc_obj <- roc(cox_data$event, risk_score)
    auc_val <- auc(roc_obj)
    print(sprintf("AUC for %s: %.3f", disease_group, auc_val))
    auc_results[[disease_group]] <- auc_val
    
    
    #surv_obj <- Surv(cox_data$age_enroll,cox_data$age, cox_data$event)
    surv_obj <- Surv(time = cox_data$age - cox_data$age_enroll, cox_data$event)
    print(surv_obj)
    print("risk_score")
    print(risk_score)
    concordance_result <- concordance(surv_obj ~ risk_score,reverse=TRUE)
    c_index <-concordance_result$concordance 
    print(sprintf("C-index for %s: %.3f", disease_group, c_index))
    concordance_results[[disease_group]] <- c_index
  }
  return(list(auc_results,concordance_results))
}







test_indices=0:10000
auc_results = test_cox_baseline_models(
  Y_test = Y_test,
  FH_processed = FH_processed,
  test_indices = test_indices,
  disease_mapping = disease_mapping,
  major_diseases = major_diseases,
  disease_names = disease_names,
  follow_up_duration_years = 10,
  fitted_models = cb,
  #pi_test=pi_test
)


auc_df <- data.frame(
  disease_group = names(auc_results[[1]]),
  auc = unlist(auc_results[[1]])
)

write.csv(auc_df,"~/Library/CloudStorage/Dropbox/auc_results_cox_20000_30000train_0_10000test_1121.csv",quote = FALSE)


concordance_df <- data.frame(
  disease_group = names(auc_results[[1]]),
  concordance = unlist(auc_results[[1]])
)

write.csv(concordance_df,"~/Library/CloudStorage/Dropbox/concordance_results_cox_20000_30000train_0_10000test_112.csv",quote = FALSE)

#### 

Y_train=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/ukb_Y_train.rds")
FH_processed = read.csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh.csv')

#pi_train=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pi_enroll_sex_20000_30000.rds")
#pi_train=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/pi_enroll_sex_20000_30000.rds")
#pi_train=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_ENROLLMENT_pooled/pi_enroll_fixedphi_sex_20000_30000.rds")

pi_train=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_prediction_jointphi_sex_pcs/pi_enroll_sex_20000_30000.rds")

cb = fit_cox_baseline_models(
  Y_train = Y_train,
  FH_processed = FH_processed,
  
  train_indices = 20001:30000,
  disease_mapping = disease_mapping,
  major_diseases = major_diseases,
  disease_names = disease_names,
  follow_up_duration_years = 10,
  pi_train=pi_train
)


rm(Y_train)
rm(pi_train)
rm(train_indices)

### 

Y_test=readRDS("/Users/sarahurbut/aladynoulli2/pyScripts/ukb_Y_test.rds")


## to do: obtain pi_enroll_test from latest fitted noulli model
#pi_test=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pi_enroll_sex_0_10000.rds")
#pi_test=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/pi_enroll_fixedphi_sex_0_10000.rds")
#pi_test=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_ENROLLMENT_pooled/pi_enroll_fixedphi_sex_0_10000.rds")
#pi_test=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_prediction_jointphi_sex_pcs/pi_enroll_sex_0_10000.rds")
test_indices=0:10000
auc_results = test_cox_baseline_models(
  Y_test = Y_test,
  FH_processed = FH_processed,
  test_indices = test_indices,
  disease_mapping = disease_mapping,
  major_diseases = major_diseases,
  disease_names = disease_names,
  follow_up_duration_years = 10,
  fitted_models = cb,
  pi_test=pi_test
)


auc_df_with_noulli <- data.frame(
  disease_group = names(auc_results[[1]]),
  auc = unlist(auc_results[[1]])
)


concordance_df_with_noulli <- data.frame(
  disease_group = names(auc_results[[2]]),
  concordance = unlist(auc_results[[2]])
)


write.csv(auc_df_with_noulli,"~/Library/CloudStorage/Dropbox/auc_results_cox_20000_30000train_0_10000test_with_noulli_1121_enrollment_joint.csv",quote = FALSE)

write.csv(auc_df_with_noulli,"~/Library/CloudStorage/Dropbox/auc_results_cox_20000_30000train_0_10000test_with_noulli_1121_retrospective_fixed.csv",quote = FALSE)

write.csv(auc_df_with_noulli,"~/Library/CloudStorage/Dropbox/auc_results_cox_20000_30000train_0_10000test_with_noulli_1121_enrollment_fixed.csv",quote = FALSE)

write.csv(concordance_df_with_noulli,"~/Library/CloudStorage/Dropbox/concordance_results_cox_20000_30000train_0_10000test_with_noulli.csv",quote = FALSE)

###
###
without=fread("~/aladynoulli2/pyScripts//auc_results_cox_20000_30000train_0_10000test.csv")[,c(2,3)]
with=read.csv("~/Library/CloudStorage/Dropbox/auc_results_cox_20000_30000train_0_10000test_with_noulli_1121.csv")[,c(2,3)]
with2=read.csv("~/aladynoulli2/pyScripts//auc_results_cox_20000_30000train_0_10000test_with_noulli.csv")[,c(2,3)]
with3=read.csv("~/Library/CloudStorage/Dropbox/auc_results_cox_20000_30000train_0_10000test_with_noulli_1121_enrollment_fixed.csv")[,c(2,3)]
newcox=merge(without,with,by="disease_group")
newcox3=merge(with,merge(with3,with2,by="disease_group"),by="disease_group")
names(newcox3)=c("disease","pooled_fixed_all","pooled_fix_enroll","old_joint")
newcox2=merge(without,with2,by="disease_group")

newcox3=merge(with,with2,by="disease_group")
colnames(newcox)=colnames(newcox2)=c("disease_group","auc_without_noulli","auc_with_noulli")


ggplot(newcox2, aes(x = auc_without_noulli, y = auc_with_noulli, color = as.factor(disease_group))) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray60", linewidth = 0.7, alpha = 0.7) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_viridis_d(name = "Disease\nGroup", option = "plasma") +
  coord_fixed() +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  ) +
  labs(
    x = "AUC: Cox without Noulli",
    y = "AUC: Cox with Noulli",
    title = "Cox Model AUC Comparison: With vs. Without Noulli Risk Score"
  )

ggplot(newcox2, aes(x = auc_without_noulli, y = auc_with_noulli, color = as.factor(disease_group))) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray60", linewidth = 0.7, alpha = 0.7) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_viridis_d(name = "Disease\nGroup", option = "plasma") +
  coord_fixed() +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  ) +
  labs(
    x = "AUC: Cox without Noulli",
    y = "AUC: Cox with Noulli",
    title = "Cox Model AUC Comparison: With vs. Without Noulli Risk Score"
  )


## combined all the aucs and no noulli
write.csv(mg,"allwithnoullipooledoldcox1121.csv",row.names = FALSE,quote = FALSE)

ggplot(newcox, aes(x = auc_without_noulli, y = auc_with_noulli, color = as.factor(disease_group))) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray60", linewidth = 0.7, alpha = 0.7) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_viridis_d(name = "Disease\nGroup", option = "plasma") +
  coord_fixed() +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  ) +
  labs(
    x = "AUC: Cox without Noulli",
    y = "AUC: Cox with Noulli",
    title = "Cox Model AUC Comparison: With vs. Without Noulli Risk Score NEW"
  )