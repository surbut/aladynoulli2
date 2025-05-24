# cox_utils.R

### here we load some stuff##

library(survival)
library(broom)
library(dplyr)

FH_processed = read.csv('/Users/sarahurbut/Library/CloudStorage/Dropbox/baselinagefamh.csv')

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

Y_train=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox/ukb_Y_train.rds")
FH_processed = read.csv('/Users/sarahurbut/Library/CloudStorage/Dropbox/baselinagefamh.csv')

pi_train=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_enroll_sex_20000_30000.rds")


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
  }
  return(auc_results)
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
disease_group = names(auc_results),
auc = unlist(auc_results)
)

write.csv(auc_df,"~/Library/CloudStorage/Dropbox/auc_results_cox_20000_30000train_0_10000test.csv",quote = FALSE)

#### 

Y_train=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox/ukb_Y_train.rds")
FH_processed = read.csv('/Users/sarahurbut/Library/CloudStorage/Dropbox/baselinagefamh.csv')

pi_train=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_enroll_sex_20000_30000.rds")


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

pi_test=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_enroll_sex_0_10000.rds")
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
disease_group = names(auc_results),
auc = unlist(auc_results)
)


write.csv(auc_df_with_noulli,"~/Library/CloudStorage/Dropbox/auc_results_cox_20000_30000train_0_10000test_with_noulli.csv",quote = FALSE)

dynamic_model_results=read.csv("model_comparison_results_dynamic_vcoxauc.csv")[,c(1,2,4,5)]
names(dynamic_model_results)[2]="aladynoulli_auc_dynamic"

static_noulli=read.csv("model_comparison_results_bootstatic_auc.csv")[,c(1,2)]
names(static_noulli)[2]="aladynoulli_auc_static"

cox_results_without=read.csv("~/Library/CloudStorage/Dropbox/auc_results_cox_20000_30000train_0_10000test.csv")[,c(2,3)]
names(cox_results_without)[1]="Disease"
names(cox_results_without)[2]="cox_auc_without_noulli"

cox_results_with_noulli=read.csv("~/Library/CloudStorage/Dropbox/auc_results_cox_20000_30000train_0_10000test_with_noulli.csv")[,c(2,3)]
names(cox_results_with_noulli)[1]="Disease"
names(cox_results_with_noulli)[2]="cox_auc_with_noulli"


m1=merge(dynamic_model_results,static_noulli,by="Disease",all=TRUE)
m2=merge(m1,cox_results_without,by="Disease",all=TRUE)
m3=merge(m2,cox_results_with_noulli,by="Disease",all=TRUE)


write.csv(m3,"~/Library/CloudStorage/Dropbox/model_comparison_everything.csv",quote = FALSE)







install.packages(c("ggplot2", "dplyr", "tidyr", "stringr", "forcats"))








# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(forcats)

# Function to parse the data
parse_model_data <- function(data_path) {
  # Read the data
  df <- read.csv(data_path)
  
  # Extract confidence intervals from the formatted strings
  extract_ci <- function(ci_str) {
    # Matches format like "0.765 (0.748-0.782)"
    pattern <- "(\\d+\\.\\d+)\\s*\\((\\d+\\.\\d+)-(\\d+\\.\\d+)\\)"
    matches <- regexec(pattern, ci_str)
    result <- regmatches(ci_str, matches)
    
    if(length(result) > 0 && length(result[[1]]) >= 4) {
      return(c(as.numeric(result[[1]][2]), 
               as.numeric(result[[1]][3]),
               as.numeric(result[[1]][4])))
    } else {
      return(c(NA, NA, NA))
    }
  }
  
  # Create a clean data frame
  model_data <- data.frame(
    Disease = df$Disease,
    Events = df$Events,
    Rate = df$Rate,
    stringsAsFactors = FALSE
  )
  
  # Extract values for Aladynoulli Dynamic
  dynamic_ci <- lapply(df$aladynoulli_auc_dynamic, extract_ci)
  model_data$dynamic_auc <- sapply(dynamic_ci, function(x) x[1])
  model_data$dynamic_lower <- sapply(dynamic_ci, function(x) x[2])
  model_data$dynamic_upper <- sapply(dynamic_ci, function(x) x[3])
  
  # Extract values for Aladynoulli Static
  static_ci <- lapply(df$aladynoulli_auc_static, extract_ci)
  model_data$static_auc <- sapply(static_ci, function(x) x[1])
  model_data$static_lower <- sapply(static_ci, function(x) x[2])
  model_data$static_upper <- sapply(static_ci, function(x) x[3])
  
  # Add Cox values
  model_data$cox_without_noulli <- df$cox_auc_without_noulli
  model_data$cox_with_noulli <- df$cox_auc_with_noulli
  
  # Calculate CI for Cox values (using the formula SE = sqrt((AUC * (1-AUC)) / (n * prevalence)))
  calc_cox_ci <- function(auc, events, total = 400000) {
    # Parse rate from string like "4.8%"
    prevalence <- events / total
    # Ensure minimum prevalence for calculation
    prevalence <- max(0.01, prevalence)
    
    # Standard error
    se <- sqrt((auc * (1 - auc)) / (total * prevalence))
    # 95% CI
    lower <- max(0, auc - 1.96 * se)
    upper <- min(1, auc + 1.96 * se)
    
    return(c(lower, upper))
  }
  
  # Add CI for Cox without Noulli
  cox_without_ci <- mapply(calc_cox_ci, 
                           model_data$cox_without_noulli, 
                           model_data$Events, 
                           SIMPLIFY = FALSE)
  model_data$cox_without_lower <- sapply(cox_without_ci, function(x) x[1])
  model_data$cox_without_upper <- sapply(cox_without_ci, function(x) x[2])
  
  # Add CI for Cox with Noulli
  cox_with_ci <- mapply(calc_cox_ci, 
                        model_data$cox_with_noulli, 
                        model_data$Events, 
                        SIMPLIFY = FALSE)
  model_data$cox_with_lower <- sapply(cox_with_ci, function(x) x[1])
  model_data$cox_with_upper <- sapply(cox_with_ci, function(x) x[2])
  
  return(model_data)
}

# Reshape data for plotting
prepare_forest_plot_data <- function(model_data) {
  # Transform to long format
  long_data <- model_data %>%
    pivot_longer(
      cols = c("dynamic_auc", "static_auc", "cox_without_noulli", "cox_with_noulli"),
      names_to = "model",
      values_to = "auc"
    ) %>%
    mutate(
      lower = case_when(
        model == "dynamic_auc" ~ dynamic_lower,
        model == "static_auc" ~ static_lower,
        model == "cox_without_noulli" ~ cox_without_lower,
        model == "cox_with_noulli" ~ cox_with_lower
      ),
      upper = case_when(
        model == "dynamic_auc" ~ dynamic_upper,
        model == "static_auc" ~ static_upper,
        model == "cox_without_noulli" ~ cox_without_upper,
        model == "cox_with_noulli" ~ cox_with_upper
      ),
      model = case_when(
        model == "dynamic_auc" ~ "Aladynoulli Dynamic",
        model == "static_auc" ~ "Aladynoulli Static",
        model == "cox_without_noulli" ~ "Cox without Noulli",
        model == "cox_with_noulli" ~ "Cox with Noulli"
      )
    ) %>%
    select(Disease, Events, Rate, model, auc, lower, upper)
  
  # Make Disease a factor ordered by dynamic AUC
  disease_order <- model_data %>%
    arrange(desc(dynamic_auc)) %>%
    pull(Disease)
  
  long_data$Disease <- factor(long_data$Disease, levels = disease_order)
  
  return(long_data)
}

# Create the forest plot
create_forest_plot <- function(plot_data) {
  # Define colors for models
  model_colors <- c(
    "Aladynoulli Dynamic" = "#4285F4",  # Blue
    "Aladynoulli Static" = "#34A853",   # Green
    "Cox without Noulli" = "#FBBC05",   # Yellow/Orange
    "Cox with Noulli" = "#EA4335"       # Red
  )
  
  # Shape the disease names to be more readable
  plot_data <- plot_data %>%
    mutate(Disease = str_replace_all(Disease, "_", " "))
  
  # Create the plot
  p <- ggplot(plot_data, aes(x = auc, y = Disease, color = model)) +
    geom_vline(xintercept = 0.5, linetype = "dashed", color = "gray70") +
    geom_point(aes(shape = model), size = 2.5, position = position_dodge(width = 0.5)) +
    geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.2, position = position_dodge(width = 0.5)) +
    scale_color_manual(values = model_colors) +
    scale_x_continuous(limits = c(0.3, 1), breaks = seq(0.3, 1, by = 0.1)) +
    labs(
      title = "Multi-Disease AUC Comparison",
      subtitle = "Comparing performance of four prediction models across 28 diseases",
      x = "AUC (95% CI)",
      y = NULL,
      color = "Model",
      shape = "Model"
    ) +
    theme_minimal() +
    theme(
      legend.position = "top",
      panel.grid.major.y = element_line(color = "gray90"),
      panel.grid.minor.y = element_blank(),
      axis.text.y = element_text(size = 10),
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 12)
    )
  
  return(p)
}

# Main function to run the analysis
create_multidisease_comparison <- function(data_path) {
  # Parse the data
  model_data <- parse_model_data(data_path)
  
  # Prepare data for plotting
  plot_data <- prepare_forest_plot_data(model_data)
  
  # Create forest plot
  forest_plot <- create_forest_plot(plot_data)
  
  # Save the plot
  ggsave("multidisease_forest_plot.png", forest_plot, width = 12, height = 14, dpi = 300)
  
  # Print summary statistics
  model_summary <- plot_data %>%
    group_by(model) %>%
    summarize(
      mean_auc = mean(auc, na.rm = TRUE),
      median_auc = median(auc, na.rm = TRUE),
      min_auc = min(auc, na.rm = TRUE),
      max_auc = max(auc, na.rm = TRUE)
    )
  
  print(model_summary)
  
  return(forest_plot)
}

# Example usage:
# forest_plot <- create_multidisease_comparison("model_comparison_everything.csv")
# print(forest_plot)

# Alternative approach with facet_wrap for separate panels by disease
create_faceted_forest_plot <- function(plot_data) {
  # Limit to top 16 diseases by dynamic AUC
  top_diseases <- plot_data %>%
    filter(model == "Aladynoulli Dynamic") %>%
    arrange(desc(Events)) %>%
    head(16) %>%
    pull(Disease)
  
  plot_data_subset <- plot_data %>%
    filter(Disease %in% top_diseases)

  
  plot_data_subset$Disease=factor(plot_data_subset$Disease,levels = top_diseases)
  # Define colors for models
  model_colors <- c(
    "Aladynoulli Dynamic" = "#4285F4",  # Blue
    "Aladynoulli Static" = "#34A853",   # Green
    "Cox without Noulli" = "#FBBC05",   # Yellow/Orange
    "Cox with Noulli" = "#EA4335"       # Red
  )
  
  # Create the faceted plot
  p <- ggplot(plot_data_subset, aes(x = model, y = auc, color = model)) +
    geom_point(size = 5) +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.1) +
    facet_wrap(~ Disease, scales = "free_y", ncol = 4) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray70") +
    scale_color_manual(values = model_colors) +
    scale_y_continuous(limits = c(0.3, 1), breaks = seq(0.3, 1, by = 0.1)) +
    labs(
      title = "Multi-Disease AUC Comparison",
      subtitle = "Performance across top 16 diseases",
      y = "AUC (95% CI)",
      x = NULL
    ) +
    theme_minimal() +
    theme(
      legend.position = "top",
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
      strip.text = element_text(face = "bold"),
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 12)
    )
  
  return(p)
}


model_data <- parse_model_data("~/Library/CloudStorage/Dropbox/model_comparison_everything.csv")
plot_data <- prepare_forest_plot_data(model_data)

# Add this line to also generate the faceted plot
faceted_plot <- create_faceted_forest_plot(plot_data)
ggsave("multidisease_faceted_plot.pdf", faceted_plot, width = 12, height = 12, dpi = 300)



### do a dynamic Cox with pi_full
pi_full=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_full_sex_0_10000.rds")
