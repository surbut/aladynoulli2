# cox_utils.R

library(survival)
library(broom)
library(dplyr)

FH_processed = read.csv('/Users/sarahurbut/Library/CloudStorage/Dropbox/baselinagefamh.csv')
Y_train = readRDS("~/aladynoulli2/pyScripts/Y_train_tensor.rds")


###
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
                                    follow_up_duration_years = 10) {
  fitted_models <- list()
  
  
  FH_train <- FH_processed[train_indices, ]
  
  disease_group = "ASCVD"
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
    print(dim(FH_train))
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
    if (length(disease_indices) == 0) {
      fitted_models[[disease_group]] <- NULL
      next
    }
    print(disease_names[sort(disease_indices)])
    # Prepare data for Cox model
    cox_data <- data.frame()
    for (i in seq_len(nrow(current_FH_train))) {
      print(i)
      age_at_enrollment <- current_FH_train$age[i]
      t_enroll <- as.integer(age_at_enrollment - 29)
      if (t_enroll < 0 || t_enroll >= dim(current_Y_train)[3])
        next
      
      end_time <- min(t_enroll + follow_up_duration_years,
                      dim(current_Y_train)[3])
      #if (end_time <= t_enroll)
      # next
      
      for (d_idx in disease_indices) {
        Y_slice <- current_Y_train[i, d_idx, t_enroll:end_time, drop = TRUE]
        if (any(Y_slice > 0, na.rm = TRUE)) {
          event_time <- which(Y_slice > 0)[1] + t_enroll - 1
          age_at_event <- 29 + event_time
          event <- 1
        } else {
          age_at_event <- 29 + end_time - 1
          event <- 0
        }
        row <- data.frame(age_enroll=t_enroll+29,
                          age = age_at_event,
                          event = event,
                          sex = current_FH_train$sex[i])
        if (length(fh_cols) > 0 &&
            all(fh_cols %in% colnames(current_FH_train))) {
          row$fh <- any(current_FH_train[i, fh_cols])
        }
        cox_data <- rbind(cox_data, row)
      }
    }
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
    fit <- try(coxph(as.formula(formula_str), data = cox_data), silent =
                 TRUE)
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

# --- Function to Evaluate Cox Models on Test Set ---
evaluate_cox_baseline_models <- function(fitted_models,
                                         Y_test,
                                         FH_test,
                                         disease_mapping,
                                         major_diseases,
                                         disease_names,
                                         follow_up_duration_years = 10) {
  test_results <- list()
  cat("\nEvaluating Cox models on test data...\n")
  
  FH_test <- as.data.frame(FH_test)
  
  for (disease_group in names(fitted_models)) {
    model <- fitted_models[[disease_group]]
    if (is.null(model)) {
      test_results[[disease_group]] <- list(
        c_index = NA,
        ci = c(NA, NA),
        n_events = 0,
        n_total = 0
      )
      next
    }
    cat(sprintf(" - Evaluating %s...\n", disease_group))
    fh_cols <- disease_mapping[[disease_group]]
    if (is.null(fh_cols))
      fh_cols <- character(0)
    
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
    current_FH_test <- FH_test[mask_test, ]
    current_Y_test <- Y_test[mask_test, , , drop = FALSE]
    if (nrow(current_FH_test) == 0) {
      cat(sprintf(
        "   Warning: No individuals for target sex code %s.\n",
        target_sex_code
      ))
      test_results[[disease_group]] <- list(
        c_index = NA,
        ci = c(NA, NA),
        n_events = 0,
        n_total = 0
      )
      next
    }
    disease_indices <- unlist(lapply(major_diseases[[disease_group]], function(disease) {
      which(tolower(disease_names) == tolower(disease))
    }))
    if (length(disease_indices) == 0) {
      test_results[[disease_group]] <- list(
        c_index = NA,
        ci = c(NA, NA),
        n_events = 0,
        n_total = 0
      )
      next
    }
    eval_data <- data.frame()
    for (i in seq_len(nrow(current_FH_test))) {
      age_at_enrollment <- current_FH_test$age[i]
      t_enroll <- as.integer(age_at_enrollment - 29)
      if (t_enroll < 0 || t_enroll >= dim(current_Y_test)[3])
        next
      end_time <- min(t_enroll + follow_up_duration_years,
                      dim(current_Y_test)[3])
      if (end_time <= t_enroll)
        next
      for (d_idx in disease_indices) {
        Y_slice <- current_Y_test[i, d_idx, t_enroll:end_time, drop = TRUE]
        if (any(Y_slice > 0, na.rm = TRUE)) {
          event_time <- which(Y_slice > 0)[1] + t_enroll - 1
          age_at_event <- 29 + event_time
          event <- 1
        } else {
          age_at_event <- 29 + end_time - 1
          event <- 0
        }
        row <- data.frame(age = age_at_event,
                          event = event,
                          sex = current_FH_test$sex[i])
        if (length(fh_cols) > 0 &&
            all(fh_cols %in% colnames(current_FH_test))) {
          row$fh <- any(data.frame(current_FH_test[i, fh_cols]))
        }
        eval_data <- rbind(eval_data, row)
      }
    }
    if (nrow(eval_data) == 0) {
      cat("   Warning: No individuals processed for evaluation.\n")
      test_results[[disease_group]] <- list(
        c_index = NA,
        ci = c(NA, NA),
        n_events = 0,
        n_total = 0
      )
      next
    }
    # Get predicted risk scores
    risk_scores <- predict(model, newdata = eval_data, type = "risk")
    eval_data$risk_scores = risk_scores
    # Calculate concordance index
    c_index <- survConcordance(Surv(age, event) ~ -1 * risk_scores, data =
                                 eval_data)$concordance
    # Bootstrap confidence interval
    n_bootstraps <- 10
    c_indices <- numeric(n_bootstraps)
    for (b in 1:n_bootstraps) {
      idx <- sample(seq_len(nrow(eval_data)), replace = TRUE)
      boot_eval <- eval_data[idx, ]
      boot_risk <- predict(model, newdata = boot_eval, type = "risk")
      c_indices[b] <- survConcordance(Surv(age, event) ~ -1 * boot_risk, data =
                                        boot_eval)$concordance
    }
    ci_lower <- quantile(c_indices, 0.025, na.rm = TRUE)
    ci_upper <- quantile(c_indices, 0.975, na.rm = TRUE)
    n_events <- sum(eval_data$event)
    n_total <- nrow(eval_data)
    test_results[[disease_group]] <- list(
      c_index = c_index,
      ci = c(ci_lower, ci_upper),
      n_events = n_events,
      n_total = n_total
    )
    cat(sprintf("   C-index: %.3f (%.3f-%.3f)\n", c_index, ci_lower, ci_upper))
    cat(sprintf("   Events: %d/%d\n", n_events, n_total))
  }
  cat("Finished evaluating Cox models.\n")
  return(test_results)
}

cb = fit_cox_baseline_models(
  Y_train = Y_train,
  FH_processed = FH_processed,
  
  train_indices = 20001:30000,
  disease_mapping = disease_mapping,
  major_diseases = major_diseases,
  disease_names = disease_names,
  follow_up_duration_years = 10
)


Y_test = readRDS("big_stuff/ukb_params.rds")$Y

e = evaluate_cox_baseline_models(
  cb,
  Y_test,
  FH_processed[1:10000, ],
  disease_mapping = disease_mapping,
  major_diseases = major_diseases,
  disease_names = disease_names,
  follow_up_duration_years = 10
)
