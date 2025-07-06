


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



# Function for time-dependent Cox modeling
fit_time_dependent_cox <- function(Y_train,
                                 FH_processed,
                                 train_indices,
                                 disease_mapping,
                                 major_diseases,
                                 disease_names,
                                 follow_up_duration_years = 10,
                                 pi_train = NULL) {
  fitted_models <- list()
  FH_train <- FH_processed[train_indices, ]
  
  for (disease_group in names(major_diseases)) {
    fh_cols <- disease_mapping[[disease_group]]
    if (is.null(fh_cols)) fh_cols <- character(0)
    if (length(fh_cols) == 0)
      cat(sprintf(" - %s: No FH columns, fitting Sex only.\n", disease_group))
    cat(sprintf(" - Fitting time-dependent Cox for %s...\n", disease_group))
    
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
    
    if (!is.null(pi_train)) {
      current_pi_train <- pi_train[mask_train, , , drop = FALSE]
    } else {
      current_pi_train <- NULL
    }
    
    if (nrow(current_FH_train) == 0) {
      cat(sprintf("   Warning: No individuals for target sex code %s in training slice.\n", target_sex_code))
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
    
    # Pre-calculate total number of rows needed
    total_rows <- 0
    print("Calculating total rows needed...")
    start_time <- Sys.time()
    for (i in seq_len(nrow(current_FH_train))) {
      age_at_enrollment <- current_FH_train$age[i]
      t_enroll <- as.integer(age_at_enrollment - 29)
      if (t_enroll < 0 || t_enroll >= dim(current_Y_train)[3]) next
      if (length(disease_indices) == 1 && t_enroll > 0) {
        if (any(current_Y_train[i, disease_indices, 1:(t_enroll-1)] == 1)) next
      }
      end_time <- min(t_enroll + follow_up_duration_years, dim(current_Y_train)[3])
      
      # Check for events to determine actual number of rows needed
      event_found <- FALSE
      for (t in t_enroll:(end_time-1)) {
        ymat <- current_Y_train[i, disease_indices, t:(t+1), drop = TRUE]
        event <- if (length(disease_indices) == 1) {
          any(ymat == 1)
        } else {
          any(ymat == 1)
        }
        total_rows <- total_rows + 1
        if (event) {
          event_found <- TRUE
          break
        }
      }
    }
    print(paste("Total rows needed:", total_rows))
    print(paste("Time for row calculation:", Sys.time() - start_time))
    
    # Pre-allocate data frame
    print("Pre-allocating data frame...")
    start_time <- Sys.time()
    tdc_data <- data.frame(
      id = integer(total_rows),
      start = numeric(total_rows),
      stop = numeric(total_rows),
      event = logical(total_rows),
      sex = integer(total_rows)
    )
    if (length(fh_cols) > 0 && all(fh_cols %in% colnames(current_FH_train))) {
      tdc_data$fh <- logical(total_rows)
    }
    if (!is.null(current_pi_train)) {
      tdc_data$noulli_risk <- numeric(total_rows)
    }
    print(paste("Time for pre-allocation:", Sys.time() - start_time))
    
    # Fill the data frame
    print("Filling data frame...")
    start_time <- Sys.time()
    row_idx <- 1
    for (i in seq_len(nrow(current_FH_train))) {
      if (i == 1) {
        print("First person details:")
        print(paste("Age at enrollment:", current_FH_train$age[i]))
        print(paste("t_enroll:", as.integer(current_FH_train$age[i] - 29)))
        if (!is.null(current_pi_train)) {
          pi_diseases <- current_pi_train[i, disease_indices, as.integer(current_FH_train$age[i] - 29)]
          print("Pi values for diseases:")
          print(pi_diseases)
          yearly_risk <- 1 - prod(1 - pi_diseases)
          print(paste("Yearly risk:", yearly_risk))
        }
      }
      
      if (i %% 1000 == 0) {
        print(paste("Processing individual", i, "of", nrow(current_FH_train)))
        print(paste("Time so far:", Sys.time() - start_time))
      }
      
      age_at_enrollment <- current_FH_train$age[i]
      t_enroll <- as.integer(age_at_enrollment - 29)
      
      if (t_enroll < 0 || t_enroll >= dim(current_Y_train)[3]) next
      
      # Prevalent disease check
      if (length(disease_indices) == 1 && t_enroll > 0) {
        if (any(current_Y_train[i, disease_indices, 1:(t_enroll-1)] == 1)) next
      }
      
      end_time <- min(t_enroll + follow_up_duration_years, dim(current_Y_train)[3])
      
      # Create time intervals for each year
      for (t in t_enroll:(end_time-1)) {
        # Check if event occurred in this interval
        ymat <- current_Y_train[i, disease_indices, t:(t+1), drop = TRUE]
        event <- if (length(disease_indices) == 1) {
          any(ymat == 1)
        } else {
          any(ymat == 1)
        }
        
        # Debug prints for events
        if (event) {
          print(paste("Found event for person", i, "at age", t + 30))
          print("Event matrix:")
          print(ymat)
          print("Disease indices:")
          print(disease_indices)
        }
        
        # Fill row
        tdc_data$id[row_idx] <- i
        tdc_data$start[row_idx] <- t + 29
        tdc_data$stop[row_idx] <- t + 30
        tdc_data$event[row_idx] <- event
        tdc_data$sex[row_idx] <- current_FH_train$sex[i]
        
        if (length(fh_cols) > 0 && all(fh_cols %in% colnames(current_FH_train))) {
          tdc_data$fh[row_idx] <- any(current_FH_train[i, fh_cols])
        }
        
        if (!is.null(current_pi_train)) {
          pi_diseases <- current_pi_train[i, disease_indices, t]
          
          stop(sprintf("NaN detected in pi_diseases for person %d (age at enrollment: %d, t_enroll: %d, year: %d)", 
                       i, current_FH_train$age[i], t_enroll, t))
          
          yearly_risk <- 1 - prod(1 - pi_diseases)
          tdc_data$noulli_risk[row_idx] <- yearly_risk
        }
        
        row_idx <- row_idx + 1
        
        # Stop if event occurred
        if (event) {
          print(paste("Stopping after event for person", i))
          break
        }
      }
    }
    print(paste("Total time for filling:", Sys.time() - start_time))
    
    # Trim any unused rows
    print(paste("Final row count:", row_idx-1))
    print("Trimming unused rows...")
    start_time <- Sys.time()
    tdc_data <- tdc_data[1:(row_idx-1), ]
    print(paste("Time for trimming:", Sys.time() - start_time))
    print("Data frame complete")
    print(paste("Number of events:", sum(tdc_data$event)))
    print(paste("Number of unique individuals:", length(unique(tdc_data$id))))
    print(paste("Average rows per person:", nrow(tdc_data)/length(unique(tdc_data$id))))
    print(paste("Number of people with events:", sum(tdc_data$event)))
    print(paste("Proportion of people with events:", sum(tdc_data$event)/length(unique(tdc_data$id))))
    
    if (nrow(tdc_data) == 0 || sum(tdc_data$event) < 5) {
      cat(sprintf("   Warning: Too few events (%d) for %s\n", sum(tdc_data$event), disease_group))
      fitted_models[[disease_group]] <- NULL
      next
    }
    
    # Fit time-dependent Cox model
    formula_str <- "Surv(start, stop, event) ~ sex"
    if ("fh" %in% colnames(tdc_data))
      formula_str <- "Surv(start, stop, event) ~ sex + fh"
    if (!is.na(target_sex_code)) {
      formula_str <- if ("fh" %in% colnames(tdc_data))
        "Surv(start, stop, event) ~ fh"
      else
        "Surv(start, stop, event) ~ 1"
    }
    if (!is.null(pi_train) && "noulli_risk" %in% colnames(tdc_data))
      formula_str <- paste(formula_str, "+ noulli_risk")
    
    print(formula_str)
    fit <- try(coxph(as.formula(formula_str), data = tdc_data, id = id), silent = TRUE)
    
    ## can we just do the concordance here?
    
    if (inherits(fit, "try-error")) {
      cat(sprintf("   Error fitting %s: %s\n", disease_group, fit))
      fitted_models[[disease_group]] <- NULL
    } else {
      fitted_models[[disease_group]] <- fit
      cat(sprintf("   Model fitted for %s using %d time intervals.\n", disease_group, nrow(tdc_data)))
    }
  }
  
  return(fitted_models)
}


#### Testing function
# Function to evaluate time-dependent Cox models
test_time_dependent_cox <- function(Y_test,
                                  FH_processed,
                                  test_indices,
                                  disease_mapping,
                                  major_diseases,
                                  disease_names,
                                  follow_up_duration_years = 10,
                                  fitted_models,
                                  pi_test = NULL) {
  auc_results <- list()
  concordance_results=list()
  tdc_data_list <- list()  # New list to store tdc_data for each disease
  FH_test <- FH_processed[test_indices, ]
  
  for (disease_group in names(major_diseases)) {
    fh_cols <- disease_mapping[[disease_group]]
    if (is.null(fh_cols)) fh_cols <- character(0)
    if (length(fh_cols) == 0)
    cat(sprintf(" - %s: No FH columns, evaluating Sex only.\n", disease_group))
    cat(sprintf(" - Evaluating time-dependent Cox for %s...\n", disease_group))
    
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
    
    if (!is.null(pi_test)) {
      current_pi_test <- pi_test[mask_test, , , drop = FALSE]
    } else {
      current_pi_test <- NULL
    }
    
    if (nrow(current_FH_test) == 0) {
      cat(sprintf("   Warning: No individuals for target sex code %s in testing slice.\n", target_sex_code))
      next
    }
    
    disease_indices <- unlist(lapply(major_diseases[[disease_group]], function(disease) {
      which(tolower(disease_names) == tolower(disease))
    }))
    
    if (length(disease_indices) == 0) next
    
    # Create time-dependent data structure for testing
    print("Calculating total rows needed...")
    start_time <- Sys.time()
    total_rows <- 0
    for (i in seq_len(nrow(current_FH_test))) {
      age_at_enrollment <- current_FH_test$age[i]
      t_enroll <- as.integer(age_at_enrollment - 29)
      if (t_enroll < 0 || t_enroll >= dim(current_Y_test)[3]) next
      if (length(disease_indices) == 1 && t_enroll > 0) {
        if (any(current_Y_test[i, disease_indices, 1:(t_enroll-1)] == 1)) next
      }
      end_time <- min(t_enroll + follow_up_duration_years, dim(current_Y_test)[3])
      # Check for events to determine actual number of rows needed
      for (t in t_enroll:(end_time-1)) {
        ymat <- current_Y_test[i, disease_indices, t:(t+1), drop = TRUE]
        event <- if (length(disease_indices) == 1) {
          any(ymat == 1)
        } else {
          any(ymat == 1)
        }
        total_rows <- total_rows + 1
        if (event) break
      }
    }
    print(paste("Total rows needed:", total_rows))
    print(paste("Time for row calculation:", Sys.time() - start_time))
    
    # Pre-allocate data frame
    print("Pre-allocating data frame...")
    start_time <- Sys.time()
    tdc_data <- data.frame(
      id = integer(total_rows),
      start = numeric(total_rows),
      stop = numeric(total_rows),
      event = logical(total_rows),
      sex = integer(total_rows),
      identifier = integer(total_rows) 
    )
    if (length(fh_cols) > 0 && all(fh_cols %in% colnames(current_FH_test))) {
      tdc_data$fh <- logical(total_rows)
    }

    
    if (!is.null(current_pi_test)) {
      tdc_data$noulli_risk <- numeric(total_rows)
    }
    print(paste("Time for pre-allocation:", Sys.time() - start_time))
    
    # Fill the data frame
    print("Filling data frame...")
    start_time <- Sys.time()
    row_idx <- 1
    for (i in seq_len(nrow(current_FH_test))) {
      if (i %% 1000 == 0) {
        print(paste("Processing individual", i, "of", nrow(current_FH_test)))
        print(paste("Time so far:", Sys.time() - start_time))
      }
      age_at_enrollment <- current_FH_test$age[i]
      t_enroll <- as.integer(age_at_enrollment - 29)
      if (t_enroll < 0 || t_enroll >= dim(current_Y_test)[3]) next
      if (length(disease_indices) == 1 && t_enroll > 0) {
        if (any(current_Y_test[i, disease_indices, 1:(t_enroll-1)] == 1)) next
      }
      end_time <- min(t_enroll + follow_up_duration_years, dim(current_Y_test)[3])
      for (t in t_enroll:(end_time-1)) {
        ymat <- current_Y_test[i, disease_indices, t:(t+1), drop = TRUE]
        event <- if (length(disease_indices) == 1) {
          any(ymat == 1)
        } else {
          any(ymat == 1)
        }
        # Debug prints for events
        if (event) {
          print(paste("Found event for person", i, "at age", t + 30))
          print("Event matrix:")
          print(ymat)
          print("Disease indices:")
          print(disease_indices)
        }
        tdc_data$id[row_idx] <- i
        tdc_data$start[row_idx] <- t + 29
        tdc_data$stop[row_idx] <- t + 30
        tdc_data$event[row_idx] <- event
        tdc_data$sex[row_idx] <- current_FH_test$sex[i]
        tdc_data$identifier[row_idx] <- current_FH_test$identifier[i]  # <-- assign here
    #
        if (length(fh_cols) > 0 && all(fh_cols %in% colnames(current_FH_test))) {
          tdc_data$fh[row_idx] <- any(current_FH_test[i, fh_cols])
        }
        if (!is.null(current_pi_test)) {
          pi_diseases <- current_pi_test[i, disease_indices, t]
          if (any(is.na(pi_diseases))) {
            stop(sprintf("NaN detected in pi_diseases for person %d (age at enrollment: %d, t_enroll: %d, year: %d)", 
                         i, current_FH_train$age[i], t_enroll, t))
          }
          
          yearly_risk <- 1 - prod(1 - pi_diseases)
          tdc_data$noulli_risk[row_idx] <- yearly_risk
        }
        row_idx <- row_idx + 1
        if (event) {
          print(paste("Stopping after event for person", i))
          break
        }
      }
      
    }
    print(paste("Total time for filling:", Sys.time() - start_time))
    print(paste("Final row count:", row_idx-1))
    print("Trimming unused rows...")
    start_time <- Sys.time()
    tdc_data <- tdc_data[1:(row_idx-1), ]

  tdc_data$pce_score <- current_FH_test$pce_goff_fuull[match(tdc_data$identifier, current_FH_test$identifier)]
  tdc_data$prevent_score <- current_FH_test$prevent_impute[match(tdc_data$identifier, current_FH_test$identifier)]

 tdc_data_list[[disease_group]] <- tdc_data
    print(paste("Time for trimming:", Sys.time() - start_time))
    print("Data frame complete")
    print(paste("Number of events:", sum(tdc_data$event)))
    print(paste("Number of unique individuals:", length(unique(tdc_data$id))))
    print(paste("Average rows per person:", nrow(tdc_data)/length(unique(tdc_data$id))))
    print(paste("Number of people with events:", sum(tdc_data$event)))
    print(paste("Proportion of people with events:", sum(tdc_data$event)/length(unique(tdc_data$id))))
    
    fit <- fitted_models[[disease_group]]
    if (is.null(fit) || nrow(tdc_data) == 0) next
    
    # Predict risk scores
    risk_scores <- predict(fit, newdata = tdc_data, type = "risk")
    
    
    # Suppose tdc_data has columns: id, start, stop, event, predicted_risk
surv_obj <- with(tdc_data, Surv(start, stop, event))
c_index <- concordance(surv_obj ~ risk_scores, data = tdc_data,reverse=TRUE)$concordance
print(sprintf("Time-dependent C-index: %.3f", c_index))

    # Calculate time-dependent AUC
    # For simplicity, we'll use the average risk score per person### NO we should use the concordance ... 
    person_risks <- aggregate(risk_scores, by = list(id = tdc_data$id), FUN = mean)
    person_events <- aggregate(tdc_data$event, by = list(id = tdc_data$id), FUN = max)
    
    roc_obj <- roc(person_events$x, person_risks$x)
    auc_val <- auc(roc_obj)
    print(sprintf("Time-dependent AUC for %s: %.3f", disease_group, auc_val))
    auc_results[[disease_group]] <- auc_val
    concordance_results[[disease_group]]=c_index
  }
  
    # Return all results including the tdc_data_list
  return(list(auc_results = auc_results, 
              concordance_results = concordance_results,
              tdc_data_list = tdc_data_list))
}

# Example usage:
# Load the time-varying probabilities


Y_train=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox/ukb_Y_train.rds")
Y_test=readRDS("ukb_Y_test")

# Fit time-dependent Cox models
tdc_models <- fit_time_dependent_cox(
  Y_train = Y_train,
  FH_processed = FH_processed,
  train_indices = 20001:30000,
  disease_mapping = disease_mapping,
  major_diseases = major_diseases,
  disease_names = disease_names,
  follow_up_duration_years = 10,
  pi_train = pi_train_full
)



pce_data = readRDS('/Users/sarahurbut/Library/Cloudstorage/Dropbox/pce_df_prevent.rds')

FH_processed2=merge(FH_processed,pce_data[,c("id","pce_goff_fuull","prevent_impute")])
# Evaluate time-dependent Cox models
tdc_auc_results <- test_time_dependent_cox(
  Y_test = Y_test,
  FH_processed = FH_processed2,
  test_indices = 0:10000,
  disease_mapping = disease_mapping,
  major_diseases = major_diseases,
  disease_names = disease_names,
  follow_up_duration_years = 10,
  fitted_models = tdc_models,
  pi_test = pi_test_full
)

# Save results
tdc_auc_df <- data.frame(
  disease_group = names(tdc_auc_results),
  auc = unlist(tdc_auc_results)
)

write.csv(tdc_auc_df, "~/Library/CloudStorage/Dropbox/auc_results_tdc_20000_30000train_0_10000test.csv", quote = FALSE)

model_comp=read.csv("~/Library/CloudStorage/Dropbox/model_comparison_everything.csv")
merge(model_comp,tdc_auc_df,by.x="Disease",by.y="disease_group")


# Calculate time-varying probabilities for training set
library(torch)

# Load the model
model <- torch::load_model("/Users/sarahurbut/Library/CloudStorage/Dropbox/resultshighamp/results/output_0_10000/model.pt")

# Get parameters
lambda_params <- model$model_state_dict$lambda_
phi <- model$model_state_dict$phi
kappa <- model$model_state_dict$kappa

# Convert to numpy arrays
lambda_np <- as.array(lambda_params)
phi_np <- as.array(phi)
kappa_np <- as.array(kappa)

# Calculate theta using softmax
exp_lambda <- exp(lambda_np)
theta <- exp_lambda / rowSums(exp_lambda, dims = 2)

# Calculate phi probabilities using sigmoid
phi_prob <- 1 / (1 + exp(-phi_np))

# Calculate pi for training set (20000-30000)
pi_train_full <- array(0, dim = c(10000, dim(phi_np)[2], dim(lambda_np)[3]))
for (i in 1:10000) {
  idx <- i + 20000  # Adjust index for training set
  pi_train_full[i,,] <- kappa_np * t(theta[idx,,] %*% phi_prob[,,])
}

# Save the calculated probabilities
saveRDS(pi_train_full, "/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_full_sex_20000_30000.rds")

# Now we can use both pi files with the time-dependent Cox model
pi_train_full <- readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_full_sex_20000_30000.rds")
pi_test_full <- readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_full_sex_0_10000.rds")

# Fit time-dependent Cox models
tdc_models <- fit_time_dependent_cox(
  Y_train = Y_train,
  FH_processed = FH_processed,
  train_indices = 20001:30000,
  disease_mapping = disease_mapping,
  major_diseases = major_diseases,
  disease_names = disease_names,
  follow_up_duration_years = 10,
  pi_train = pi_train_full
)

# Evaluate time-dependent Cox models
tdc_auc_results <- test_time_dependent_cox(
  Y_test = Y_test,
  FH_processed = FH_processed,
  test_indices = 0:10000,
  disease_mapping = disease_mapping,
  major_diseases = major_diseases,
  disease_names = disease_names,
  follow_up_duration_years = 10,
  fitted_models = tdc_models,
  pi_test = pi_test_full
)

# Save results
tdc_auc_df <- data.frame(
  disease_group = names(tdc_auc_results[[1]]),
  auc = unlist(tdc_auc_results[[1]])
)

tdc_c_df <- data.frame(
  disease_group = names(tdc_auc_results[[2]]),
  c = unlist(tdc_auc_results[[2]])
)

write.csv(tdc_auc_df, "~/Library/CloudStorage/Dropbox/auc_results_tdc_20000_30000train_0_10000test.csv", quote = FALSE)

write.csv(tdc_c_df, "~/Library/CloudStorage/Dropbox/c_index_results_tdc_20000_30000train_0_10000test.csv", quote = FALSE)


#### 

# Load your data
# df <- read.csv('your_data.csv')  # Replace with your data loading code

# Define predictors
X <- df[, c('sex', 'family_history', 'pce_score', 'prevent_score')]  # Add PCE/prevent scores
y <- df$event  # Your binary outcome

# Fit logistic regression WITHOUT Aladynoulli
model_without_noulli <- glm(event ~ sex + family_history + pce_score + prevent_score, 
                            data = df, 
                            family = binomial(link = "logit"))

# Predict risk scores WITHOUT Aladynoulli
risk_scores_without_noulli <- predict(model_without_noulli, type = "response")

# Fit logistic regression WITH Aladynoulli
model_with_noulli <- glm(event ~ sex + family_history + pce_score + prevent_score + noulli_risk, 
                         data = df, 
                         family = binomial(link = "logit"))

# Predict risk scores WITH Aladynoulli
risk_scores_with_noulli <- predict(model_with_noulli, type = "response")

# Compute C-index for both models
library(lifelines)

c_index_without_noulli <- concordance_index(y, -risk_scores_without_noulli, event_observed = 1)
c_index_with_noulli <- concordance_index(y, -risk_scores_with_noulli, event_observed = 1)

cat(sprintf("C-index WITHOUT Aladynoulli: %.3f\n", c_index_without_noulli))
cat(sprintf("C-index WITH Aladynoulli: %.3f\n", c_index_with_noulli))



# Now you can easily plot ROC curves for any disease
library(pROC)
library(ggplot2)
library(survival)
# Example: Plot ROC curve for ASCVD
disease <- "ASCVD"

tdc_auc_results <- test_time_dependent_cox(
  Y_test = Y_test,
  FH_processed = FH_processed2,
  test_indices = 0:10000,
  disease_mapping = disease_mapping,
  major_diseases = major_diseases,
  disease_names = disease_names,
  follow_up_duration_years = 10,
  fitted_models = tdc_models,
  pi_test = pi_test_full
)
results=tdc_auc_results

auc_results <- results$auc_results
concordance_results <- results$concordance_results
tdc_data_list <- results$tdc_data_list
tdc_data <- tdc_data_list[[disease]]
fitted_models=tdc_models
risk_scores <- predict(fitted_models[[disease]], newdata = tdc_data, type = "risk")
person_risks <- aggregate(risk_scores, by = list(id = tdc_data$id), FUN = mean)
person_events <- aggregate(tdc_data$event, by = list(id = tdc_data$id), FUN = max)

roc_obj <- roc(person_events$x, person_risks$x)
plot(roc_obj, main = paste("ROC Curve for", disease))

library(pROC)
library(ggplot2)

disease <- "ASCVD"
tdc_data <- tdc_data_list[[disease]]

# 1. Get per-person event status
person_events <- aggregate(tdc_data$event, by = list(id = tdc_data$id), FUN = max)

# 2. Get per-person risk scores for each model
# Your model (Noulli)
risk_scores <- predict(fitted_models[[disease]], newdata = tdc_data, type = "risk")
person_noulli_risks <- aggregate(risk_scores, by = list(id = tdc_data$id), FUN = mean)

# PREVENT (static, so just take the unique value per person)
person_prevent_risks <- aggregate(tdc_data$prevent_score, by = list(id = tdc_data$id), FUN = function(x) unique(x)[1])

# PCE (static, so just take the unique value per person)
person_pce_risks <- aggregate(tdc_data$pce_score, by = list(id = tdc_data$id), FUN = function(x) unique(x)[1])

# 3. Calculate ROC objects
roc_noulli <- roc(person_events$x, person_noulli_risks$x)
roc_prevent <- roc(person_events$x, person_prevent_risks$x)
roc_pce <- roc(person_events$x, person_pce_risks$x)

# 4. Plot all ROC curves together
plot(roc_noulli, col = "blue", lwd = 2, main = paste("ROC Curves for", disease))
lines(roc_prevent, col = "red", lwd = 2)
lines(roc_pce, col = "green", lwd = 2)
legend("bottomright",
       legend = c(
         sprintf("Noulli (AUC = %.3f)", auc(roc_noulli)),
         sprintf("PREVENT (AUC = %.3f)", auc(roc_prevent)),
         sprintf("PCE (AUC = %.3f)", auc(roc_pce))
       ),
       col = c("blue", "red", "green"),
       lwd = 2)


### males

library(pROC)
library(ggplot2)
disease <- "ASCVD"
tdc_data <- tdc_data_list[[disease]]
disease <- "ASCVD"
tdc_data <- tdc_data_list[[disease]]

# 1. Get per-person event status

tdc_data=tdc_data[tdc_data$sex==1,]
person_events <- aggregate(tdc_data$event, by = list(id = tdc_data$id), FUN = max)

# 2. Get per-person risk scores for each model
# Your model (Noulli)
risk_scores <- predict(fitted_models[[disease]], newdata = tdc_data, type = "risk")
person_noulli_risks <- aggregate(risk_scores, by = list(id = tdc_data$id), FUN = mean)

# PREVENT (static, so just take the unique value per person)
person_prevent_risks <- aggregate(tdc_data$prevent_score, by = list(id = tdc_data$id), FUN = function(x) unique(x)[1])

# PCE (static, so just take the unique value per person)
person_pce_risks <- aggregate(tdc_data$pce_score, by = list(id = tdc_data$id), FUN = function(x) unique(x)[1])

# 3. Calculate ROC objects
roc_noulli <- roc(person_events$x, person_noulli_risks$x)
roc_prevent <- roc(person_events$x, person_prevent_risks$x)
roc_pce <- roc(person_events$x, person_pce_risks$x)


pdf("MaleROC.pdf")
# 4. Plot all ROC curves together
plot(roc_noulli, col = "blue", lwd = 2, main = paste("ROC Curves for Males", disease))
lines(roc_prevent, col = "red", lwd = 2)
lines(roc_pce, col = "green", lwd = 2)
legend("bottomright",
       legend = c(
         sprintf("Noulli (AUC = %.3f)", auc(roc_noulli)),
         sprintf("PREVENT (AUC = %.3f)", auc(roc_prevent)),
         sprintf("PCE (AUC = %.3f)", auc(roc_pce))
       ),
       col = c("blue", "red", "green"),
       lwd = 2)

dev.off()

#### sex = 0 

disease <- "ASCVD"
tdc_data <- tdc_data_list[[disease]]
tdc_data=tdc_data[tdc_data$sex==0,]
person_events <- aggregate(tdc_data$event, by = list(id = tdc_data$id), FUN = max)

# 2. Get per-person risk scores for each model
# Your model (Noulli)
risk_scores <- predict(fitted_models[[disease]], newdata = tdc_data, type = "risk")
person_noulli_risks <- aggregate(risk_scores, by = list(id = tdc_data$id), FUN = mean)

# PREVENT (static, so just take the unique value per person)
person_prevent_risks <- aggregate(tdc_data$prevent_score, by = list(id = tdc_data$id), FUN = function(x) unique(x)[1])

# PCE (static, so just take the unique value per person)
person_pce_risks <- aggregate(tdc_data$pce_score, by = list(id = tdc_data$id), FUN = function(x) unique(x)[1])

# 3. Calculate ROC objects

surv_obj <- with(tdc_data, Surv(start, stop, event))
# 3. Calculate ROC objects
roc_noulli <- roc(person_events$x, person_noulli_risks$x)
roc_prevent <- roc(person_events$x, person_prevent_risks$x)
roc_pce <- roc(person_events$x, person_pce_risks$x)


pdf("FeMaleROC.pdf")
# 4. Plot all ROC curves together
plot(roc_noulli, col = "blue", lwd = 2, main = paste("ROC Curves for Females", disease))
lines(roc_prevent, col = "red", lwd = 2)
lines(roc_pce, col = "green", lwd = 2)
legend("bottomright",
       legend = c(
         sprintf("Noulli (AUC = %.3f)", auc(roc_noulli)),
         sprintf("PREVENT (AUC = %.3f)", auc(roc_prevent)),
         sprintf("PCE (AUC = %.3f)", auc(roc_pce))
       ),
       col = c("blue", "red", "green"),
       lwd = 2)

dev.off()
####

# repeat with the new pis:
Y_train=readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox/ukb_Y_train.rds")


pce_data = readRDS('/Users/sarahurbut/Library/Cloudstorage/Dropbox/pce_df_prevent.rds')
FH_processed = read.csv('/Users/sarahurbut/Library/CloudStorage/Dropbox/baselinagefamh.csv')
# Now we can use both pi files with the time-dependent Cox model
pi_train_full <- readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_full_leakage_free_20000_30000.rds")

# Fit time-dependent Cox models
tdc_models <- fit_time_dependent_cox(
  Y_train = Y_train,
  FH_processed = FH_processed,
  train_indices = 20001:30000,
  disease_mapping = disease_mapping,
  major_diseases = major_diseases,
  disease_names = disease_names,
  follow_up_duration_years = 10,
  pi_train = pi_train_full
)

saveRDS(object = tdc_models,file = "tdcmodels_leakagefree_75.rds")
rm(Y_train)

rm(pi_train_full)

gc()

Y_test=readRDS("ukb_Y_test.rds")
pi_test_full <- readRDS("/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_full_leakage_free_0_10000.rds")

# Evaluate time-dependent Cox models
tdc_auc_results <- test_time_dependent_cox(
  Y_test = Y_test,
  FH_processed = FH_processed,
  test_indices = 0:10000,
  disease_mapping = disease_mapping,
  major_diseases = major_diseases,
  disease_names = disease_names,
  follow_up_duration_years = 10,
  fitted_models = tdc_models,
  pi_test = pi_test_full
)

# Save results
tdc_auc_df <- data.frame(
  disease_group = names(tdc_auc_results[[1]]),
  auc = unlist(tdc_auc_results[[1]])
)

tdc_c_df <- data.frame(
  disease_group = names(tdc_auc_results[[2]]),
  c = unlist(tdc_auc_results[[2]])
)


write.csv(tdc_auc_df, "~/Library/CloudStorage/Dropbox/auc_results_tdc_20000_30000train_0_10000test_noleak.csv", quote = FALSE)

write.csv(tdc_c_df, "~/Library/CloudStorage/Dropbox/c_index_results_tdc_20000_30000train_0_10000test_noleak.csv", quote = FALSE)


