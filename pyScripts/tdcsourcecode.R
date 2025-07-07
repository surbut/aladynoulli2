
library(survival)
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
    
    if(is.null(fitted_models)) {
     
    
    target_sex_code <- NA
    if (disease_group == "Breast_Cancer")
      target_sex_code <- 0
    if (disease_group == "Prostate_Cancer")
      target_sex_code <- 1


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
    if (!is.null(pi_test) && "noulli_risk" %in% colnames(tdc_data))
      formula_str <- paste(formula_str, "+ noulli_risk")
    
    print(formula_str)
    fit <- try(coxph(as.formula(formula_str), data = tdc_data, id = id), silent = TRUE)

    risk_scores <- predict(fit, newdata = tdc_data, type = "risk")
    
    
    # Suppose tdc_data has columns: id, start, stop, event, predicted_risk
    surv_obj <- with(tdc_data, Surv(start, stop, event))
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

    fit <- fitted_models[[disease_group]]

    if (is.null(fit) || nrow(tdc_data) == 0) next
    
    # Predict risk scores
    risk_scores <- predict(fit, newdata = tdc_data, type = "risk")
    
    
    # Suppose tdc_data has columns: id, start, stop, event, predicted_risk
    surv_obj <- with(tdc_data, Surv(start, stop, event))
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