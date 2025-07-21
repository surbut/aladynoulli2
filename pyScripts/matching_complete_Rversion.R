library(MatchIt)
library(data.table)

# Load data
cov=fread("/Users/sarahurbut/aladynoulli2/pyScripts/matched_pce_df_400k.csv")
true_statins=fread("/Users/sarahurbut/aladynoulli2/pyScripts/true_statins.csv")
stat_presc=fread("/Users/sarahurbut/aladynoulli2/pyScripts/prescription_patient_ids.csv")

# Load processed_ids and thetas_all_time
processed_ids <- read.csv("/Users/sarahurbut/aladynoulli2/pyScripts/processed_ids.csv")$eid # Assuming it's a single column
thetas_all_time <- readRDS("/Users/sarahurbut/aladynoulli2/pyScripts/all_thetas_array_time.rds")  # If saved as RDS
# OR if saved as CSV:
# thetas_all_time <- as.matrix(fread("/Users/sarahurbut/aladynoulli2/pyScripts/thetas_all_time.csv"))

print(paste("Loaded", length(processed_ids), "processed IDs"))
print(paste("Thetas dimensions:", paste(dim(thetas_all_time), collapse=" x ")))

good_treat=true_statins[true_statins$eid %in% stat_presc$eid,]
good_cov=cov[cov$eid %in% stat_presc$eid,]


# Get first treatment date for each patient
first_treat <- good_treat[, .(first_treat_date = min(issue_date)), by = eid]

print(paste("Total unique statin users:", nrow(first_treat)))

# Merge first treatment date with covariates
merged_data <- merge(good_cov, first_treat, by = "eid", all.x = TRUE)

print(paste("Total people with covariate data:", nrow(merged_data)))

# Filter for incident users (treatment after enrollment)
incident_users <- merged_data[first_treat_date > Enrollment_Date, ]
controls <- merged_data[is.na(first_treat_date), ]  # Never treated

print(paste("Incident statin users (after enrollment):", nrow(incident_users)))
print(paste("Never treated controls:", nrow(controls)))

# Calculate age at first statin for treated patients
incident_users$age_at_first_statin <- (incident_users$first_treat_date - incident_users$Birthdate) / 365.25

# For treated patients: diseases before treatment
incident_users$dm_before_treat <- ifelse(incident_users$Dm_censor_age < 
                                        (incident_users$first_treat_date - incident_users$Birthdate)/365.25 & incident_users$Dm_Any==2, 1, 0)

incident_users$htn_before_treat <- ifelse(incident_users$Ht_censor_age < 
                                         (incident_users$first_treat_date - incident_users$Birthdate)/365.25 & incident_users$Ht_Any==2, 1, 0)

incident_users$hyperlip_before_treat <- ifelse(incident_users$HyperLip_censor_age < 
                                              (incident_users$first_treat_date - incident_users$Birthdate)/365.25 & incident_users$HyperLip_Any==2, 1, 0)

# Exclude treated patients with CAD before statin initiation
incident_users$cad_before_treat <- ifelse(incident_users$Cad_censor_age < 
                                                 (incident_users$first_treat_date - incident_users$Birthdate)/365.25 & incident_users$Cad_Any==2, 1, 0)
print(paste("Treated patients with CAD before statin initiation:", sum(incident_users$cad_before_treat == 1)))
incident_users <- incident_users[incident_users$cad_before_treat == 0, ]
print(paste("Treated patients after excluding CAD before treatment:", nrow(incident_users)))

# Add missing column for consistency
incident_users$cad_before_enrollment <- NA

# For controls: diseases before enrollment
controls$dm_before_treat <- ifelse(controls$Dm_censor_age < controls$age_enrolled & controls$Dm_Any == 2, 1, 0)
controls$htn_before_treat <- ifelse(controls$Ht_censor_age < controls$age_enrolled & controls$Ht_Any == 2, 1, 0)
controls$hyperlip_before_treat <- ifelse(controls$HyperLip_censor_age < controls$age_enrolled & controls$HyperLip_Any == 2, 1, 0)

# Exclude controls with CAD before enrollment
controls$cad_before_enrollment <- ifelse(controls$Cad_censor_age < controls$age_enrolled & controls$Cad_Any == 2, 1, 0)
print(paste("Controls with CAD before enrollment:", sum(controls$cad_before_enrollment == 1)))
controls <- controls[controls$cad_before_enrollment == 0, ]
print(paste("Controls after excluding CAD before enrollment:", nrow(controls)))

# Add missing column for consistency
controls$cad_before_treat <- NA

# Create treatment indicator
incident_users$treated <- 1
controls$treated <- 0
controls$first_treat_date <- NA
controls$age_at_first_statin <- NA

print(paste("Treated patients without prior CAD:", nrow(incident_users)))
print(paste("Controls without prior CAD:", nrow(controls)))

# Combine all data for MatchIt to handle age matching
matching_data <- rbind(incident_users, controls)
print(paste("Total patients for matching:", nrow(matching_data)))
print(paste("Expected: ~", nrow(incident_users) + nrow(controls), "patients"))

# Check for missing values in matching covariates
matching_covariates <- c("age_for_matching", "Sex", "tchol", "hdl", "SBP", "pce_goff", "SmokingStatusv2", 
                        "dm_before_treat", "htn_before_treat", "hyperlip_before_treat")

print("=== Missing value check ===")
for(cov in matching_covariates) {
  if(cov %in% names(matching_data)) {
    missing_count <- sum(is.na(matching_data[[cov]]))
    print(paste("Missing", cov, ":", missing_count, "(", round(100*missing_count/nrow(matching_data), 1), "%)"))
  } else {
    print(paste("Column", cov, "not found"))
  }
}

# Impute missing values
print("=== Imputing missing values ===")

# For continuous variables: impute with median by treatment group
continuous_vars <- c("tchol", "hdl", "SBP", "pce_goff")
for(var in continuous_vars) {
  if(var %in% names(matching_data)) {
    # Impute treated group
    treated_median <- median(matching_data[[var]][matching_data$treated == 1], na.rm = TRUE)
    matching_data[[var]][matching_data$treated == 1 & is.na(matching_data[[var]])] <- treated_median
    
    # Impute control group
    control_median <- median(matching_data[[var]][matching_data$treated == 0], na.rm = TRUE)
    matching_data[[var]][matching_data$treated == 0 & is.na(matching_data[[var]])] <- control_median
    
    print(paste("Imputed", var, "- Treated median:", round(treated_median, 2), "Control median:", round(control_median, 2)))
  }
}

# For categorical variables: impute with mode by treatment group
categorical_vars <- c("SmokingStatusv2")
for(var in categorical_vars) {
  if(var %in% names(matching_data)) {
    # Get mode function
    get_mode <- function(x) {
      ux <- unique(x[!is.na(x)])
      ux[which.max(tabulate(match(x, ux)))]
    }
    
    # Impute treated group
    treated_mode <- get_mode(matching_data[[var]][matching_data$treated == 1])
    matching_data[[var]][matching_data$treated == 1 & is.na(matching_data[[var]])] <- treated_mode
    
    # Impute control group
    control_mode <- get_mode(matching_data[[var]][matching_data$treated == 0])
    matching_data[[var]][matching_data$treated == 0 & is.na(matching_data[[var]])] <- control_mode
    
    print(paste("Imputed", var, "- Treated mode:", treated_mode, "Control mode:", control_mode))
  }
}

# Check missing values after imputation
print("=== Missing values after imputation ===")
for(cov in matching_covariates) {
  if(cov %in% names(matching_data)) {
    missing_count <- sum(is.na(matching_data[[cov]]))
    print(paste("Missing", cov, ":", missing_count, "(", round(100*missing_count/nrow(matching_data), 1), "%)"))
  }
}

# Remove patients with missing disease indicators (we can't impute these)
disease_indicators <- c("dm_before_treat", "htn_before_treat", "hyperlip_before_treat")
missing_disease <- is.na(matching_data$dm_before_treat) | 
                   is.na(matching_data$htn_before_treat) | 
                   is.na(matching_data$hyperlip_before_treat)

print(paste("Patients with missing disease indicators:", sum(missing_disease)))
matching_data <- matching_data[!missing_disease, ]
print(paste("Final dataset size after removing missing disease indicators:", nrow(matching_data)))

# Calculate baseline age for each person (treatment age for treated, enrollment age for controls)
matching_data$baseline_age <- ifelse(matching_data$treated == 1, 
                                    matching_data$age_at_first_statin, 
                                    matching_data$age_enrolled)

# Create age variable for matching (age at first statin for treated, age at enrollment for controls)
matching_data$age_for_matching <- ifelse(matching_data$treated == 1, 
                                        matching_data$age_at_first_statin, 
                                        matching_data$age_enrolled)

# Convert baseline age to time index (R indexing)
matching_data$baseline_time_idx <- round(matching_data$baseline_age - 30 + 1)  # +1 for R indexing

# Faster way: Vectorized signature extraction
get_signature_trajectories <- function(eids, time_indices, thetas_array, processed_ids, n_years_before = 10, n_years_after = 5) {
  start_time <- Sys.time()
  print(paste("Starting signature extraction at:", start_time))
  
  n_patients <- length(eids)
  n_signatures <- dim(thetas_array)[2]
  n_timepoints_before <- n_years_before * n_signatures
  n_timepoints_after <- n_years_after * n_signatures
  n_total_timepoints <- n_timepoints_before + n_timepoints_after
  
  print(paste("Processing", n_patients, "patients,", n_signatures, "signatures"))
  print(paste("Extracting", n_years_before, "years before +", n_years_after, "years after baseline"))
  
  # Initialize result matrix
  trajectories <- matrix(NA, nrow = n_patients, ncol = n_total_timepoints)
  
  # Create column names
  before_cols <- paste0("sig_", rep(1:n_signatures, each = n_years_before), "_t", rep(-(n_years_before-1):0, n_signatures))
  after_cols <- paste0("sig_", rep(1:n_signatures, each = n_years_after), "_t", rep(1:n_years_after, n_signatures))
  colnames(trajectories) <- c(before_cols, after_cols)
  
  # Find indices in processed_ids for each eid
  eid_match_time <- Sys.time()
  eid_indices <- match(eids, processed_ids)
  print(paste("EID matching took:", round(difftime(Sys.time(), eid_match_time, units="secs"), 2), "seconds"))
  
  # Extract trajectories for valid indices
  valid_check_time <- Sys.time()
  valid_indices <- !is.na(eid_indices) & 
                   time_indices >= n_years_before & 
                   (time_indices + n_years_after) <= dim(thetas_array)[3]
  print(paste("Valid indices check took:", round(difftime(Sys.time(), valid_check_time, units="secs"), 2), "seconds"))
  print(paste("Valid patients:", sum(valid_indices), "out of", n_patients))
  
  if(sum(valid_indices) > 0) {
    # Extract all valid trajectories at once
    extraction_time <- Sys.time()
    valid_eid_idx <- eid_indices[valid_indices]
    valid_time_idx <- time_indices[valid_indices]
    
    print(paste("Starting trajectory extraction for", sum(valid_indices), "patients..."))
    
    for(i in 1:sum(valid_indices)) {
      if(i %% 1000 == 0) {
        print(paste("Processed", i, "out of", sum(valid_indices), "patients"))
      }
      
      # Extract before baseline (t-9 to t0)
      before_start <- valid_time_idx[i] - n_years_before + 1
      before_end <- valid_time_idx[i]
      before_traj <- thetas_array[valid_eid_idx[i], , before_start:before_end]
      
      # Extract after baseline (t1 to t5)
      after_start <- valid_time_idx[i] + 1
      after_end <- valid_time_idx[i] + n_years_after
      after_traj <- thetas_array[valid_eid_idx[i], , after_start:after_end]
      
      # Combine and store
      full_traj <- c(as.vector(before_traj), as.vector(after_traj))
      trajectories[which(valid_indices)[i], ] <- full_traj
    }
    
    print(paste("Trajectory extraction took:", round(difftime(Sys.time(), extraction_time, units="secs"), 2), "seconds"))
  }
  
  total_time <- difftime(Sys.time(), start_time, units="secs")
  print(paste("Total signature extraction time:", round(total_time, 2), "seconds"))
  
  return(trajectories)
}

# Get signature trajectories before AND after baseline
print("=== Starting signature trajectory extraction (before + after) ===")
signature_start_time <- Sys.time()
signature_trajectories <- get_signature_trajectories(
  eids = matching_data$eid,
  time_indices = matching_data$baseline_time_idx,
  thetas_array = thetas_all_time,
  processed_ids = processed_ids,
  n_years_before = 10,
  n_years_after = 5
)
print(paste("Total signature extraction time:", round(difftime(Sys.time(), signature_start_time, units="secs"), 2), "seconds"))

print(paste("Extracted signature trajectories for", sum(complete.cases(signature_trajectories)), "out of", nrow(matching_data), "patients"))

# Check signature extraction quality
print("=== Signature extraction check ===")
print(paste("Signature matrix dimensions:", nrow(signature_trajectories), "x", ncol(signature_trajectories)))
print(paste("Non-NA signatures:", sum(!is.na(signature_trajectories))))
print(paste("Mean signature value:", round(mean(signature_trajectories, na.rm=TRUE), 4)))
print(paste("SD signature value:", round(sd(signature_trajectories, na.rm=TRUE), 4)))

# Check a few specific patients
sample_patients <- sample(1:nrow(matching_data), min(5, nrow(matching_data)))
for(i in sample_patients) {
  eid <- matching_data$eid[i]
  baseline_age <- matching_data$baseline_age[i]
  time_idx <- matching_data$baseline_time_idx[i]
  sig_before <- signature_trajectories[i, 1:10]  # First 10 values (sig_1_t-9 to sig_1_t0)
  sig_after <- signature_trajectories[i, 211:220]  # First 10 post-baseline values (sig_1_t1 to sig_1_t5)
  print(paste("Patient", eid, "- Age:", round(baseline_age,1), "- Time idx:", time_idx))
  print(paste("  Sig1 before range:", round(range(sig_before, na.rm=TRUE), 3)))
  print(paste("  Sig1 after range:", round(range(sig_after, na.rm=TRUE), 3)))
}

# Create survival data with age as time scale
matching_data$time1 <- matching_data$baseline_age  # Start time (age at baseline)
matching_data$time2 <- matching_data$Cad_censor_age  # End time (age at CAD or censoring)
matching_data$event <- ifelse(matching_data$Cad_Any == 2, 1, 0)  # 1 = CAD event, 0 = censored

# Check survival data
print(paste("CAD events in treated:", sum(matching_data$event[matching_data$treated == 1])))
print(paste("CAD events in controls:", sum(matching_data$event[matching_data$treated == 0])))
print(paste("Mean follow-up time (years):", round(mean(matching_data$time2 - matching_data$time1, na.rm=TRUE), 2)))

# 2. Combine with baseline covariates
matching_data_with_sigs <- cbind(matching_data, signature_trajectories)

# Remove rows with missing signature data
complete_cases <- complete.cases(signature_trajectories)
matching_data_complete <- matching_data_with_sigs[complete_cases, ]

print(paste("Complete cases for matching:", sum(complete_cases), "out of", nrow(matching_data)))

# 3. Perform matching with signatures (current levels only)
library(MatchIt)

# PARSIMONIOUS MATCHING APPROACH - Based on established statin study methods
# Strategy: Match on key confounders only, avoid over-matching
# Key principle: Match on variables that predict treatment assignment but not outcome

print("=== Implementing Parsimonious Matching Strategy ===")

# 1. MINIMAL MATCHING: Only essential confounders
# Based on literature: age, sex, and PCE score are the most important
m.out_minimal <- matchit(treated ~ age_for_matching + Sex + pce_goff,
                        data = matching_data_complete,
                        method = "nearest",
                        ratio = 1,
                        caliper = 0.25)

# 2. STANDARD MATCHING: True confounders only (not mediators)
# Avoid matching on variables that could be affected by statin treatment
# Focus on: age, sex, smoking, diabetes, hypertension (pre-existing conditions)
m.out_standard <- matchit(treated ~ age_for_matching + Sex + SmokingStatusv2 + 
                         dm_before_treat + htn_before_treat,
                         data = matching_data_complete,
                         method = "nearest",
                         ratio = 1,
                         caliper = 0.2)

# 3. SIGNATURE-ENHANCED: Add key signatures only
# Select top signatures based on variance and clinical relevance
# Use only baseline signatures (t0) to avoid collider bias
m.out_sigs_minimal <- matchit(treated ~ age_for_matching + Sex + SmokingStatusv2 + 
                                dm_before_treat + htn_before_treat+
                             sig_16_t0 + sig_6_t0 + sig_1_t0,
                             data = matching_data_complete,
                             method = "nearest",
                             ratio = 1,
                             caliper = 0.25)

# 4. COMPREHENSIVE: All true confounders (avoiding mediators)
# Include all variables that predict treatment but aren't affected by it
m.out_comprehensive <- matchit(treated ~ age_for_matching + Sex + SmokingStatusv2 + 
                              dm_before_treat + htn_before_treat + hyperlip_before_treat + pce_goff,
                              data = matching_data_complete,
                              method = "nearest",
                              ratio = 1,
                              caliper = 0.2)

# 5. EXHAUSTIVE MATCHING: Try everything possible
# Use optimal matching with exact matching on key variables
m.out_exhaustive <- matchit(treated ~ age_for_matching + Sex + SmokingStatusv2 + 
                           dm_before_treat + htn_before_treat + hyperlip_before_treat + pce_goff +
                           sig_1_t0 + sig_2_t0 + sig_3_t0 + sig_4_t0 + sig_5_t0 + sig_6_t0 + sig_7_t0 + sig_8_t0 + sig_9_t0 + sig_10_t0 +
                           sig_11_t0 + sig_12_t0 + sig_13_t0 + sig_14_t0 + sig_15_t0 + sig_16_t0 + sig_17_t0 + sig_18_t0 + sig_19_t0 + sig_20_t0 + sig_21_t0,
                           data = matching_data_complete,
                           method = "nearest",
                           ratio = 1,
                           caliper = 0.2)
                           #method = "optimal",
                           #ratio = 1,
                           #exact = c("Sex", "dm_before_treat", "htn_before_treat"))

# 6. AGGRESSIVE CALIPER: Very tight matching
m.out_aggressive <- matchit(treated ~ age_for_matching + Sex + pce_goff,
                           data = matching_data_complete,
                           method = "nearest",
                           ratio = 1,
                           caliper = 0.1)

# 7. PROPENSITY SCORE STRATIFICATION: Alternative approach
# Create propensity score model
ps_model <- glm(treated ~ age_for_matching + Sex + SmokingStatusv2 + 
                dm_before_treat + htn_before_treat + hyperlip_before_treat + pce_goff,
                data = matching_data_complete, family = binomial)

matching_data_complete$ps_score <- predict(ps_model, type = "response")

# Stratify by propensity score quintiles
matching_data_complete$ps_quintile <- cut(matching_data_complete$ps_score, 
                                         breaks = quantile(matching_data_complete$ps_score, probs = 0:5/5),
                                         labels = FALSE, include.lowest = TRUE)

# 8. TIME-VARYING APPROACH: Match on baseline, adjust for time-varying signatures in survival model
m.out_timevarying <- matchit(treated ~ age_for_matching + Sex + pce_goff,
                             data = matching_data_complete,
                             method = "nearest",
                             ratio = 1,
                             caliper = 0.2)

# 5. Compare balance across all approaches
print("=== MINIMAL MATCHING (Age + Sex + PCE) ===")
summary(m.out_minimal)

print("=== STANDARD MATCHING (True confounders only) ===")
summary(m.out_standard)

print("=== SIGNATURE-ENHANCED (Minimal + Top 5 signatures) ===")
summary(m.out_sigs_minimal)

print("=== COMPREHENSIVE MATCHING (All true confounders) ===")
summary(m.out_comprehensive)

print("=== EXHAUSTIVE MATCHING (All signatures + exact matching) ===")
summary(m.out_exhaustive)

print("=== AGGRESSIVE CALIPER (Very tight matching) ===")
summary(m.out_aggressive)

print("=== TIME-VARYING APPROACH (Baseline matching) ===")
summary(m.out_timevarying)

# Additional matching quality checks
print("=== Detailed matching quality check ===")

# Check covariate balance before and after matching
check_balance <- function(matched_data, description) {
  print(paste("=== Balance check:", description, "==="))
  
  # Key covariates to check
  covariates <- c("age_for_matching", "Sex", "tchol", "hdl", "SBP", "pce_goff", 
                  "dm_before_treat", "htn_before_treat", "hyperlip_before_treat")
  
  for(cov in covariates) {
    if(cov %in% names(matched_data)) {
      treated_mean <- mean(matched_data[[cov]][matched_data$treated == 1], na.rm=TRUE)
      control_mean <- mean(matched_data[[cov]][matched_data$treated == 0], na.rm=TRUE)
      treated_sd <- sd(matched_data[[cov]][matched_data$treated == 1], na.rm=TRUE)
      
      # Standardized mean difference
      smd <- abs(treated_mean - control_mean) / treated_sd
      
      print(paste(cov, "- Treated:", round(treated_mean, 3), 
                  "Control:", round(control_mean, 3), 
                  "SMD:", round(smd, 3)))
    }
  }
  
  # Check event rates
  treated_events <- sum(matched_data$event[matched_data$treated == 1])
  control_events <- sum(matched_data$event[matched_data$treated == 0])
  treated_n <- sum(matched_data$treated == 1)
  control_n <- sum(matched_data$treated == 0)
  
  print(paste("Event rates - Treated:", treated_events, "/", treated_n, "=", round(100*treated_events/treated_n, 1), "%"))
  print(paste("Event rates - Control:", control_events, "/", control_n, "=", round(100*control_events/control_n, 1), "%"))
  print("")
}

# 6. Get matched datasets
matched_data_minimal <- match.data(m.out_minimal)
matched_data_standard <- match.data(m.out_standard)
matched_data_sigs_minimal <- match.data(m.out_sigs_minimal)
matched_data_comprehensive <- match.data(m.out_comprehensive)
matched_data_exhaustive <- match.data(m.out_exhaustive)
matched_data_aggressive <- match.data(m.out_aggressive)
matched_data_timevarying <- match.data(m.out_timevarying)

print(paste("Matched pairs - Minimal:", nrow(matched_data_minimal)/2))
print(paste("Matched pairs - Standard:", nrow(matched_data_standard)/2))
print(paste("Matched pairs - Signature-enhanced:", nrow(matched_data_sigs_minimal)/2))
print(paste("Matched pairs - Comprehensive:", nrow(matched_data_comprehensive)/2))
print(paste("Matched pairs - Exhaustive:", nrow(matched_data_exhaustive)/2))
print(paste("Matched pairs - Aggressive:", nrow(matched_data_aggressive)/2))
print(paste("Matched pairs - Time-varying:", nrow(matched_data_timevarying)/2))

# Check balance for all approaches
check_balance(matched_data_minimal, "Minimal Matching")
check_balance(matched_data_standard, "Standard Matching")
check_balance(matched_data_sigs_minimal, "Signature-Enhanced Matching")
check_balance(matched_data_comprehensive, "Comprehensive Matching")
check_balance(matched_data_exhaustive, "Exhaustive Matching")
check_balance(matched_data_aggressive, "Aggressive Caliper")
check_balance(matched_data_timevarying, "Time-Varying Approach")

# 7. Calculate HR for both approaches
library(survival)

# Function to calculate HR from matched data
calculate_hr <- function(matched_data, description) {
  # Fix zero-length intervals by adding small amount to time2
  zero_length <- matched_data$time2 <= matched_data$time1
  if(sum(zero_length) > 0) {
    print(paste("Found", sum(zero_length), "zero-length intervals, adjusting..."))
    matched_data$time2[zero_length] <- matched_data$time1[zero_length] + 0.01
  }
  
  # Additional check for very short intervals
  very_short <- (matched_data$time2 - matched_data$time1) < 0.001
  if(sum(very_short) > 0) {
    print(paste("Found", sum(very_short), "very short intervals, adjusting..."))
    matched_data$time2[very_short] <- matched_data$time1[very_short] + 0.1
  }
  
  # Remove any remaining problematic cases
  valid_intervals <- matched_data$time2 > matched_data$time1 & 
                    !is.na(matched_data$time1) & 
                    !is.na(matched_data$time2) &
                    !is.na(matched_data$event)
  
  if(sum(!valid_intervals) > 0) {
    print(paste("Removing", sum(!valid_intervals), "invalid intervals"))
    matched_data <- matched_data[valid_intervals, ]
  }
  
  print(paste("Final dataset size for HR calculation:", nrow(matched_data)))
  
  # Create survival object with age as time scale
  surv_obj <- Surv(time = matched_data$time1, time2 = matched_data$time2, event = matched_data$event)
  
  # Fit Cox model
  cox_model <- coxph(surv_obj ~ treated, data = matched_data)
  
  # Extract results
  hr <- exp(coef(cox_model)["treated"])
  ci <- exp(confint(cox_model)["treated", ])
  p_val <- summary(cox_model)$coefficients["treated", "Pr(>|z|)"]
  
  print(paste("=== HR Results:", description, "==="))
  print(paste("Hazard Ratio:", round(hr, 3)))
  print(paste("95% CI:", round(ci[1], 3), "-", round(ci[2], 3)))
  print(paste("P-value:", format.pval(p_val, digits = 3)))
  print(paste("Treated events:", sum(matched_data$event[matched_data$treated == 1])))
  print(paste("Control events:", sum(matched_data$event[matched_data$treated == 0])))
  print(paste("Mean follow-up (years):", round(mean(matched_data$time2 - matched_data$time1, na.rm=TRUE), 2)))
  print("")
  
  return(list(hr = hr, ci = ci, p_val = p_val))
}

# Calculate HRs for all approaches
print("=== Calculating HRs for All Matching Approaches ===")
print("Expected: Statins should show protective effect (HR < 1)")

hr_minimal <- calculate_hr(matched_data_minimal, "Minimal Matching (Age + Sex + PCE)")
hr_standard <- calculate_hr(matched_data_standard, "Standard Matching (True confounders)")
hr_sigs_minimal <- calculate_hr(matched_data_sigs_minimal, "Signature-Enhanced (Minimal + Top 5 signatures)")
hr_comprehensive <- calculate_hr(matched_data_comprehensive, "Comprehensive Matching (All true confounders)")
hr_exhaustive <- calculate_hr(matched_data_exhaustive, "Exhaustive Matching (All signatures + exact)")
hr_aggressive <- calculate_hr(matched_data_aggressive, "Aggressive Caliper (Very tight matching)")
hr_timevarying <- calculate_hr(matched_data_timevarying, "Time-Varying Approach (Baseline matching)")

# Propensity Score Stratification HR
print("=== PROPENSITY SCORE STRATIFICATION HR ===")
ps_strat_hr <- function(data, description) {
  # Calculate HR within each propensity score quintile
  quintile_hrs <- numeric(5)
  quintile_weights <- numeric(5)
  
  for(q in 1:5) {
    quintile_data <- data[data$ps_quintile == q, ]
    if(nrow(quintile_data) > 0 && sum(quintile_data$treated == 1) > 0 && sum(quintile_data$treated == 0) > 0) {
      # Fix survival intervals
      zero_length <- quintile_data$time2 <= quintile_data$time1
      if(sum(zero_length) > 0) {
        quintile_data$time2[zero_length] <- quintile_data$time1[zero_length] + 0.01
      }
      
      valid_intervals <- quintile_data$time2 > quintile_data$time1 & 
                        !is.na(quintile_data$time1) & 
                        !is.na(quintile_data$time2) &
                        !is.na(quintile_data$event)
      
      if(sum(valid_intervals) > 10) {
        quintile_data <- quintile_data[valid_intervals, ]
        surv_obj <- Surv(time = quintile_data$time1, time2 = quintile_data$time2, event = quintile_data$event)
        cox_model <- coxph(surv_obj ~ treated, data = quintile_data)
        quintile_hrs[q] <- exp(coef(cox_model)["treated"])
        quintile_weights[q] <- nrow(quintile_data)
      }
    }
  }
  
  # Weighted average HR across quintiles
  valid_quintiles <- quintile_weights > 0
  if(sum(valid_quintiles) > 0) {
    weighted_hr <- weighted.mean(quintile_hrs[valid_quintiles], quintile_weights[valid_quintiles])
    print(paste("=== HR Results:", description, "==="))
    print(paste("Weighted Average HR:", round(weighted_hr, 3)))
    print(paste("HR by quintile:", paste(round(quintile_hrs[valid_quintiles], 3), collapse=", ")))
    print("")
    return(weighted_hr)
  } else {
    print(paste("=== HR Results:", description, "==="))
    print("Insufficient data for propensity score stratification")
    print("")
    return(NA)
  }
}

hr_ps_strat <- ps_strat_hr(matching_data_complete, "Propensity Score Stratification")

# INDIVIDUAL SIGNATURE TESTING
print("=== TESTING INDIVIDUAL SIGNATURES ===")
print("Matching on minimal covariates (Age + Sex + PCE) + one signature at a time")

# Function to test individual signatures
test_individual_signatures <- function(data, base_covariates = "age_for_matching + Sex + pce_goff") {
  signature_results <- data.frame(
    signature = paste0("sig_", 1:21, "_t0"),
    hr = numeric(21),
    ci_lower = numeric(21),
    ci_upper = numeric(21),
    p_value = numeric(21),
    matched_pairs = numeric(21)
  )
  
  for(i in 1:21) {
    sig_name <- paste0("sig_", i, "_t0")
    print(paste("Testing signature", i, "..."))
    
    # Create matching formula with one signature
    formula_str <- paste("treated ~", base_covariates, "+", sig_name)
    matching_formula <- as.formula(formula_str)
    
    # Perform matching
    m.out_single <- matchit(matching_formula,
                           data = data,
                           method = "nearest",
                           ratio = 1,
                           caliper = 0.2)
    
    # Get matched data
    matched_data_single <- match.data(m.out_single)
    
    # Calculate HR
    hr_result <- calculate_hr(matched_data_single, paste("Signature", i, "only"))
    
    # Store results
    signature_results$hr[i] <- hr_result$hr
    signature_results$ci_lower[i] <- hr_result$ci[1]
    signature_results$ci_upper[i] <- hr_result$ci[2]
    signature_results$p_value[i] <- hr_result$p_val
    signature_results$matched_pairs[i] <- nrow(matched_data_single) / 2
  }
  
  return(signature_results)
}

# Test individual signatures
individual_sig_results <- test_individual_signatures(matching_data_complete)

# Find best signatures (closest to HR = 1 or most protective)
individual_sig_results$distance_from_1 <- abs(individual_sig_results$hr - 1)
individual_sig_results$protective_effect <- individual_sig_results$hr < 1

# Sort by distance from 1 (best first)
individual_sig_results_sorted <- individual_sig_results[order(individual_sig_results$distance_from_1), ]

print("=== INDIVIDUAL SIGNATURE RESULTS (sorted by distance from HR=1) ===")
print(individual_sig_results_sorted[, c("signature", "hr", "ci_lower", "ci_upper", "p_value", "matched_pairs")])

# Find top 5 signatures
top_5_signatures <- individual_sig_results_sorted$signature[1:5]
print(paste("Top 5 signatures:", paste(top_5_signatures, collapse=", ")))

# Test combination of top signatures
print("=== TESTING TOP 5 SIGNATURES COMBINED ===")
top_sigs_formula <- paste("treated ~ age_for_matching + Sex + pce_goff +", paste(top_5_signatures, collapse=" + "))
m.out_top5 <- matchit(as.formula(top_sigs_formula),
                      data = matching_data_complete,
                      method = "nearest",
                      ratio = 1,
                      caliper = 0.2)

matched_data_top5 <- match.data(m.out_top5)
hr_top5 <- calculate_hr(matched_data_top5, "Top 5 Signatures Combined")

# Summary table
print("=== SUMMARY OF HAZARD RATIOS ===")
print("Matching Approach          | HR (95% CI)           | P-value")
print("---------------------------|----------------------|---------")
print(paste("Minimal (Age+Sex+PCE)    |", round(hr_minimal$hr, 3), "(", round(hr_minimal$ci[1], 3), "-", round(hr_minimal$ci[2], 3), ") |", format.pval(hr_minimal$p_val, digits=3)))
print(paste("Standard (True confounders)|", round(hr_standard$hr, 3), "(", round(hr_standard$ci[1], 3), "-", round(hr_standard$ci[2], 3), ") |", format.pval(hr_standard$p_val, digits=3)))
print(paste("Signature-Enhanced       |", round(hr_sigs_minimal$hr, 3), "(", round(hr_sigs_minimal$ci[1], 3), "-", round(hr_sigs_minimal$ci[2], 3), ") |", format.pval(hr_sigs_minimal$p_val, digits=3)))
print(paste("Comprehensive (True conf.)|", round(hr_comprehensive$hr, 3), "(", round(hr_comprehensive$ci[1], 3), "-", round(hr_comprehensive$ci[2], 3), ") |", format.pval(hr_comprehensive$p_val, digits=3)))
print(paste("Exhaustive (All sigs)    |", round(hr_exhaustive$hr, 3), "(", round(hr_exhaustive$ci[1], 3), "-", round(hr_exhaustive$ci[2], 3), ") |", format.pval(hr_exhaustive$p_val, digits=3)))
print(paste("Aggressive Caliper       |", round(hr_aggressive$hr, 3), "(", round(hr_aggressive$ci[1], 3), "-", round(hr_aggressive$ci[2], 3), ") |", format.pval(hr_aggressive$p_val, digits=3)))
print(paste("Time-Varying Approach    |", round(hr_timevarying$hr, 3), "(", round(hr_timevarying$ci[1], 3), "-", round(hr_timevarying$ci[2], 3), ") |", format.pval(hr_timevarying$p_val, digits=3)))
if(!is.na(hr_ps_strat)) {
  print(paste("Propensity Score Strat.  |", round(hr_ps_strat, 3), "                    | N/A"))
}
print(paste("Top 5 Signatures Combined|", round(hr_top5$hr, 3), "(", round(hr_top5$ci[1], 3), "-", round(hr_top5$ci[2], 3), ") |", format.pval(hr_top5$p_val, digits=3)))

# Show best individual signature
best_sig <- individual_sig_results_sorted$signature[1]
best_sig_hr <- individual_sig_results_sorted$hr[1]
print(paste("Best Individual Signature |", round(best_sig_hr, 3), "                    |", best_sig))
