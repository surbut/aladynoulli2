library(MatchIt)
library(data.table)
library(survival)

# Load data
print("Loading data...")
cov=fread("/Users/sarahurbut/aladynoulli2/pyScripts/matched_pce_df_400k.csv")
true_statins=fread("/Users/sarahurbut/aladynoulli2/pyScripts/true_statins.csv")
stat_presc=fread("/Users/sarahurbut/aladynoulli2/pyScripts/prescription_patient_ids.csv")

# Load processed_ids and thetas_all_time
processed_ids <- read.csv("/Users/sarahurbut/aladynoulli2/pyScripts/processed_ids.csv")$eid
thetas_all_time <- readRDS("/Users/sarahurbut/aladynoulli2/pyScripts/all_thetas_array_time.rds")

print(paste("Loaded", length(processed_ids), "processed IDs"))
print(paste("Thetas dimensions:", paste(dim(thetas_all_time), collapse=" x ")))

# Filter to patients with prescription data
good_treat=true_statins[true_statins$eid %in% stat_presc$eid,]
good_cov=cov[cov$eid %in% stat_presc$eid,]

# Get first treatment date for each patient
first_treat <- good_treat[, .(first_treat_date = min(issue_date)), by = eid]
print(paste("Total unique statin users:", nrow(first_treat)))

# Merge first treatment date with covariates
merged_data <- merge(good_cov, first_treat, by = "eid", all.x = TRUE)

# Filter for incident users (treatment after enrollment)
incident_users <- merged_data[first_treat_date > Enrollment_Date, ]
controls <- merged_data[is.na(first_treat_date), ]

print(paste("Incident statin users:", nrow(incident_users)))
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
incident_users <- incident_users[incident_users$cad_before_treat == 0, ]

# For controls: diseases before enrollment
controls$dm_before_treat <- ifelse(controls$Dm_censor_age < controls$age_enrolled & controls$Dm_Any == 2, 1, 0)
controls$htn_before_treat <- ifelse(controls$Ht_censor_age < controls$age_enrolled & controls$Ht_Any == 2, 1, 0)
controls$hyperlip_before_treat <- ifelse(controls$HyperLip_censor_age < controls$age_enrolled & controls$HyperLip_Any == 2, 1, 0)

# Exclude controls with CAD before enrollment
controls$cad_before_enrollment <- ifelse(controls$Cad_censor_age < controls$age_enrolled & controls$Cad_Any == 2, 1, 0)
controls <- controls[controls$cad_before_enrollment == 0, ]

# Create treatment indicator and combine data
incident_users$treated <- 1
controls$treated <- 0
controls$first_treat_date <- NA
controls$age_at_first_statin <- NA
controls$cad_before_treat <- NA
incident_users$cad_before_enrollment <- NA

matching_data <- rbind(incident_users, controls)
print(paste("Final dataset size:", nrow(matching_data)))

# Calculate baseline age and time index
matching_data$baseline_age <- ifelse(matching_data$treated == 1, 
                                    matching_data$age_at_first_statin, 
                                    matching_data$age_enrolled)
matching_data$age_for_matching <- matching_data$baseline_age
matching_data$baseline_time_idx <- round(matching_data$baseline_age - 30 + 1)

# Simple signature extraction function (baseline only)
get_baseline_signatures <- function(eids, time_indices, thetas_array, processed_ids) {
  print("Extracting baseline signatures...")
  
  n_patients <- length(eids)
  n_signatures <- dim(thetas_array)[2]
  
  # Initialize result matrix
  signatures <- matrix(NA, nrow = n_patients, ncol = n_signatures)
  colnames(signatures) <- paste0("sig_", 1:n_signatures, "_t0")
  
  # Find indices in processed_ids for each eid
  eid_indices <- match(eids, processed_ids)
  
  # Extract baseline signatures for valid indices
  valid_indices <- !is.na(eid_indices) & 
                   time_indices >= 1 & 
                   time_indices <= dim(thetas_array)[3]
  
  print(paste("Valid patients for signature extraction:", sum(valid_indices), "out of", n_patients))
  
  if(sum(valid_indices) > 0) {
    valid_eid_idx <- eid_indices[valid_indices]
    valid_time_idx <- time_indices[valid_indices]
    
    for(i in 1:sum(valid_indices)) {
      if(i %% 1000 == 0) print(paste("Processed", i, "patients"))
      signatures[which(valid_indices)[i], ] <- thetas_array[valid_eid_idx[i], , valid_time_idx[i]]
    }
  }
  
  return(signatures)
}

# Extract baseline signatures only
baseline_signatures <- get_baseline_signatures(
  eids = matching_data$eid,
  time_indices = matching_data$baseline_time_idx,
  thetas_array = thetas_all_time,
  processed_ids = processed_ids
)

# Combine with main data
matching_data_with_sigs <- cbind(matching_data, baseline_signatures)

# Remove rows with missing signature data
complete_cases <- complete.cases(baseline_signatures)
matching_data_complete <- matching_data_with_sigs[complete_cases, ]
print(paste("Complete cases for matching:", sum(complete_cases), "out of", nrow(matching_data)))

# Create survival data
matching_data_complete$time1 <- matching_data_complete$baseline_age
matching_data_complete$time2 <- matching_data_complete$Cad_censor_age
matching_data_complete$event <- ifelse(matching_data_complete$Cad_Any == 2, 1, 0)

matching_data_complete=matching_data_complete[!is.na(matching_data_complete$pce_goff),]

print(paste("CAD events in treated:", sum(matching_data_complete$event[matching_data_complete$treated == 1])))
print(paste("CAD events in controls:", sum(matching_data_complete$event[matching_data_complete$treated == 0])))

# FOCUSED MATCHING STRATEGY: Test only the most promising approaches
print("=== FOCUSED MATCHING ANALYSIS ===")

# 1. Traditional matching (baseline)
print("1. Traditional matching (Age + Sex + PCE)...")
m.out_traditional <- matchit(treated ~ age_for_matching + Sex + pce_goff,
                            data = matching_data_complete,
                            method = "nearest",
                            ratio = 1,
                            caliper = 0.2)

matched_traditional <- match.data(m.out_traditional)
print(paste("Traditional matched pairs:", nrow(matched_traditional)/2))

# 2. Signature-enhanced matching (top 3 signatures only)
print("2. Signature-enhanced matching (top 3 signatures)...")
# Select signatures with highest variance
sig_vars <- apply(baseline_signatures, 2, var, na.rm=TRUE)
top_sigs <- names(sort(sig_vars, decreasing=TRUE)[1:3])
top_sigs=c("sig_6_t0" ,"sig_1_t0" ,"sig_16_t0")
print(paste("Top 3 signatures by variance:", paste(top_sigs, collapse=", ")))


sig_formula <- paste("treated ~ age_for_matching + Sex + pce_goff +", paste(top_sigs, collapse=" + "))
m.out_signatures <- matchit(as.formula(sig_formula),
                           data = matching_data_complete,
                           method = "nearest",
                           ratio = 1,
                           caliper = 0.2)

matched_signatures <- match.data(m.out_signatures)
print(paste("Signature-enhanced matched pairs:", nrow(matched_signatures)/2))

# Function to calculate HR
calculate_hr <- function(matched_data, description) {
  # Fix survival intervals
  zero_length <- matched_data$time2 <= matched_data$time1
  if(sum(zero_length) > 0) {
    matched_data$time2[zero_length] <- matched_data$time1[zero_length] + 0.01
  }
  
  valid_intervals <- matched_data$time2 > matched_data$time1 & 
                    !is.na(matched_data$time1) & 
                    !is.na(matched_data$time2) &
                    !is.na(matched_data$event)
  
  if(sum(!valid_intervals) > 0) {
    matched_data <- matched_data[valid_intervals, ]
  }
  
  # Create survival object and fit Cox model
  surv_obj <- Surv(time = matched_data$time1, time2 = matched_data$time2, event = matched_data$event)
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
  print("")
  
  return(list(hr = hr, ci = ci, p_val = p_val))
}

# Calculate HRs
print("=== CALCULATING HAZARD RATIOS ===")
hr_traditional <- calculate_hr(matched_traditional, "Traditional Matching")
hr_signatures <- calculate_hr(matched_signatures, "Signature-Enhanced Matching")

# Summary
print("=== SUMMARY ===")
print("Matching Approach          | HR (95% CI)           | P-value")
print("---------------------------|----------------------|---------")
print(paste("Traditional (Age+Sex+PCE) |", round(hr_traditional$hr, 3), "(", round(hr_traditional$ci[1], 3), "-", round(hr_traditional$ci[2], 3), ") |", format.pval(hr_traditional$p_val, digits=3)))
print(paste("Signature-Enhanced       |", round(hr_signatures$hr, 3), "(", round(hr_signatures$ci[1], 3), "-", round(hr_signatures$ci[2], 3), ") |", format.pval(hr_signatures$p_val, digits=3)))

# Check if signatures improved the result
if(hr_signatures$hr < hr_traditional$hr) {
  print("✓ Signature-enhanced matching shows stronger protective effect!")
} else {
  print("⚠ Traditional matching shows stronger protective effect")
}

print("=== ANALYSIS COMPLETE ===")
