## Matching conversion

library(reticulate)
## convert ro R
#use_condaenv("r-tensornoulli")
use_condaenv("/opt/miniconda3/envs/new_env_pyro2", required = TRUE)
torch <- import("torch")
tensor_to_r <- function(tensor) {
  as.array(tensor$detach()$cpu()$numpy())
}
np <- import("numpy")
pd <- import("pandas")

# ---------------------------------------------
# 2. Load data: PRS, patient IDs, PRS/disease names, G-matrix, covariates
# ---------------------------------------------
# Load polygenic risk scores (thetas) and processed patient IDs
# These are the main subject IDs and their PRS values
# thetas: (n_individuals, n_prs)


thetas = as.array(np$load("thetas.npy"))
# thetas: array of dim (N, D, T)
mean_thetas <- apply(thetas, c(2,3), mean) # D x T
sd_thetas <- apply(thetas, c(2,3), sd)     # D x T

# z-score array
# z-score array
z_thetas <- array(NA, dim=dim(thetas))
for (d in 1:dim(thetas)[2]) {
  for (t in 1:dim(thetas)[3]) {
    z_thetas[,d,t] <- (thetas[,d,t] - mean_thetas[d,t]) / sd_thetas[d,t]
  }
}
processed_ids = as.integer(np$load("processed_patient_ids.npy"))

# Load PRS names and labels for plotting/interpretation
prs_names = read.table('prs_names.csv',header = T)
prs_labels = prs_names['Names']

# Load disease names for reference
# (Assumes second column contains names)
disease_names = read.table("disease_names.csv",sep=",",header = T)[,"x"]

# Load G-matrix (genotype/PRS matrix)
G = tensor_to_r(torch$load("/Users/sarahurbut/Library/CloudStorage/Dropbox/data_for_running/G_matrix.pt",weights_only=FALSE))


# Load covariate data (demographics, labs, etc.)
cov = read.csv('/Users/sarahurbut/aladynoulli2/pyScripts/matched_pce_df_400k.csv')

names(cov)[1]= "eid"
cov$eid=as.integer(cov$eid)

# Parse enrollment date and calculate age at enrollment
cov$enrollment=as.Date(cov$Enrollment_Date)
cov$Birthdate=as.Date(cov$Birthdate)

cov['age_at_enroll'] = difftime(cov$enrollment,cov$Birthdate)/365.25


# Create named vectors (like Python dicts)
age_at_enroll <- setNames(cov$age_at_enroll, cov$eid)
eid_to_yob <- setNames(cov$birth_year, cov$eid)


prescription_path <- 'prescriptions.csv'
df_treat <- read.csv(prescription_path)
df_treat$eid <- as.integer(df_treat$eid)
df_treat <- merge(df_treat, cov[, c("eid", "birth_year")], by = "eid", all.x = TRUE)
df_treat$from <- as.Date(df_treat$from)

cov_eids <- unique(cov$eid)
processed_eids <- unique(processed_ids)
cat("EIDs in cov but not in processed_ids:", setdiff(cov_eids, processed_eids), "\n")
cat("EIDs in processed_ids but not in cov:", setdiff(processed_eids, cov_eids), "\n")
cat("Number of EIDs in cov:", length(cov_eids), "\n")
cat("Number of EIDs in processed_ids:", length(processed_eids), "\n")
cat("Number of overlapping EIDs:", length(intersect(cov_eids, processed_eids)), "\n")


drug_category <- "statins"
if (drug_category == "All") {
  df_drug <- df_treat
} else {
  df_drug <- subset(df_treat, category == drug_category)
}
num_unique_eids <- length(unique(df_drug$eid))
cat("Number of unique individuals in", drug_category, ":", num_unique_eids, "\n")
treat_eids <- unique(df_drug$eid)
overlap_treat_cov <- intersect(treat_eids, cov_eids)
cat("Number of people in df_drug:", length(treat_eids), "\n")
cat("Number of people in both df_drug and cov:", length(overlap_treat_cov), "\n")


library(dplyr)
first_presc <- df_drug %>%
  group_by(eid) %>%
  summarize(from = min(from, na.rm = TRUE)) %>%
  left_join(cov[, c("eid", "Birthdate", "Enrollment_Date")], by = "eid")

first_presc$Birthdate <- as.Date(first_presc$Birthdate)
first_presc$from <- as.Date(first_presc$from)
first_presc$Enrollment_Date <- as.Date(first_presc$Enrollment_Date)

first_presc$age_at_first_script <- as.numeric(difftime(first_presc$from, first_presc$Birthdate, units = "days")) / 365.25

incident_treated <- subset(first_presc, from > Enrollment_Date)
incident_treated$age_at_first_script <- as.numeric(difftime(incident_treated$from, incident_treated$Birthdate, units = "days")) / 365.25
incident_treated$years_since_30 <- round(incident_treated$age_at_first_script - 30)
dim(incident_treated)


# LDL and CAD PRS indices
ldl_idx <- which(prs_labels == "LDL_SF")
cad_idx <- which(prs_labels == "CAD")

# prev_dm
cov$prev_dm <- as.integer(cov$Dm_Any == 2 & cov$Dm_censor_age < cov$age_at_enroll)
# prev_dm1
cov$prev_dm1 <- as.integer(cov$DmT1_Any == 2 & cov$DmT1_censor_age < cov$age_at_enroll)
# prev_ht
cov$prev_ht <- as.integer(cov$Ht_Any == 2 & cov$Ht_censor_age < cov$age_at_enroll)
# prev_hl
cov$prev_hl <- as.integer(cov$HyperLip_Any == 2 & cov$HyperLip_censor_age < cov$age_at_enroll)

# Named vectors for covariates
eid_to_dm2_prev <- setNames(cov$prev_dm, cov$eid)
eid_to_antihtnbase <- setNames(cov$prev_ht, cov$eid)
eid_to_htn <- setNames(cov$prev_ht, cov$eid)
eid_to_smoke <- setNames(cov$SmokingStatusv2, cov$eid)
eid_to_dm1_prev <- setNames(cov$prev_dm1, cov$eid)
eid_to_hl_prev <- setNames(cov$prev_hl, cov$eid)
eid_to_sex <- setNames(cov$Sex, cov$eid)
eid_to_age <- setNames(cov$age_at_enroll, cov$eid)
eid_to_race <- setNames(cov$race, cov$eid)
eid_to_pce_goff <- setNames(cov$pce_goff, cov$eid)
eid_to_tchol <- setNames(cov$tchol, cov$eid)
eid_to_hdl <- setNames(cov$hdl, cov$eid)
eid_to_sbp <- setNames(cov$SBP, cov$eid)

# G is a matrix: rows = processed_ids, columns = PRS
eid_to_ldl_prs <- setNames(G[, ldl_idx], processed_ids)
eid_to_cad_prs <- setNames(G[, cad_idx], processed_ids)


treated_eids <- incident_treated$eid
treated_t0s <- incident_treated$years_since_30
treated_t0_dict <- setNames(treated_t0s, treated_eids)

treated_eids_set <- unique(treated_eids)
untreated_eids <- setdiff(processed_ids, treated_eids_set)

controls_df <- cov[cov$eid %in% untreated_eids, ]
controls <- controls_df
controls$years_since_30 <- round(controls$age_at_enroll - 30)
control_eids <- controls$eid
control_t0s <- controls$years_since_30

cat(length(untreated_eids), "\n")
cat(length(treated_eids_set), "\n")
cat(length(untreated_eids) + length(treated_eids_set), "\n")


covariate_dicts <- list(
  age_at_enroll = eid_to_age,
  sex = eid_to_sex,
  dm2_prev = eid_to_dm2_prev,
  antihtnbase = eid_to_antihtnbase,
  dm1_prev = eid_to_hl_prev,  # Note: check if this is intentional
  smoke = eid_to_smoke,
  ldl_prs = eid_to_ldl_prs,
  cad_prs = eid_to_cad_prs,
  tchol = eid_to_tchol,
  hdl = eid_to_hdl,
  sbp = eid_to_sbp,
  pce_goff = eid_to_pce_goff
)

treated_eids_list <- as.integer(treated_eids)
treated_t0s_list <- as.integer(treated_t0s)
control_eids_list <- as.integer(control_eids)
control_t0s_list <- as.integer(control_t0s)



build_features <- function(eids, t0s, processed_ids, thetas, covariate_dicts, sig_indices = NULL, window = 10, eid_to_idx = NULL) {
  features <- list()
  indices <- c()
  kept_eids <- c()
  n_signatures <- dim(thetas)[2]
  if (is.null(sig_indices)) sig_indices <- 1:n_signatures
  expected_length <- length(sig_indices) * window
  if (is.null(eid_to_idx)) {
    eid_to_idx <- setNames(seq_along(processed_ids), as.character(processed_ids))
  }
  for (i in seq_along(eids)) {
    eid <- eids[i]
    t0 <- t0s[i]
    idx <- eid_to_idx[as.character(eid)]
    if (is.na(idx) || t0 < window) next
    sig_traj <- as.vector(thetas[idx, sig_indices, (t0-window+1):t0])
    if (length(sig_traj) != expected_length) next
    age <- covariate_dicts$age_at_enroll[as.character(eid)]
    sex <- covariate_dicts$sex[as.character(eid)]
    dm2 <- covariate_dicts$dm2_prev[as.character(eid)]
    antihtn <- covariate_dicts$antihtnbase[as.character(eid)]
    dm1 <- covariate_dicts$dm1_prev[as.character(eid)]
    smoke <- covariate_dicts$smoke[as.character(eid)]
    ldl_prs <- covariate_dicts$ldl_prs[as.character(eid)]
    cad_prs <- covariate_dicts$cad_prs[as.character(eid)]
    tchol <- covariate_dicts$tchol[as.character(eid)]
    hdl <- covariate_dicts$hdl[as.character(eid)]
    sbp <- covariate_dicts$sbp[as.character(eid)]
    pce_goff <- covariate_dicts$pce_goff[as.character(eid)]
    features[[length(features)+1]] <- c(sig_traj, age, sex, dm2, antihtn, dm1, ldl_prs, cad_prs, tchol, hdl, sbp, pce_goff)
    indices <- c(indices, idx)
    kept_eids <- c(kept_eids, eid)
    
  }
  features_mat <- do.call(rbind, features)
  return(list(features = features_mat, indices = indices, kept_eids = kept_eids))
}

eid_to_idx <- setNames(seq_along(processed_ids), as.character(processed_ids))


treated_features_out <- build_features(
  eids = treated_eids_list,
  t0s = treated_t0s_list,
  processed_ids = processed_ids,
  thetas = z_thetas,
  covariate_dicts = covariate_dicts,
  sig_indices = 1:21, # or whatever indices you want
  window = 10
)
treated_features <- treated_features_out$features
treated_indices <- treated_features_out$indices
treated_eids_matched <- treated_features_out$kept_eids


time1=proc.time()
control_features_out <- build_features(
  eids = control_eids_list,
  t0s = control_t0s_list,
  processed_ids = processed_ids,
  thetas = thetas,
  covariate_dicts = covariate_dicts,
  sig_indices = 1:21, # or whatever indices you want
  window = 10,
  eid_to_idx = eid_to_idx
)
tot_time=proc.time()-time1
print(tot_time)
control_features <- control_features_out$features
control_indices <- control_features_out$indices
control_eids_matched <- control_features_out$kept_eids