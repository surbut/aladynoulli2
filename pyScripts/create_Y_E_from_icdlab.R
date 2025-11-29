#--- Setup -------------------------------------------------
library(data.table)
library(dplyr)
library(torch)

#--- Load existing ICD/Phecode data ------------------------
icdlab <- readRDS("~/Dropbox-Personal/icd10phe_lab.rds")

# Convert to data.table for efficiency
if (!is.data.table(icdlab)) {
  icdlab <- as.data.table(icdlab)
}

# Check column names (adjust if needed)
print("Column names:")
print(names(icdlab))
print(paste("Total rows:", nrow(icdlab)))
print(paste("Unique patients:", length(unique(icdlab[[1]]))))

# Assume columns are: patient_id, age_diag, phecode (diag_icd10)
# Adjust column names if different
col_names <- names(icdlab)
patient_col <- col_names[1]  # First column should be patient ID
age_col <- col_names[grep("age", col_names, ignore.case = TRUE)[1]]  # Age column
phecode_col <- col_names[grep("phecode|icd|diag", col_names, ignore.case = TRUE)[1]]  # Phecode column

print(paste("Using columns:"))
print(paste("  Patient:", patient_col))
print(paste("  Age:", age_col))
print(paste("  Phecode:", phecode_col))

# Rename for consistency
icdlab <- icdlab %>%
  rename(
    eid = !!sym(patient_col),
    age_diag = !!sym(age_col),
    diag_icd10 = !!sym(phecode_col)
  )

#--- Filter age range (29-80) ----------------------------
icdlab_filtered <- icdlab %>%
  filter(age_diag >= 29, age_diag <= 80) %>%
  mutate(age_idx = round(age_diag) - 29) %>%
  filter(age_idx >= 0)  # Ensure non-negative

print(paste("After filtering (age 29-80):", nrow(icdlab_filtered), "rows"))

#--- Create integer encodings ------------------------------
eids <- unique(icdlab_filtered$eid)
phecodes <- unique(icdlab_filtered$diag_icd10)

eid_index <- setNames(seq_along(eids) - 1, eids)  # 0-indexed for Python
phecode_index <- setNames(seq_along(phecodes) - 1, phecodes)  # 0-indexed for Python

n_patients <- length(eids)
n_diseases <- length(phecodes)
n_timepoints <- max(icdlab_filtered$age_idx) + 1

print(paste("Patients:", n_patients))
print(paste("Diseases (phecodes):", n_diseases))
print(paste("Timepoints:", n_timepoints))

#--- Create Y tensor --------------------------------------
Y <- torch_zeros(c(n_patients, n_diseases, n_timepoints), dtype = torch_int8())

print("Filling Y tensor...")
pb <- txtProgressBar(min = 0, max = nrow(icdlab_filtered), style = 3)

for (i in seq_len(nrow(icdlab_filtered))) {
  row <- icdlab_filtered[i]
  eid_idx <- eid_index[[as.character(row$eid)]]
  phecode_idx <- phecode_index[[as.character(row$diag_icd10)]]
  age_idx <- row$age_idx + 1  # R is 1-indexed
  
  if (!is.na(eid_idx) && !is.na(phecode_idx) && !is.na(age_idx)) {
    Y[eid_idx + 1, phecode_idx + 1, age_idx] <- 1L  # R is 1-indexed
  }
  
  if (i %% 10000 == 0) {
    setTxtProgressBar(pb, i)
  }
}
close(pb)

#--- Create E tensor (event times) -----------------------
print("Creating E tensor...")
create_event_matrix <- function(Y_tensor) {
  n_patients <- Y_tensor$size(1)
  n_diseases <- Y_tensor$size(2)
  n_times <- Y_tensor$size(3)
  max_time <- n_times - 1
  
  E <- torch_full(c(n_patients, n_diseases), max_time, dtype = torch_int16())
  
  # Find all events
  events <- torch_nonzero(Y_tensor == 1, as_tuple = FALSE)
  
  if (events$numel() > 0) {
    for (i in seq_len(events$size(1))) {
      patient <- events[i, 1]$item() + 1  # Convert to R 1-indexed
      disease <- events[i, 2]$item() + 1
      time <- events[i, 3]$item() + 1
      
      if (time < E[patient, disease]$item()) {
        E[patient, disease] <- (time - 1)  # Store as 0-indexed
      }
    }
  }
  
  return(E)
}

E <- create_event_matrix(Y)

#--- Save outputs -----------------------------------------
output_dir <- "~/Dropbox-Personal/data_for_running/"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

torch_save(Y, file.path(output_dir, "Y_tensor.pt"))
torch_save(E, file.path(output_dir, "E_matrix.pt"))

# Also save as numpy format (for Python compatibility)
Y_np <- Y$numpy()
E_np <- E$numpy()

# Save using reticulate or write directly
library(reticulate)
np <- import("numpy")
np$savez_compressed(
  file.path(output_dir, "Y_tensor.npz"),
  Y = Y_np
)
np$savez_compressed(
  file.path(output_dir, "E_matrix.npz"),
  E = E_np
)

# Save metadata
metadata <- list(
  n_patients = n_patients,
  n_diseases = n_diseases,
  n_timepoints = n_timepoints,
  eids = eids,
  phecodes = phecodes,
  eid_index = eid_index,
  phecode_index = phecode_index
)

saveRDS(metadata, file.path(output_dir, "Y_E_metadata.rds"))

print("="*80)
print("COMPLETE")
print("="*80)
print(paste("Y tensor shape:", paste(Y$shape, collapse = " x ")))
print(paste("E tensor shape:", paste(E$shape, collapse = " x ")))
print(paste("Output directory:", output_dir))
print(paste("Files saved:"))
print(paste("  - Y_tensor.pt"))
print(paste("  - E_matrix.pt"))
print(paste("  - Y_tensor.npz"))
print(paste("  - E_matrix.npz"))
print(paste("  - Y_E_metadata.rds"))

