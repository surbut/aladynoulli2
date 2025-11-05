#!/usr/bin/env Rscript
# Transform MGB Medication Data to Pathway Analysis Format
#
# Converts MGB medication data (with EMPI, Medication, Medication_Date) 
# into the format expected by pathway analysis code.

library(data.table)
library(lubridate)

#' Infer BNF category from medication name (simple keyword matching)
#' 
#' This is a basic implementation for common medications.
#' Many MGB medications may not match - that's fine, the analysis
#' will work with medication names directly.
#' 
#' For full BNF mapping, you'd need a proper medication dictionary or API.
infer_bnf_category <- function(drug_names) {
  # BNF category keyword mappings
  bnf_keywords <- list(
    '01' = c('omeprazole', 'lansoprazole', 'pantoprazole', 'ranitidine', 
             'metoclopramide', 'domperidone'),
    '02' = c('atorvastatin', 'simvastatin', 'rosuvastatin', 'lisinopril', 
             'ramipril', 'metoprolol', 'bisoprolol', 'atenolol', 'amlodipine',
             'losartan', 'valsartan', 'aspirin', 'clopidogrel', 'warfarin',
             'nitroglycerin', 'nitrostat'),
    '03' = c('salbutamol', 'albuterol', 'budesonide', 'fluticasone', 
             'montelukast', 'prednisone'),
    '04' = c('paracetamol', 'acetaminophen', 'ibuprofen', 'naproxen', 
             'tramadol', 'morphine', 'gabapentin', 'pregabalin', 'sertraline',
             'citalopram', 'fluoxetine'),
    '05' = c('amoxicillin', 'azithromycin', 'cephalexin', 'ciprofloxacin', 
             'metronidazole'),
    '06' = c('metformin', 'insulin', 'glipizide', 'glyburide', 
             'levothyroxine', 'synthroid'),
    '10' = c('naproxen', 'ibuprofen', 'diclofenac', 'celecoxib', 'methotrexate')
  )
  
  # Create reverse mapping
  keyword_to_bnf <- list()
  for (bnf_code in names(bnf_keywords)) {
    for (keyword in bnf_keywords[[bnf_code]]) {
      keyword_to_bnf[[tolower(keyword)]] <- bnf_code
    }
  }
  
  # Match drugs to BNF categories
  bnf_codes <- character(length(drug_names))
  for (i in seq_along(drug_names)) {
    drug_lower <- tolower(trimws(drug_names[i]))
    matched_bnf <- NA_character_
    
    # Try to find matching keyword
    for (keyword in names(keyword_to_bnf)) {
      if (grepl(keyword, drug_lower, fixed = TRUE)) {
        matched_bnf <- keyword_to_bnf[[keyword]]
        break
      }
    }
    
    bnf_codes[i] <- matched_bnf
  }
  
  return(bnf_codes)
}

#' Transform MGB medication data to pathway analysis format
#' 
#' @param mgb_med_file Path to MGB medication data file (CSV or similar)
#' @param patient_birth_dates_file Path to file with patient birth dates (optional)
#' @param output_file Path to save transformed data (optional)
#' @param patient_id_col Name of patient ID column (default: 'EMPI')
#' @param medication_col Name of medication name column (default: 'Medication')
#' @param date_col Name of medication date column (default: 'Medication_Date')
transform_mgb_medications <- function(mgb_med_file,
                                      patient_birth_dates_file = NULL,
                                      output_file = NULL,
                                      patient_id_col = 'EMPI',
                                      medication_col = 'Medication',
                                      date_col = 'Medication_Date') {
  
  cat(strrep("=", 80), "\n")
  cat("TRANSFORMING MGB MEDICATION DATA\n")
  cat(strrep("=", 80), "\n")
  
  # Load MGB medication data
  cat("\n1. Loading MGB medication data from:", mgb_med_file, "\n")
  
  # Try to auto-detect separator
  first_line <- readLines(mgb_med_file, n = 1)
  if (grepl("\t", first_line)) {
    separator <- "\t"
    cat("   Detected tab separator\n")
  } else if (grepl(",", first_line)) {
    separator <- ","
    cat("   Detected comma separator\n")
  } else {
    separator <- "|"
    cat("   Using pipe separator (default)\n")
  }
  
  mgb_meds <- fread(mgb_med_file, sep = separator, stringsAsFactors = FALSE)
  cat("   Loaded", nrow(mgb_meds), "rows\n")
  cat("   Columns:", paste(colnames(mgb_meds), collapse = ", "), "\n")
  
  # Check required columns
  required_cols <- c(patient_id_col, medication_col, date_col)
  missing_cols <- setdiff(required_cols, colnames(mgb_meds))
  if (length(missing_cols) > 0) {
    cat("   ERROR: Missing required columns:", paste(missing_cols, collapse = ", "), "\n")
    cat("   Available columns:", paste(colnames(mgb_meds), collapse = ", "), "\n")
    return(NULL)
  }
  
  # Rename columns to standard format
  cat("\n2. Renaming columns to standard format\n")
  setnames(mgb_meds, patient_id_col, 'eid')
  setnames(mgb_meds, medication_col, 'Medication_Name', skip_absent = TRUE)
  setnames(mgb_meds, date_col, 'Medication_Date', skip_absent = TRUE)
  
  # Create required columns
  cat("\n3. Creating required columns for pathway analysis\n")
  
  # drug_name: Use medication name directly (normalize for consistency)
  # Keep original names but normalize for matching/grouping
  mgb_meds[, drug_name := tolower(trimws(Medication_Name))]
  
  # read_2: Use medication name as code (MGB uses different naming than UKB)
  # This is fine - pathway analysis will work with any medication names
  mgb_meds[, read_2 := drug_name]
  
  # bnf_code: Try to infer from medication name (optional - can be NA)
  # This is just for category grouping, not required for analysis
  cat("   Inferring BNF categories from medication names (optional)...\n")
  mgb_meds[, bnf_code := infer_bnf_category(Medication_Name)]
  n_matched <- sum(!is.na(mgb_meds$bnf_code))
  n_total <- nrow(mgb_meds)
  cat("   Matched", n_matched, "of", n_total, "medications to BNF categories\n")
  cat("   (Remaining", n_total - n_matched, "will use medication names directly)\n")
  
  # Convert dates
  cat("\n4. Processing dates\n")
  mgb_meds[, Medication_Date := as.Date(Medication_Date, tryFormats = c(
    "%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d/%m/%Y", "%Y/%m/%d"
  ))]
  
  n_valid_dates <- sum(!is.na(mgb_meds$Medication_Date))
  cat("   Successfully parsed", n_valid_dates, "dates\n")
  
  # Calculate age at prescription if birth dates available
  if (!is.null(patient_birth_dates_file)) {
    cat("\n   Loading patient birth dates from:", patient_birth_dates_file, "\n")
    tryCatch({
      birth_dates <- fread(patient_birth_dates_file, stringsAsFactors = FALSE)
      
      if ('EMPI' %in% colnames(birth_dates) && 'birth_date' %in% colnames(birth_dates)) {
        birth_dates[, birth_date := as.Date(birth_date)]
        birth_dates <- birth_dates[, .(EMPI, birth_date)]
        setkey(birth_dates, EMPI)
        
        # Merge with medication data
        mgb_meds <- merge(mgb_meds, birth_dates, by.x = "eid", by.y = "EMPI", all.x = TRUE)
        mgb_meds[, age_at_prescription := as.numeric(Medication_Date - birth_date) / 365.25]
        mgb_meds[, birth_date := NULL]  # Remove temporary column
        
        n_valid_ages <- sum(!is.na(mgb_meds$age_at_prescription))
        cat("   Calculated age at prescription for", n_valid_ages, "records\n")
      } else {
        cat("   WARNING: Birth date file missing required columns (EMPI, birth_date)\n")
        mgb_meds[, age_at_prescription := NA_real_]
      }
    }, error = function(e) {
      cat("   WARNING: Could not load birth dates:", e$message, "\n")
      mgb_meds[, age_at_prescription := NA_real_]
    })
  } else {
    cat("   WARNING: No birth date file provided - age calculation skipped\n")
    mgb_meds[, age_at_prescription := NA_real_]
  }
  
  # Clean up: Remove rows with missing essential data
  initial_rows <- nrow(mgb_meds)
  mgb_meds <- mgb_meds[!is.na(eid) & !is.na(drug_name) & !is.na(Medication_Date)]
  final_rows <- nrow(mgb_meds)
  
  if (initial_rows != final_rows) {
    cat("\n5. Removed", initial_rows - final_rows, "rows with missing essential data\n")
  }
  
  # Select columns for output (matching expected format)
  output_cols <- c('eid', 'drug_name', 'read_2', 'bnf_code', 'Medication_Date', 'Medication_Name')
  if ('Clinic' %in% colnames(mgb_meds)) {
    output_cols <- c(output_cols, 'Clinic')
  }
  if ('Inpatient_Outpatient' %in% colnames(mgb_meds)) {
    output_cols <- c(output_cols, 'Inpatient_Outpatient')
  }
  if ('age_at_prescription' %in% colnames(mgb_meds)) {
    output_cols <- c(output_cols, 'age_at_prescription')
  }
  
  output_df <- mgb_meds[, ..output_cols]
  
  # Summary statistics
  cat("\n6. TRANSFORMATION SUMMARY\n")
  cat("   Total prescription records:", nrow(output_df), "\n")
  cat("   Unique patients:", length(unique(output_df$eid)), "\n")
  cat("   Unique medications:", length(unique(output_df$drug_name)), "\n")
  cat("   Medications with BNF codes:", sum(!is.na(output_df$bnf_code)), 
      "(", round(100 * sum(!is.na(output_df$bnf_code)) / nrow(output_df), 1), "%)\n")
  cat("   Date range:", as.character(min(output_df$Medication_Date, na.rm = TRUE)), 
      "to", as.character(max(output_df$Medication_Date, na.rm = TRUE)), "\n")
  
  if ('age_at_prescription' %in% colnames(output_df)) {
    valid_ages <- output_df$age_at_prescription[!is.na(output_df$age_at_prescription)]
    if (length(valid_ages) > 0) {
      cat("   Age range:", round(min(valid_ages), 1), "to", round(max(valid_ages), 1), "years\n")
    }
  }
  
  # Save output
  if (is.null(output_file)) {
    output_file <- gsub("\\.(csv|txt)$", "_transformed.csv", mgb_med_file)
  }
  
  cat("\n7. Saving transformed data to:", output_file, "\n")
  fwrite(output_df, output_file, sep = "\t")
  cat("   Saved", nrow(output_df), "rows\n")
  
  return(output_df)
}

# Command line interface
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) < 1) {
    cat("Usage: Rscript transform_mgb_medications.R <input_file> [birth_dates_file] [output_file]\n")
    cat("\nExample:\n")
    cat("  Rscript transform_mgb_medications.R mgb_medications.csv\n")
    cat("  Rscript transform_mgb_medications.R mgb_medications.csv birth_dates.csv mgb_meds_transformed.csv\n")
    quit(status = 1)
  }
  
  input_file <- args[1]
  birth_dates_file <- if (length(args) >= 2) args[2] else NULL
  output_file <- if (length(args) >= 3) args[3] else NULL
  
  result <- transform_mgb_medications(
    mgb_med_file = input_file,
    patient_birth_dates_file = birth_dates_file,
    output_file = output_file
  )
  
  if (!is.null(result)) {
    cat("\nSUCCESS: Transformation complete!\n")
    cat("\nFirst few rows of transformed data:\n")
    print(head(result))
  } else {
    cat("\nERROR: Transformation failed\n")
    quit(status = 1)
  }
}

