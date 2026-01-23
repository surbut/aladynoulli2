#!/usr/bin/env Rscript

# Check alignment between Y_binary.rds and disease_names.csv

library(data.table)

cat("===== CHECKING DISEASE NAME ALIGNMENT =====\n\n")

# Load Y_binary.rds
cat("Loading Y_binary.rds...\n")
Y_ten <- readRDS("/Users/sarahurbut/Downloads/Y_binary.rds")
cat(sprintf("  Dimensions: %d rows × %d columns\n", nrow(Y_ten), ncol(Y_ten)))

# Check if Y_ten already has column names
has_colnames <- !is.null(colnames(Y_ten)) && any(colnames(Y_ten) != "")
cat(sprintf("  Column names present: %s\n", ifelse(has_colnames, "YES", "NO")))

if (has_colnames) {
  cat("  First 5 column names:", paste(head(colnames(Y_ten), 5), collapse = ", "), "\n")
  cat("  Column 59 name:", colnames(Y_ten)[59], "\n")
  cat("  Column 141 name:", colnames(Y_ten)[141], "\n")
  cat("  Column 210 name:", ifelse(length(colnames(Y_ten)) >= 210, colnames(Y_ten)[210], "N/A"), "\n")
}

# Load disease names
cat("\nLoading disease_names.csv...\n")
disease_names_dt <- fread("/Users/sarahurbut/Downloads/disease_names.csv", header = TRUE)
cat(sprintf("  Number of rows: %d\n", nrow(disease_names_dt)))
cat(sprintf("  Columns: %s\n", paste(names(disease_names_dt), collapse = ", ")))

# Extract disease names (column "x")
dxnames <- disease_names_dt[, x]
cat(sprintf("  Number of disease names: %d\n", length(dxnames)))

# Check alignment
cat("\n===== ALIGNMENT CHECK =====\n")
cat(sprintf("Y_ten columns: %d\n", ncol(Y_ten)))
cat(sprintf("Disease names: %d\n", length(dxnames)))

if (ncol(Y_ten) == length(dxnames)) {
  cat("✓ Column count matches!\n\n")
} else {
  cat("⚠️  MISMATCH: Column count doesn't match!\n\n")
}

# Check specific indices
cat("Checking specific disease indices:\n")
test_indices <- c(1, 10, 59, 141, 210, ncol(Y_ten))
for (idx in test_indices) {
  if (idx <= length(dxnames) && idx <= ncol(Y_ten)) {
    cat(sprintf("  Index %d: %s\n", idx, dxnames[idx]))
    
    # If Y_ten has column names, check if they match
    if (!is.null(colnames(Y_ten)) && colnames(Y_ten)[idx] != "") {
      if (colnames(Y_ten)[idx] == dxnames[idx]) {
        cat(sprintf("    ✓ Y_ten column name matches: %s\n", colnames(Y_ten)[idx]))
      } else {
        cat(sprintf("    ⚠️  Y_ten column name differs: %s vs %s\n", 
                    colnames(Y_ten)[idx], dxnames[idx]))
      }
    }
  }
}

# Check first and last few
cat("\nFirst 5 diseases:\n")
for (i in 1:min(5, length(dxnames))) {
  cat(sprintf("  %d: %s\n", i, dxnames[i]))
}

cat("\nDisease 59 (Acidosis check):\n")
if (59 <= length(dxnames)) {
  cat(sprintf("  %s\n", dxnames[59]))
  if (dxnames[59] == "Acidosis") {
    cat("  ✓ Correct: Acidosis is at index 59\n")
  } else {
    cat(sprintf("  ⚠️  Expected 'Acidosis' but got '%s'\n", dxnames[59]))
  }
}

cat("\nDisease 141 (Raynaud's check):\n")
if (141 <= length(dxnames)) {
  cat(sprintf("  %s\n", dxnames[141]))
  if (dxnames[141] == "Raynaud's syndrome") {
    cat("  ✓ Correct: Raynaud's syndrome is at index 141\n")
  } else {
    cat(sprintf("  ⚠️  Expected 'Raynaud\\'s syndrome' but got '%s'\n", dxnames[141]))
  }
}

cat("\nDisease 210 (Peritonitis check):\n")
if (210 <= length(dxnames)) {
  cat(sprintf("  %s\n", dxnames[210]))
  if (dxnames[210] == "Peritonitis and retroperitoneal infections") {
    cat("  ✓ Correct: Peritonitis is at index 210\n")
  } else {
    cat(sprintf("  ⚠️  Expected 'Peritonitis and retroperitoneal infections' but got '%s'\n", dxnames[210]))
  }
}

cat("\nLast 5 diseases:\n")
start_idx <- max(1, length(dxnames) - 4)
for (i in start_idx:length(dxnames)) {
  cat(sprintf("  %d: %s\n", i, dxnames[i]))
}

cat("\n===== SUMMARY =====\n")
if (ncol(Y_ten) == length(dxnames)) {
  cat("✓ Column count matches - alignment should be correct if Y_ten columns are in same order as CSV rows\n")
} else {
  cat("⚠️  Column count mismatch - alignment may be incorrect\n")
}

