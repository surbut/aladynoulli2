# Load required libraries
library(readr)
library(dplyr)
library(tidyr)
library(knitr)
library(kableExtra)
library(formattable)

# Read the CSV file
data <- read_csv("all_offsets_results_fixedphi_from40_70.csv")

# Create the wide format table
create_disease_table <- function(data) {
  
  # Prepare data for reshaping
  data_prep <- data %>%
    select(Disease, offset, auc, n_events) %>%
    # Round AUC to 3 decimal places for better display
    mutate(auc = round(auc, 3))
  
  # Create separate tables for AUC and n_events
  auc_wide <- data_prep %>%
    select(Disease, offset, auc) %>%
    pivot_wider(
      names_from = offset, 
      values_from = auc,
      names_prefix = "Year_",
      names_sort = TRUE
    ) %>%
    # Replace NA with empty string for display
    mutate(across(-Disease, ~ifelse(is.na(.), "", as.character(.))))
  
  events_wide <- data_prep %>%
    select(Disease, offset, n_events) %>%
    pivot_wider(
      names_from = offset, 
      values_from = n_events,
      names_prefix = "Year_",
      names_sort = TRUE
    )
  
  # Combine AUC and events data by interleaving columns
  years <- sort(unique(data$offset))
  combined_table <- auc_wide[1]  # Start with Disease column
  
  for (year in years) {
    auc_col <- paste0("Year_", year)
    events_col <- paste0("Year_", year)
    
    # Add AUC column (keep as character to handle empty strings)
    combined_table[[paste0("AUC_", year)]] <- auc_wide[[auc_col]]
    
    # Add events column and create highlight indicator
    events_values <- events_wide[[events_col]]
    combined_table[[paste0("Events_", year)]] <- events_values
    
    # Create a column to track which cells should be highlighted (n_events > 5)
    combined_table[[paste0("Highlight_", year)]] <- events_values > 5
  }
  
  return(combined_table)
}

# Create the table
disease_table <- create_disease_table(data)

# Print basic table structure
cat("Table dimensions:", nrow(disease_table), "x", ncol(disease_table), "\n")
cat("Diseases included:", nrow(disease_table), "\n")
cat("Years covered:", length(unique(data$offset)), "(from", min(data$offset), "to", max(data$offset), ")\n")

# Display first few rows and columns to verify structure
print("First 5 diseases and first 6 columns:")
print(disease_table[1:5, 1:7])

# Function to create a nicely formatted table with conditional formatting
create_formatted_table <- function(disease_table, max_years_display = 10) {
  
  # Select subset of years for display (you can modify this)
  years_to_show <- 0:(max_years_display-1)
  
  # Create display table with selected years
  display_cols <- c("Disease")
  for (year in years_to_show) {
    display_cols <- c(display_cols, paste0("AUC_", year), paste0("Events_", year))
  }
  
  display_table <- disease_table[, display_cols, drop = FALSE]
  
  # Create column names for display
  col_names <- c("Disease")
  for (year in years_to_show) {
    col_names <- c(col_names, paste0("AUC Y", year), paste0("Events Y", year))
  }
  names(display_table) <- col_names
  
  return(display_table)
}

# Create a display version (showing first 10 years)
display_table <- create_formatted_table(disease_table, max_years_display = 10)
print("Display table (first 10 years):")
print(display_table)

# Function to create HTML table with highlighting using kableExtra
create_html_table <- function(disease_table, max_years_display = 10) {
  
  years_to_show <- 0:(max_years_display-1)
  
  # Prepare display data
  display_data <- disease_table[, c("Disease"), drop = FALSE]
  highlight_matrix <- matrix(FALSE, nrow = nrow(disease_table), ncol = 0)
  
  for (year in years_to_show) {
    auc_col <- paste0("AUC_", year)
    events_col <- paste0("Events_", year)
    highlight_col <- paste0("Highlight_", year)
    
    display_data[[paste0("AUC_Y", year)]] <- disease_table[[auc_col]]
    display_data[[paste0("Events_Y", year)]] <- disease_table[[events_col]]
    
    # Add highlight info
    highlight_matrix <- cbind(highlight_matrix, 
                              FALSE,  # Don't highlight AUC column
                              disease_table[[highlight_col]])  # Highlight events if > 5
  }
  
  # Create the table
  html_table <- kable(display_data, "html", escape = FALSE) %>%
    kable_styling(bootstrap_options = c("striped", "hover", "condensed"), 
                  full_width = FALSE, font_size = 12)
  
  # Apply conditional formatting for events > 5
  for (year_idx in 1:length(years_to_show)) {
    auc_col_idx <- 1 + (year_idx * 2) - 1  # Events columns are at positions 3, 5, 7, etc.
    
    # Find rows where events > 5 for this year
    highlight_rows <- which(disease_table[[paste0("Highlight_", years_to_show[year_idx])]])
    
    if (length(highlight_rows) > 0) {
      html_table <- html_table %>%
        column_spec(auc_col_idx, 
                    background = ifelse(1:nrow(disease_table) %in% highlight_rows, 
                                        "yellow", "white"))
    }
  }
  
  return(html_table)
}

# Save the full table to CSV for easy import into other tools
write_csv(disease_table, "disease_table_wide_format.csv")
cat("\nFull table saved as 'disease_table_wide_format.csv'\n")

# Example: Create HTML table (first 10 years)
if (requireNamespace("kableExtra", quietly = TRUE)) {
  html_table <- create_html_table(disease_table, max_years_display = 30)
  
  # Save HTML table
  save_kable(html_table, "disease_table_formatted.html")
  cat("Formatted HTML table saved as 'disease_table_formatted.html'\n")
}

# Summary statistics
cat("\nSummary:\n")
cat("- Total diseases:", nrow(disease_table), "\n")
cat("- Years covered:", length(unique(data$offset)), "\n")
cat("- Total AUC values:", sum(!is.na(data$auc)), "\n")
cat("- Missing AUC values:", sum(is.na(data$auc)), "\n")
cat("- Cases with >5 events:", sum(data$n_events > 5, na.rm = TRUE), "\n")

# Show which diseases have the most cases with >5 events
high_event_summary <- data %>%
  filter(n_events > 5) %>%
  group_by(Disease) %>%
  summarise(years_with_high_events = n(), .groups = 'drop') %>%
  arrange(desc(years_with_high_events))

cat("\nDiseases with most years having >5 events:\n")
print(high_event_summary)