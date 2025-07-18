



# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(forcats)

# Function to parse the data
parse_model_data <- function(data_path) {
  # Read the data
  df <- readRDS(data_path)
  
  # Extract confidence intervals from the formatted strings
  extract_ci <- function(ci_str) {
    # Matches format like "0.765 (0.748-0.782)"
    pattern <- "(\\d+\\.\\d+)\\s*\\((\\d+\\.\\d+)-(\\d+\\.\\d+)\\)"
    matches <- regexec(pattern, ci_str)
    result <- regmatches(ci_str, matches)
    
    if(length(result) > 0 && length(result[[1]]) >= 4) {
      return(c(as.numeric(result[[1]][2]), 
               as.numeric(result[[1]][3]),
               as.numeric(result[[1]][4])))
    } else {
      return(c(NA, NA, NA))
    }
  }
  
  # Create a clean data frame
  model_data <- data.frame(
    Disease = df$Disease,
    Events = df$Events,
    Rate = df$Rate,
    stringsAsFactors = FALSE
  )
  
  # Extract values for Aladynoulli Dynamic
  dynamic_ci <- lapply(df$aladynoulli_auc_dynamic, extract_ci)
  model_data$dynamic_auc <- sapply(dynamic_ci, function(x) x[1])
  model_data$dynamic_lower <- sapply(dynamic_ci, function(x) x[2])
  model_data$dynamic_upper <- sapply(dynamic_ci, function(x) x[3])
  
  # Extract values for Aladynoulli Static
  static_ci <- lapply(df$aladynoulli_auc_static, extract_ci)
  model_data$static_auc <- sapply(static_ci, function(x) x[1])
  model_data$static_lower <- sapply(static_ci, function(x) x[2])
  model_data$static_upper <- sapply(static_ci, function(x) x[3])
  
  # Add Cox values
  model_data$cox_without_noulli <- df$cox_auc_without_noulli
  model_data$cox_with_noulli <- df$cox_auc_with_noulli

  # Add TD Cox AUC (as numeric)
  model_data$td_cox_auc <- as.numeric(df$td_cox)
  
  # Calculate CI for Cox values (using the formula SE = sqrt((AUC * (1-AUC)) / (n * prevalence)))
  calc_cox_ci <- function(auc, events, total = 400000) {
    prevalence <- events / total
    prevalence <- max(0.01, prevalence)
    se <- sqrt((auc * (1 - auc)) / (total * prevalence))
    lower <- max(0, auc - 1.96 * se)
    upper <- min(1, auc + 1.96 * se)
    return(c(lower, upper))
  }
  
  # Add CI for Cox without Noulli
  cox_without_ci <- mapply(calc_cox_ci, 
                           model_data$cox_without_noulli, 
                           model_data$Events, 
                           SIMPLIFY = FALSE)
  model_data$cox_without_lower <- sapply(cox_without_ci, function(x) x[1])
  model_data$cox_without_upper <- sapply(cox_without_ci, function(x) x[2])
  
  # Add CI for Cox with Noulli
  cox_with_ci <- mapply(calc_cox_ci, 
                        model_data$cox_with_noulli, 
                        model_data$Events, 
                        SIMPLIFY = FALSE)
  model_data$cox_with_lower <- sapply(cox_with_ci, function(x) x[1])
  model_data$cox_with_upper <- sapply(cox_with_ci, function(x) x[2])

  # Add CI for TD Cox
  td_cox_ci <- mapply(calc_cox_ci, 
                      model_data$td_cox_auc, 
                      model_data$Events, 
                      SIMPLIFY = FALSE)
  model_data$td_cox_lower <- sapply(td_cox_ci, function(x) x[1])
  model_data$td_cox_upper <- sapply(td_cox_ci, function(x) x[2])
  
  # Calculate CI for Aladynoulli Dynamic using actual sample size (e.g., 400,000)
  aladyn_dynamic_ci <- mapply(calc_cox_ci, 
                            model_data$dynamic_auc, 
                            model_data$Events, 
                            MoreArgs = list(total = 400000), 
                            SIMPLIFY = FALSE)
  model_data$dynamic_lower <- sapply(aladyn_dynamic_ci, function(x) x[1])
  model_data$dynamic_upper <- sapply(aladyn_dynamic_ci, function(x) x[2])

  # Calculate CI for Aladynoulli Static using actual sample size (e.g., 400,000)
  aladyn_static_ci <- mapply(calc_cox_ci, 
                           model_data$static_auc, 
                           model_data$Events, 
                           MoreArgs = list(total = 400000), 
                           SIMPLIFY = FALSE)
  model_data$static_lower <- sapply(aladyn_static_ci, function(x) x[1])
  model_data$static_upper <- sapply(aladyn_static_ci, function(x) x[2])
  
  # Use the value from Dynamic1year as the AUC, and extract lower/upper from X.2
  model_data$one_year_auc <- as.numeric(df$Dynamic1year)
  # Clean up dashes in X.2
  one_year_ci_str <- gsub("[\u2013\u2014]", "-", as.character(df$X.2))
  one_year_ci <- lapply(one_year_ci_str, extract_ci)
  model_data$one_year_lower <- sapply(one_year_ci, function(x) x[1])
  model_data$one_year_upper <- sapply(one_year_ci, function(x) x[2])
  
  # After extracting one_year_auc, one_year_lower, one_year_upper, recalculate CI for 1-year AUC using total=400000
  one_year_ci_calc <- mapply(calc_cox_ci, 
                            model_data$one_year_auc, 
                            model_data$Events, 
                            MoreArgs = list(total = 400000), 
                            SIMPLIFY = FALSE)
  model_data$one_year_lower <- sapply(one_year_ci_calc, function(x) x[1])
  model_data$one_year_upper <- sapply(one_year_ci_calc, function(x) x[2])
  
 
  # Calculate CI for TDC C-statistic using the same formula as for AUC
  tdc_cox_ci <- mapply(calc_cox_ci, 
                       model_data$tdc_cox_cstat, 
                       model_data$Events, 
                       MoreArgs = list(total = 400000), 
                       SIMPLIFY = FALSE)
  model_data$tdc_cox_lower <- sapply(tdc_cox_ci, function(x) x[1])
  model_data$tdc_cox_upper <- sapply(tdc_cox_ci, function(x) x[2])
  
  return(model_data)
}
