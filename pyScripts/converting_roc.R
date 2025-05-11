sigmoid <- function(x) {
  1/(1 + exp(-x))
}
evaluate_major_diseases_wsex <- function(params, Y, disease_names, pce_df, follow_up_duration_years=10) {
  # Define disease groups
  major_diseases <- list(
    'ASCVD' = c('Myocardial infarction', 'Coronary atherosclerosis', 
                'Other acute and subacute forms of ischemic heart disease',
                'Unstable angina (intermediate coronary syndrome)', 
                'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'),
    'Diabetes' = c('Type 2 diabetes'),
    'Atrial_Fib' = c('Atrial fibrillation and flutter'),
    'CKD' = c('Chronic renal failure [CKD]', 'Chronic Kidney Disease, Stage III'),
    'All_Cancers' = c('Colon cancer', 'Cancer of bronchus; lung', 
                      'Breast cancer [female]', 'Cancer of prostate', 
                      'Malignant neoplasm of bladder', 'Secondary malignant neoplasm'),
    'Stroke' = c('Cerebral artery occlusion, with cerebral infarction', 'Cerebral ischemia'),
    'Heart_Failure' = c('Congestive heart failure (CHF) NOS', 'Heart failure NOS'),
    'Breast_Cancer' = c('Breast cancer [female]','Malignant neoplasm of female breast'),  # Sex-specific
    'Prostate_Cancer' = c('Cancer of prostate'),    # Sex-specific
    'Lung_Cancer' = c('Cancer of bronchus; lung'),
    'Bladder_Cancer'= c('Malignant neoplasm of bladder'),
    'Secondary_Cancer'=c('Secondary malignant neoplasm', 'Secondary malignancy of lymph nodes', 'Secondary malignancy of respiratory organs', 'Secondary malignant neoplasm of digestive systems'),
    'Depression' = c('Major depressive disorder'),
    'Anxiety'=c('Anxiety disorder'),
    'Bipolar_Disorder'= c('Bipolar'),
    'Rheumatoid_Arthritis'=c('Rheumatoid arthritis'),
    'Psoriasis'=c('Psoriasis vulgaris'),
    'Ulcerative_Colitis'=c('Ulcerative colitis'),
    'Crohns_Disease'=c('Regional enteritis'),
    'Asthma'=c('Asthma'),
    'Parkinsons'=c("Parkinson's disease"),
    'Multiple_Sclerosis'=c('Multiple sclerosis'),
    'Thyroid_Disorders'=c('Thyrotoxicosis with or without goiter', 'Secondary hypothyroidism', 'Hypothyroidism NOS')
  )
  
  # Input validation
  if(!'Sex' %in% names(pce_df)) stop("'Sex' column not found in pce_df")
  if(!'age' %in% names(pce_df)) stop("'age' column not found in pce_df")
  
  # Get dimensions once
  lambda <- params$lambda  # N x K x T
  phi <- params$phi      # K x D x T
  kappa <- params$kappa  # scalar
  
  N <- dim(Y)[1]
  D <- dim(Y)[2]
  T <- dim(Y)[3]
  K <- dim(phi)[1]
  
  # Calculate theta using softmax across K dimension
  softmax_by_k <- function(x) {
    exp_x <- exp(x)
    sum_exp <- apply(exp_x, c(1,3), sum)  
    sweep(exp_x, c(1,3), sum_exp, '/')
  }
  theta <- softmax_by_k(lambda)  # N x K x T
  
  results <- list()
  
  for(disease_group in names(major_diseases)) {
    cat(sprintf("\nEvaluating %s (%d-Year Outcome, 1-Year Score)...\n", 
                disease_group, follow_up_duration_years))
    
    # Get disease indices
    disease_list <- major_diseases[[disease_group]]
    disease_indices <- integer(0)
    for(disease in disease_list) {
      matches <- grep(tolower(disease), tolower(disease_names), value=FALSE)
      disease_indices <- unique(c(disease_indices, matches))
    }
    cat(sprintf("\nGroup: %s\n", disease_group))
    cat("Indices: ", disease_indices, "\n")
    cat("Names: ", paste(disease_names[disease_indices], collapse=", "), "\n")
    if(length(disease_indices) == 0) {
      cat(sprintf("No valid matching disease indices found for %s.\n", disease_group))
      results[[disease_group]] <- list(auc=NA, n_events=0, event_rate=0)
      next
    }
    
    # Sex filtering
    target_sex <- NULL
    if(disease_group == "Breast_Cancer") target_sex <- "Female"
    if(disease_group == "Prostate_Cancer") target_sex <- "Male"
    if(!is.null(target_sex)) {
      mask_pce <- pce_df$Sex == target_sex
      cat(sprintf("Filtering for %s: Found %d individuals in cohort\n", 
                  target_sex, sum(mask_pce)))
      if(sum(mask_pce) == 0) {
        results[[disease_group]] <- list(auc=NA, n_events=0, event_rate=0)
        next
      }
    } else {
      mask_pce <- rep(TRUE, nrow(pce_df))
    }
    int_indices_pce <- which(mask_pce)
    
    # Slice all arrays to the same set of individuals
    current_theta <- theta[int_indices_pce,, , drop=FALSE]
    current_Y <- Y[int_indices_pce,, , drop=FALSE]
    current_pce_df <- pce_df[int_indices_pce,]
    current_N <- length(int_indices_pce)
    
    # Calculate probabilities for this disease group
    # Only for the selected individuals and diseases
    pi_pred <- array(0, dim=c(current_N, length(disease_indices), T))
    for(t in 1:T) {
      for(d_idx in seq_along(disease_indices)) {
        disease_phi <- phi[, disease_indices[d_idx], t]  # K x 1
        logit_phi <- sigmoid(disease_phi)

        pi_pred[, d_idx, t] <- as.numeric(kappa) * (current_theta[,,t] %*% logit_phi)
      }
    }
    
    risks <- numeric(current_N)
    outcomes <- numeric(current_N)
    processed_indices <- integer(0)
    
    for(i in seq_len(current_N)) {
      age <- current_pce_df$age[i]
      t_enroll <- as.integer(age - 30)
      if(t_enroll < 0 || t_enroll >= T) next
      
      # Calculate 1-year risk for this disease group
      disease_probs <- pi_pred[i, , t_enroll + 1]
      risks[i] <- 1 - prod(1 - disease_probs)
      
      # Check outcome in follow-up window
      end_time <- min(t_enroll + follow_up_duration_years, T)
      if(end_time < t_enroll) next
      
      # Look for any event in the disease group
      for(d_idx in seq_along(disease_indices)) {
        if(any(current_Y[i, disease_indices[d_idx], (t_enroll + 1):end_time] > 0)) {
          outcomes[i] <- 1
          break
        }
      }
      processed_indices <- c(processed_indices, i)
    }
    
    # Calculate metrics
    if(length(processed_indices) > 0) {
      risks_processed <- risks[processed_indices]
      outcomes_processed <- outcomes[processed_indices]
      n_processed <- length(processed_indices)

      # --- DEBUG PRINTS: Add here ---
      if (disease_group == "Bipolar_Disorder" || disease_group == "Depression") {
        cat("\n--- DEBUG:", disease_group, "---\n")
        cat("Risks:\n")
        print(risks_processed)
        cat("Outcomes:\n")
        print(outcomes_processed)
        cat("-------------------------------\n")
      }
      # --- END DEBUG PRINTS ---

      if(length(unique(outcomes_processed)) > 1) {
        require(pROC)
        auc_score <- auc(outcomes_processed, risks_processed,direction="<")
      } else {
        auc_score <- NA
        cat("Warning: Only one class present for AUC.\n")
      }
      n_events <- sum(outcomes_processed)
      event_rate <- (n_events / n_processed) * 100

      if (disease_group == "Bipolar_Disorder" || disease_group == "Depression") {
        df <- data.frame(risk=risks_processed, outcome=outcomes_processed)
        write.csv(df, paste0("debugRsoft_", disease_group, ".csv"), row.names=FALSE)
      }
    } else {
      auc_score <- NA
      n_events <- 0
      event_rate <- 0
      n_processed <- 0
    }
    
    results[[disease_group]] <- list(
      auc = auc_score,
      n_events = n_events,
      event_rate = event_rate,
      n_processed = n_processed
    )
    
    cat(sprintf("AUC (Score: 1-Yr Risk, Outcome: %d-Yr Event): %s (calculated on %d individuals)\n",
                follow_up_duration_years,
                ifelse(is.na(auc_score), "N/A", sprintf("%.3f", auc_score)),
                n_processed))
    cat(sprintf("Events (%d-Year in Eval Cohort): %d (%.1f%%) (from %d individuals)\n",
                follow_up_duration_years, n_events, event_rate, n_processed))
  }
  
  # Print summary table
  cat(sprintf("\nSummary of Results (Prospective %d-Year Outcome, 1-Year Score, Sex-Adjusted):\n",
              follow_up_duration_years))
  cat(paste(rep("-", 60), collapse=""), "\n")
  cat(sprintf("%-20s %-8s %-10s %-10s\n", "Disease Group", "AUC", "Events", "Rate (%)"))
  cat(paste(rep("-", 60), collapse=""), "\n")
  
  for(group in names(results)) {
    res <- results[[group]]
    auc_str <- ifelse(is.na(res$auc), "N/A", sprintf("%.3f", res$auc))
    rate_str <- ifelse(is.null(res$event_rate), "N/A", sprintf("%.1f", res$event_rate))
    cat(sprintf("%-20s %-8s %-10d %-10s\n", 
                group, auc_str, res$n_events, rate_str))
  }
  
  return(results)
}

# Use the function

pce_df = readRDS('/Users/sarahurbut/Dropbox/pce_df_prevent.rds')
ukb_params=readRDS("/Users/sarahurbut/aladynoulli2/pyScripts/big_stuff/ukb_params_enrollment.rds")
ukb_results <- evaluate_major_diseases_wsex(
  params = ukb_params,
  Y = ukb_params$Y[pce_df$Sex%in%"Male",,],
  disease_names = as.character(ukb_params$disease_names[,1]),
  pce_df = pce_df[pce_df$Sex=="Male",],
  follow_up_duration_years = 10
)


###

df=read.csv("model_comparison_results.csv")


library(ggplot2)
library(dplyr)
library(stringr)
library(readr)

df <- df %>%
  mutate(
    aladyn_auc = as.numeric(str_extract(Aladynoulli_AUC, "^[0-9.]+")),
    aladyn_low = as.numeric(str_extract(Aladynoulli_AUC, "(?<=\\()[0-9.]+")),
    aladyn_high = as.numeric(str_extract(Aladynoulli_AUC, "(?<=-)[0-9.]+(?=\\))")),
    cox_auc = as.numeric(str_extract(Cox_AUC, "^[0-9.]+")),
    cox_low = as.numeric(str_extract(Cox_AUC, "(?<=\\()[0-9.]+")),
    cox_high = as.numeric(str_extract(Cox_AUC, "(?<=-)[0-9.]+(?=\\))")),
    Rate_num = as.numeric(str_remove(Rate, "%")),
    aladyn_se = (aladyn_high - aladyn_low) / (2 * 1.96),
    cox_se = (cox_high - cox_low) / (2 * 1.96),
    z = (aladyn_auc - cox_auc) / sqrt(aladyn_se^2 + cox_se^2),
    p = 2 * (1 - pnorm(abs(z))),
    significance = case_when(
      p < 0.001 ~ "***",
      p < 0.01 ~ "**",
      p < 0.05 ~ "*",
      TRUE ~ ""
    ),
    model_better = case_when(
      aladyn_auc > cox_auc ~ "Aladynoulli",
      cox_auc > aladyn_auc ~ "Cox",
      TRUE ~ "Equal"
    )
  )

# Sort by event rate
# Sort by event rate
df <- df %>% arrange((Rate_num))
df$event_label <- paste0(df$Rate, " (", df$Events*39, ")")
df$Disease <- factor(df$Disease, levels = df$Disease)
df=df[df$Disease%in%c("ASCVD","Diabetes","Anemia","All_Cancers","Prostate_Cancer","Depression","COPD","Atrial_Fib",
                      "Breast_Cancer","Anxiety","Osteoporosis","Heart Failure","CKD","Stroke","Rheumatoid_Arthritis","Colorectal_Cacner","Lung_Cancer","Bladder_Cancer",
                      "Ulcerative_Colitis","Parkinsons"),]
# Plot
pauc2=ggplot(df, aes(x = Disease))+lims(y=c(0.35,1))+
  geom_segment(aes(y = cox_auc, yend = aladyn_auc, xend = Disease,
                   color = model_better), size = 1.2, alpha = 0.5) +
  geom_errorbar(aes(ymin = aladyn_low, ymax = aladyn_high), width = 0.15,
                color = "#d62728", size = 0.6) +
  geom_errorbar(aes(ymin = cox_low, ymax = cox_high), width = 0.15,
                color = "#1f77b4", size = 0.6) +
  geom_point(aes(y = aladyn_auc, shape = "Aladynoulli"), color =  "#d62728", size = 3) +
  geom_text(aes(y = 0.85, label = event_label), hjust = 0, size = 4, color = "black") +
  geom_point(aes(y = cox_auc, shape = "Cox"), color ="#1f77b4", size = 3) +
  geom_text(aes(y = pmax(aladyn_auc, cox_auc) + 0.02, label = significance), size = 4) +
  scale_shape_manual(values = c("Aladynoulli" = 16, "Cox" = 17)) +
  scale_color_manual(values = c("Aladynoulli" ="#d62728", "Cox" = "#1f77b4",  "Equal" = "gray")) +
  labs(title = "AUC Comparison Across Diseases",
       subtitle = "Sorted by Event Rate with 95% Confidence Intervals",
       y = "AUC", x = NULL, shape = "Model", color = "Better Model") +
  coord_flip() +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top",
        plot.title = element_text(face = "bold", size = 16),
        axis.text.y = element_text(size = 11))


ggsave(plot = pauc2,filename = "paucplot.pdf",width=15)

# Sort by event rate
df <- df %>% arrange((Rate_num))
df$Disease <- factor(df$Disease, levels = df$Disease)
df=df[df$Disease%in%c("ASCVD","Diabetes","Anemia","All_Cancers","Prostate_Cancer","Depression","COPD","Atrial_Fib",
                   "Breast_Cancer","Anxiety","Osteoporosis","Heart Failure","CKD","Stroke","Rheumatoid_Arthritis","Colorectal_Cacner","Lung_Cancer","Bladder_Cancer",
                   "Ulcerative_Colitis","Parkinsons"),]
# Plot
pauc=ggplot(df, aes(x = Disease)) +
  geom_segment(aes(y = cox_auc, yend = aladyn_auc, xend = Disease,
                   color = model_better), size = 1.2, alpha = 0.5) +
  geom_point(aes(y = aladyn_auc, shape = "Aladynoulli"), color = "#1f77b4", size = 3) +
  geom_point(aes(y = cox_auc, shape = "Cox"), color = "#d62728", size = 3) +
  geom_text(aes(y = pmax(aladyn_auc, cox_auc) + 0.015, label = significance), size = 4) +
  scale_shape_manual(values = c("Aladynoulli" = 16, "Cox" = 17)) +
  scale_color_manual(values = c("Aladynoulli" = "#1f77b4", "Cox" = "#d62728", "Equal" = "gray")) +
  labs(title = "AUC Comparison Across Diseases",
       subtitle = "Sorted by Event Rate; Significance Based on AUC Difference",
       y = "AUC", x = NULL, shape = "Model", color = "Better Model") +
  coord_flip() +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top",
        plot.title = element_text(face = "bold", size = 16),
        axis.text.y = element_text(size = 11))

ggsave(plot = pauc,filename = "paucplot.pdf")

library(ggplot2)
library(patchwork)
library(viridis)
library(dplyr)


# Install if needed
## restart R, only need to do once
library(reticulate)

library(pals)
## convert ro R
#use_condaenv("r-tensornoulli")
use_condaenv("/opt/miniconda3/envs/new_env_pyro2", required = TRUE)
torch <- import("torch")
tensor_to_r <- function(tensor) {
  as.array(tensor$detach()$cpu()$numpy())
}



library(reticulate)
torch <- import("torch")

# Load the tensor
Y_train_tensor <- torch$load("big_stuff/Y_train_tensor.pt")

# Convert to R array
Y_train_array <- as.array(Y_train_tensor$detach()$cpu()$numpy())

# Save as RDS
saveRDS(Y_train_array, "Y_train_tensor.rds")

E_full_tensor=torch$load("E_full_tensor.pt")
E=tensor_to_r(E_full_tensor)
dim(E)
saveRDS(E,"E_full_tensor.rds")

