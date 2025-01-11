library(tidyverse)
library(reticulate)

# 2. Activate the environment
use_condaenv("r-tensornoulli")

# 3. Install required packages
#conda_install("r-tensornoulli", packages = c("numpy", "pandas", "scipy"))

# 4. Verify installation
py_config()  # Should show your new environment


# Load the saved model data
model_data <- py_load_object("~/Dropbox (Personal)/aladyn_model_full.pkl")

pi_pred=model_data$pi_pred
phi=model_data$phi
lambda=model_data$lambda
event=model_data$event_times
event_times=event+1

# Function to calculate remaining lifetime risk from annual transitions
calc_remaining_risk <- function(pi_pred, from_age) {
  # Convert from 3D array to more manageable form
  N <- dim(pi_pred)[1]
  D <- dim(pi_pred)[2]
  T <- dim(pi_pred)[3]
  
  # For each person and disease, calculate probability of developing disease 
  # between from_age and end of follow-up
  remaining_risk <- array(0, dim=c(N, D))
  
  for(n in 1:N) {
    for(d in 1:D) {
      # Get annual transition probabilities from from_age onwards
      yearly_pi <- pi_pred[n, d, from_age:T]
      # Probability of developing = 1 - prob of never developing
      remaining_risk[n,d] <- 1 - prod(1 - yearly_pi)
    }
  }
  
  return(remaining_risk)
}

# Calculate remaining risk at different ages
ages <- seq(30, 75, by=5)
aladyn_risks <- list()

for(age in ages) {
  time_idx <- age - 29  # Convert age to time index
  aladyn_risks[[as.character(age)]] <- calc_remaining_risk(model_data$pi_pred, time_idx)
}


library(survival)


# Function to fit Cox model and get predicted risks
# Function to fit Cox model and get predicted risks

model_data$metadata$smoke[is.na(model_data$metadata$smoke)]=0
# Function to fit Cox model and get predicted risks
fit_cox_risk <- function(event_times, event, sex, smoke, from_age) {
  N <- length(event_times)
  all_risks <- rep(NA, N)  # Initialize risks for everyone
  
  # Create survival data
  df <- data.frame(
    id = 1:N,
    time = as.numeric(event_times),
    event = event,
    sex = sex,
    smoke = smoke
  )
  
  # Get indices of people still at risk
  at_risk_idx <- which(df$time > from_age)
  
  if(length(at_risk_idx) > 0) {
    # Create filtered dataset
    df_risk <- df[at_risk_idx,]
    df_risk$time <- df_risk$time - from_age
    
    # Fit Cox and get risks for at-risk people
    cox_fit <- coxph(Surv(time, event) ~ sex + smoke, data=df_risk)
    at_risk_risks <- 1 - exp(-predict(cox_fit, type="expected"))
    
    # Assign risks back to full population
    all_risks[at_risk_idx] <- at_risk_risks
  }
  
  return(all_risks)
}

# Calculate risks at each age
cox_risks <- list()
for(age in ages) {
  print(age)
  time_idx <- age - 29
  cox_risks[[as.character(age)]] <- matrix(NA, nrow=N, ncol=D)

  
  for(d in 1:D) {
    print(d)
    cox_risks[[as.character(age)]][,d] <- fit_cox_risk(
      event_times = model_data$event_times[,d]+1,
      event = rowSums(Y[,d,time_idx:T]),  # Sum future events
      sex = model_data$metadata$sex,
      smoke = model_data$metadata$smoke,
      from_age = time_idx
    )
  }
}