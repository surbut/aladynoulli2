# Load necessary library
library(dplyr)
library(data.table)
#expanded_data = fread("~/Dropbox (Personal)/forapp/patient_diagnoses_expanded_data.csv")

expanded_data=readRDS("~/Dropbox (Personal)/icdstuff_forARRAY.rds")


# Read data (assuming the data is in a CSV file called 'data.csv')
df <- expanded_data


# Define unique identifiers
unique_eids <- unique(df$eid)
unique_diseases <- unique(df$diag_icd10)

#unique_time_bins <- sort(unique(df$age_diag))
unique_time_bins <- sort(unique(df$round))
# Create mappings
eid_to_index <- setNames(1:length(unique_eids), unique_eids)
disease_to_index <- setNames(1:length(unique_diseases), unique_diseases)
time_bin_to_index <- setNames(1:length(unique_time_bins), unique_time_bins)

# Initialize the array
N <- length(unique_eids)
D <- length(unique_diseases)
T <- length(unique_time_bins)

array <- array(0, dim = c(N, D, T), dimnames = list(unique_eids, unique_diseases, unique_time_bins))

# Populate the array using names directly
for (i in seq_len(nrow(df))) {
  eid <- as.character(df$eid[i])
  disease <- as.character(df$diag_icd10[i])
  time_bin <- as.integer(time_bin_to_index[as.character(df$age_diag[i])])


  array[eid, disease, time_bin] <- 1
}
saveRDS(array,"disease_array_incidence.rds")
#saveRDS(array,"disease_array.rds")
## mean disease count over lifetime for person
person_means=apply(array,1,function(x) mean(x))
summary(person_means)
## mean person count over lifetime for disease
disease_means=apply(array,2,function(x) mean(x))
summary(disease_means)

###



## mean disease count per year for person
person_means=apply(array,c(1,3),function(x) mean(x))
summary(person_means)
## mean person count per year  for disease
disease_means=apply(array,c(2,3),function(x) mean(x))
summary(disease_means)


# Sample 10 people
sampled_people <- sample(rownames(array), 10)

# Sample 10 diseases
sampled_diseases <- sample(colnames(array), 10)

# Filter for sampled people
person_means_sampled <- person_means[sampled_people, ]

# Filter for sampled diseases
disease_means_sampled <- disease_means[sampled_diseases, ]


library(reshape2)
# Convert person_means_sampled to a data frame
person_means_df <- melt(person_means_sampled)
names(person_means_df) <- c("person", "year", "mean_disease_count")


# Convert disease_means_sampled to a data frame
disease_means_df <- melt(disease_means_sampled)
names(disease_means_df) <- c("disease", "year", "mean_person_count")


#image(t(as.matrix(s)))
# Load ggplot2
library(ggplot2)
library(tidyr)


# Plot mean disease count per year for each sampled person
ggplot(person_means_df, aes(x = as.numeric(year), y = mean_disease_count, color = as.factor(person))) +
  geom_smooth() +facet_wrap(~person)+
  labs(title = "Mean Disease Count per Year for Each Sampled Person",
       x = "Year",
       y = "Mean Disease Count") +
  theme_classic()

# Plot mean person count per year for each sampled disease
ggplot(disease_means_df, aes(x = as.numeric(year), y = mean_person_count, color = as.factor(disease))) +
    geom_smooth() +facet_wrap(~disease)+
  labs(title = "Mean Person Count per Year for Each Sampled Disease",
       x = "Year",
       y = "Mean Person Average") +theme_classic()



# and has dimensions [individuals, diseases, time]

# Reshape the data
long_data <- as.data.frame.table(array) %>%
  rename(Individual = Var1, Disease = Var2, Time = Var3, Value = Freq) %>%
  mutate(Time = as.numeric(as.character(Time)))

# Create the plot
ggplot(long_data[long_data$Disease%in%sample(long_data$Disease,10),], aes(x = Time, y = Individual, fill = Value)) +
  geom_tile() +
  facet_wrap(~ Disease, scales = "free_y") +
  scale_fill_viridis(option = "plasma") +
  theme_minimal() +
  labs(title = "Disease Progression Over Time for Each Individual",
       x = "Time", y = "Individual", fill = "Value") +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        panel.grid = element_blank()) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0))


# Load ggplot2
library(ggplot2)


# Calculate the PER YEAR SUMfor a disease
s <- apply(array, MARGIN = c(2, 3), FUN = mean)
summary(s)
summary(-log(s))
# Normalize the sums to percentages

library(reshape2)
s_df=melt(s)
colnames(s_df) <- c("Disease", "Year", "MeanCount")

# Scale the MeanCount to a more visible range
s_df$ScaledMeanCount <- s_df$MeanCount/max(s_df$MeanCount) * 100  # Scaling factor can be adjusted

# Create the heatmap
ggplot(s_df, aes(x = as.numeric(Year), y = as.factor(Disease), fill =ScaledMeanCount)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue", limits = c(0, max(s_df$ScaledMeanCount)), oob = scales::squish) +
  labs(title = "Disease Means Over Time",
       x = "Year",
       y = "Disease",
       fill = "Mean Count")+
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),axis.text.y =element_blank())

####

## simulate according to empiricial covariance

logit=function(p){exp(p/(1-p))}
library(mvtnorm)
mu_it=rmvnorm(n=10,mean=rep(logit(mean(array[1,,])),dim(array)[3]),sigma=cov(array[1,,]))
mu_dt=rmvnorm(n=10,mean=rep(logit(mean(array[,1,])),dim(array)[3]),sigma=cov(array[,1,]))
matplot(t(mu_it),type="l",col=1:10,main="Individuals")
matplot(t(mu_dt),type="l",col=1:10,main="Diseases")

time_slices=11
K=10
d <- list()
for (i in 1:time_slices) {
  file <- data.frame(t(as.matrix(fread(paste0("~/Dropbox (Personal)/UKB_topic_app/topic_term_probs_time_slice_", i-1, ".csv")))))
  names(file) <- c("diag_icd10", paste0("Topic", seq(1:K)))
  df <- data.frame(merge(file, ATM::disease_info_phecode_icd10[, c("phecode", "phenotype")], by.x = "diag_icd10", by.y = "phecode"))
  d[[i]] <- df
}

time_slice_labels <- c("30-35", "35-40", "40-45", "45-50", "50-55", "55-60", "60-65", "65-70", "70-75", "75-80", "80+")
b=bind_rows(lapply(1:time_slices, function(t) {
  lapply(1:K, function(topic_id){
  topic_col <- sym(paste0("Topic", topic_id))
  d[[t]] %>%
    select(diag_icd10, phenotype, !!topic_col) %>%
    mutate(time_slice = time_slice_labels[t],topic=topic_id) %>%  # Use the labels here
    rename(prob = !!topic_col)
})}))



library(kernlab)
gp_pred=array(0,dim=c(length(unique(b$phenotype)),length(unique(b$topic)),length(seq(30,80,5))))
for(p in 1:length(unique(b$phenotype))){

  for(k in 1:length(unique(b$topic))){
  pheno=unique(b$phenotype)[p]
  x=seq(30,80,5)
  fit=kernlab::gausspr(y=b$prob[b$phenotype==pheno&b$topic==k], x=x)
  gppred=kernlab::predict(fit,x)
  gp_pred[p,k,]=gppred
  }
}

gp_pred[is.na(gp_pred)]=0

result <- b %>%
  group_by(phenotype) %>%
  filter(prob == max(prob)) %>%
  distinct(diag_icd10, phenotype, .keep_all = TRUE)%>%arrange(desc(prob))

goodmat=NULL
for(i in 1:nrow(result)){
  p=result[i,]$phenotype
  k=result[i,]$topic
  g=b[b$phenotype==p&b$topic==k,]
  goodmat=rbind(g,goodmat)
}

ggplot(goodmat, aes(x = as.factor(time_slice), y = as.factor(phenotype), fill = prob)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = median(b$prob)) +
  labs(title = paste("Heatmap of Predicted Prob for Max Topic for each disease"),
       x = "Time Slice",
       y = "Phenotype") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1), axis.text.y = element_blank())


s=sample(result$phenotype[1:100],1)
ggplot(b[b$phenotype%in%s,],aes(x = as.factor(time_slice), y = as.factor(phenotype), fill = prob)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = median(b$prob)) +
  labs(title = paste("Heatmap of Predicted Prob across topics for",s),
       x = "Time Slice",
       y = "Phenotype") +
  theme_minimal() +facet_wrap(~topic)+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1), axis.text.y = element_blank())


set=result$phenotype[1:10]
ggplot(b[b$phenotype%in%set,],aes(x = as.factor(time_slice), y = prob, col = as.factor(phenotype),group=as.factor(phenotype))) +
  geom_smooth() +
  labs(title = paste("Heatmap of Predicted Prob across topics for Disease"),
       x = "Time Slice",
       y = "Phenotype") +
  theme_minimal() +facet_wrap(~topic)+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1), axis.text.y = element_blank())




library(kernlab)
library(ggplot2)
library(tidyr)

# Create the array to store the predictions
gp_pred <- array(0, dim = c(length(unique(b$phenotype)), length(unique(b$topic)), length(seq(30, 80, 5))))

# Get the unique phenotypes and topics
phenotypes <- result$phenotype[1:10]
topics <- result$topic[1:10]
time_points <- seq(30, 80, 5)

# Prepare a data frame to store the actual and predicted values for plotting
plot_data <- data.frame()

for (p in 1:length(phenotypes)) {
  #for (k in 1:length(topics)) {
    pheno <- phenotypes[p]
    topic <- topics[p]

    # Extract the data for the current phenotype and topic
    sub_data <- b[b$phenotype == pheno & b$topic == topic, ]

    if (nrow(sub_data) > 0) {
      # Ensure the time points are correctly matched with probabilities
      x <- time_points
      y <- sub_data$prob

      # Fit the Gaussian process model
      fit <- gausspr(x = x, y = y)

      # Predict using the Gaussian process model
      gppred <- predict(fit, time_points)



      # Add the actual and predicted values to the plot data
      plot_data <- rbind(plot_data, data.frame(
        phenotype = pheno,
        topic = topic,
        time_slice = x,
        prob_actual = y,
        prob_pred = gppred
      ))
    }
  }


# Convert phenotype and topic to factors for plotting
plot_data$phenotype <- as.factor(plot_data$phenotype)
plot_data$topic <- as.factor(plot_data$topic)

# Gather the actual and predicted values for ggplot
plot_data_long <- plot_data %>%
  pivot_longer(cols = c(prob_actual, prob_pred), names_to = "type", values_to = "prob")

# Plot using ggplot2
ggplot(plot_data_long, aes(x = time_slice, y = prob, color = type,group = type)) +
  geom_point() +
  geom_smooth() +
  facet_wrap(~ phenotype, scales = "free_y") +
  labs(title = "Actual vs Predicted Probabilities",
       x = "Time Slice",
       y = "Probability") +
  theme_minimal() +
  theme(legend.position = "top")


## real data
array=readRDS("~/Dropbox (Personal)/UKB_topic_app/disease_array.rds")
library(kernlab)
library(ggplot2)
library(reshape2)
library(dplyr)

# Assuming `real_data` is the array with dimensions N x D x T
# Example: real_data <- array(0, dim = c(N, D, T))

# Prepare a data frame to store the actual and predicted values for plotting
plot_data <- data.frame()

# Get dimensions
N <- dim(array)[1]
D <- dim(array)[2]
T <- dim(array)[3]

# Define time points
time_points <- seq(1, T)


# Function to fit GP and predict
fit_gp_and_predict <- function(y, x, new_x) {
  fit <- gausspr(x = x, y = y)
  predict(fit, new_x)
}


#array=readRDS("~/Dropbox (Personal)/UKB_topic_app/disease_array.rds")

array=readRDS("~/Desktop/disease_array_incidence.rds")
D=dim(array)[2]

datnames=data.frame(dimnames(array)[[2]])
names(datnames)=c("phecode")
phenonames=merge(datnames,ATM::disease_info_phecode_icd10,by="phecode")[,"phenotype"]

plot_data <- data.frame()

for (d in 1:D) {
  # Compute the average probability for each time point across individuals
  avg_prob <- colMeans(array[, d, ])

  # Fit GP and predict
  gppred <- fit_gp_and_predict(avg_prob, time_points, time_points)

  # Store the actual and predicted values in the plot_data
  plot_data <- rbind(plot_data, data.frame(
    disease = phenonames[d],
    time_slice = time_points,
    prob_actual = avg_prob,
    prob_pred = gppred
  ))
}

lab=data.frame(time_slice=seq(1:11),time_slice_labels)
mf=merge(plot_data,lab,by="time_slice")
write.csv(mf,"~/Desktop/disease_plot_data_frequencies.csv")
y=array(0,dim=c(length(unique(phenonames)),10000,length(time_points)))


for(i in 1:length(unique(disease_plot_data$disease))){
  df=disease_plot_data[disease_plot_data$disease==unique(disease_plot_data$disease)[i],]
  for(t in 1:length(unique(disease_plot_data$time_slice))){

    y[i,,t]=rbinom(10000,1,prob = max(0,df$prob_pred[t]))
  }
}

cor(colMeans(y[1,,]),disease_plot_data[disease_plot_data$disease%in%unique(disease_plot_data$disease)[1],"prob_pred"])

n=matrix(0,nrow(plot_data),1000)
for(i in 1:nrow(disease_plot_data)){
  n[,i]=rbinom(1000,1,prob = max(0,disease_plot_data$prob_pred[i]))
}


head(plot_data)
disease_plot_data=plot_data
# Gather the actual and predicted values for ggplot
plot_data_long <- plot_data %>%
  pivot_longer(cols = c(prob_actual, prob_pred), names_to = "type", values_to = "prob")

# Plot using ggplot2
ggplot(plot_data_long, aes(x = time_slice, y = prob, color = type)) +
  geom_point() +
  geom_smooth() +
  facet_wrap(~ disease, scales = "free_y") +
  labs(title = "Actual vs Predicted Average Probabilities Across Individuals",
       x = "Time Slice",
       y = "Probability") +
  theme_minimal() +
  theme(legend.position = "top")
###

plot_data <- data.frame()

# Get dimensions
N <- dim(array)[1]
D <- dim(array)[2]
T <- dim(array)[3]

# Define time points
time_points <- seq(1, T)




for (i in sample(N,10)) {
  # Compute the average probability for each time point across individuals
  avg_prob <- colMeans(array[i,, ])

  # Fit GP and predict
  gppred <- fit_gp_and_predict(avg_prob, time_points, time_points)

  # Store the actual and predicted values in the plot_data
  plot_data <- rbind(plot_data, data.frame(
    ind = rownames(array)[i],
    time_slice = time_points,
    prob_actual = avg_prob,
    prob_pred = gppred
  ))
}

# Convert disease to factor for plotting


# Gather the actual and predicted values for ggplot
plot_data_long <- plot_data %>%
  pivot_longer(cols = c(prob_actual, prob_pred), names_to = "type", values_to = "prob")

# Plot using ggplot2
ggplot(plot_data_long, aes(x = time_slice, y = prob, color = type)) +
  geom_point() +
  geom_smooth() +
  facet_wrap(~ ind, scales = "free_y") +
  labs(title = "Actual vs Predicted Average Probabilities Across Diseases",
       x = "Time Slice",
       y = "Probability") +
  theme_minimal() +
  theme(legend.position = "top")


# Assuming array is your 3D array with dimensions (individuals x diseases x time)
individuals <- dim(array)[1]
diseases <- dim(array)[2]
time_points <- dim(array)[3]

ind_means=apply(array,1,mean)
dise_means=apply(array,2,mean)
# Convert 3D array to a 2D matrix (individuals x (diseases * time_points))
matrix_data <- matrix(array, nrow = individuals, ncol = diseases * time_points)

# Step 1: Subtract Row Means
row_means <- rowMeans(matrix_data)
centered_matrix <- sweep(matrix_data, 1, row_means)

# Step 2: Subtract Column Means
column_means <- colMeans(centered_matrix)
centered_matrix <- sweep(centered_matrix, 2, column_means)



# Step 3: Perform SVD
svd_result <- svd(centered_matrix)

# Extracting SVD components
U <- svd_result$u
D <- svd_result$d
V <- svd_result$v

# Printing the results
print("U matrix (left singular vectors):")
print(U)
print("Singular values:")
print(D)
print("V matrix (right singular vectors):")
print(V)


###

array=readRDS("~/Dropbox (Personal)/UKB_topic_app/disease_array.rds")

first_array=readRDS("~/Desktop/disease_array_incidence.rds")

dm=apply(array,c(2,3),mean) ## each entry is the average number of patients for a given diagnoses at time t
pm=apply(array,c(1,3),mean) ## each entry is the average number of diagnoses for a given patient at time t


fdm=apply(first_array,c(2,3),mean) ## each entry is the average number of patients for a given diagnoses at time t
fpm=apply(first_array,c(1,3),mean) ## each entry is the average number of diagnoses for a given patient at time t


matplot(t(dm[sample(nrow(dm),10),]),ylab="Mean Number of Patients Diagnosed with a condition over time")
matplot(t(pm[sample(nrow(pm),10),]),ylab="Mean Number of Cumulative Diagnoses for patient over time")
library(ggplot2)
ggplot(melt(fdm[sample(nrow(dm),10),]),aes(x=Var2,y=value,group=Var1,color=as.factor(Var1)))+
  geom_smooth()+facet_wrap(~Var1)+labs(y="Disease Incidence")+theme_classic()

ggplot(melt(dm[sample(nrow(dm),10),]),aes(x=Var2,y=value,group=Var1,color=as.factor(Var1)))+
  geom_smooth()+facet_wrap(~Var1)+labs(y="Disease Prevalence")+theme_classic()



matplot(t(fdm[sample(nrow(dm),10),]),ylab="Mean Number of Patients Diagnosed with a condition over time",type="l")
matplot(t(fpm[sample(nrow(pm),10),]),ylab="Mean Number of Cumulative Diagnoses for patient over time",type="l")

## normalize

c=apply(array,c(2,3),function(x){scale(x,center = TRUE,scale = TRUE)})





dims <- dim(first_array)
norm_array <- array(0, dim=dims, dimnames=dimnames(first_array))
disease_time_mean_array=array(0,dim=c(dims[2],dims[3]),dimnames=list(dimnames(first_array)[[2]],dimnames(first_array)[[3]]))
norm_first_array <- array(0, dim=dims, dimnames=dimnames(first_array))


## here we normalize for total number of diag
for(i in 1:dims[1]) {
  ind_total=sum(first_array[i,,])
  norm_first_array[i, , ]=first_array[i, , ]/ind_total
}

a=apply(norm_first_array,1,sum)

dims <- dim(first_array)

### now we have normlazied for total number of diagnoses

disease_time_mean_array=array(0,dim=c(dims[2],dims[3]),dimnames=list(dimnames(first_array)[[2]],dimnames(first_array)[[3]]))
norm_for_svd_array=array(0,dim=c(dims[1],dims[2],dims[3]),dimnames=list(dimnames(first_array)[[1]],dimnames(first_array)[[2]],dimnames(first_array)[[3]]))

disease_time_mean_array_unnorm=array(0,dim=c(dims[2],dims[3]),dimnames=list(dimnames(first_array)[[2]],dimnames(first_array)[[3]]))
norm_for_svd_array_unn=array(0,dim=c(dims[1],dims[2],dims[3]),dimnames=list(dimnames(first_array)[[1]],dimnames(first_array)[[2]],dimnames(first_array)[[3]]))

# Loop over diseases and time points to calculate means and normalize
for (d in 1:dims[2]) {
  for (t in 1:dims[3]) {
    # Calculate the mean of the current disease at the current time point across all individuals, using the individaul correcte dof his lifetime diagnoses
    disease_time_mean <- mean(norm_first_array[, d, t])
    # Calculate the mean of the current disease at the current time point across all individuals
    disease_time_mean_unnorm <- mean(first_array[, d, t])
    disease_time_mean_array[d,t] <- disease_time_mean
    disease_time_mean_array_unnorm[d,t] <- disease_time_mean_unnorm
    # Subtract the mean from each individual's value for the current disease at the current time point
    norm_for_svd_array[, d, t] <- norm_first_array[, d, t] - disease_time_mean
    norm_for_svd_array_unn[, d, t] <- first_array[, d, t] - disease_time_mean_unnorm
  }
}

all.equal(disease_time_mean_array,fdm)

s=sample(nrow(disease_time_mean_array),10)
matplot(t(disease_time_mean_array_unnorm[s,]),ylab="Mean Number of Patients Diagnosed with a condition over time",type="l")
lines(colMeans(fdm),col="red",lwd=3)


# Get the disease names/descriptions corresponding to the phecodes
phecodes_in_array = dimnames(first_array)[[2]]

# Extract the phecode column and unlist it to a numeric vector
phecodes_in_disease_info = as.numeric(unlist(ATM::disease_info_phecode_icd10[,"phecode"]))

disease_names = ATM::disease_info_phecode_icd10[match(phecodes_in_array, phecodes_in_disease_info),"phenotype" ]$phenotype


eigen_vs=array(0,dim=c(dims[2],dims[2],dims[3]),dimnames=list(disease_names,paste0("EV:",rep(1:dims[2])),dimnames(first_array)[[3]]))
eigen_us=array(0,dim=c(dims[1],dims[2],dims[3]),dimnames=list(dimnames(first_array)[[1]],dimnames(first_array)[[2]],dimnames(first_array)[[3]]))


for(t in 1:dims[3]){


  s=svd(norm_for_svd_array_unn[,,t])




  eigen_vs[,,t]=s$v
  eigen_us[,,t]=s$u
}

m=reshape2::melt(disease_time_mean_array[1:10,])
ggplot(m,aes(x=Var2,y=value,col=as.factor(Var1)))+geom_smooth()
library(reshape2)

# Heatmap for V
v_df <- melt(eigen_vs[,,t])
ggplot(v_df, aes(x = Var2, y = Var1, fill = value)) +
  geom_tile() +
  scale_fill_gradient2() +
  labs(x = "Topic", y = "Disease", title = paste("Disease-Topic Associations at Time", t))

# Heatmap for U
u_df <- melt(eigen_us[,,t])
ggplot(u_df, aes(x = Var2, y = Var1, fill = value)) +
  geom_tile() +
  scale_fill_gradient2() +
  labs(x = "Topic", y = "Individual", title = paste("Individual-Topic Associations at Time", t))

install.packages("NMF")
library(NMF)  # or another NMF package

# Number of factors (latent topics)
n_factors = 10  # Adjust this based on your understanding of the data

# Store the factor matrices for each time point
W_array_spca = array(0, dim = c(dims[1], n_factors, dims[3]),dimnames=list(dimnames(first_array)[[1]],paste0("Factor:",rep(1:n_factors)),dimnames(first_array)[[3]]))
H_array_spca = array(0, dim = c(n_factors, dims[2], dims[3]),dimnames=list(paste0("Factor:",rep(1:n_factors)),disease_names,dimnames(first_array)[[3]]))


# Sparsity parameter (higher values lead to more sparse loadings)
lambda = 0.1   # Adjust this based on your desired level of sparsity

for (t in 1:dims[3]) {
  # Perform SPCA on the unnormalized, centered data for time point t
  spca_result = spca(first_array[,,t], K = n_factors,
                     type = "predictor", lambda = lambda,
                     sparse = "penalty", para = rep(lambda, n_factors))

  H_array_spca[,,t] = t(spca_result$loadings)   # Individual-factor loadings (analogous to lambda)

  h=t(solve(t(W_array_spca[,,t]) %*% W_array_spca[,,t]) %*%
      transpose(W_array_spca[,,t]) %*% norm_for_svd_array_unn[,,t])
  W_array_spca[,,t] = spca_result$scores      # Factor-disease loadings (analogous to phi)
}


### increase the sparsity


## creatin different expanded array:
df=fread('~/Dropbox (Personal)/UKB_topic_app/patient_diagnoses_expanded_data.csv')


unique_ids=sample(unique(df$eid),1000)


# Function to expand the diagnoses for an individual
expand_diagnoses <- function(individual_data) {
  expanded_data <- data.frame()

  for(i in 1:nrow(individual_data)) {
    current_age_bin <- individual_data$age_bin[i]
    diag_code <- individual_data$diag_icd10[i]

    # Create a sequence of age bins from the current to the max (e.g., 65 -> 70 -> 75 -> ...)
    future_age_bins <- seq(current_age_bin, 95, by = 5)  # Adjust 95 to your maximum age_bin if different

    # Create rows for each future age bin
    new_rows <- data.frame(
      eid = individual_data$eid[i],
      diag_icd10 = diag_code,
      age_diag = individual_data$age_diag[i],
      age_bin = future_age_bins
    )

    expanded_data <- rbind(expanded_data, new_rows)
  }

  return(expanded_data)
}



# Apply the function to each individual
#expanded_df <- df[df$eid%in%unique_ids,] %>%
expanded_df <- df %>%
  group_by(eid) %>%
  do(expand_diagnoses(.))

write.csv(expanded_df,"~/Dropbox (Personal)/UKB_topic_app/patient_diagnoses_expanded_data_cumulative.csv", row.names = FALSE)


for(i in 1:length(unique(mf$disease))){
  d=mf[mf$disease==unique(mf$disease)[i],]
  for(t in 1:length(unique(mf$time_slice))){
    y[i,,t]=rbinom(10000,1,prob = max(0,d$prob_pred[t]))
  }
}


