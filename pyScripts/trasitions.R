## transitions
## 
## 

all_patient_diseases = data.frame(read.csv("~/aladynoulli2/pyScripts/all_patient_diseases.csv", fill = T))
sig_refs = read.csv("~/aladynoulli2/pyScripts/reference_thetas.csv", header = T)
E=readRDS("E_full_tensor.rds")
colnames(E)=all_patient_diseases$X0


## identify patients who had myocardial infarction before age 51
sick=E[E[,"Myocardial infarction"]<51,]

## column by column, idenfity number of people with MI who had anohter disease that occurred before myocardial infarction
pre_disease=apply(sick,2,function(x){sum(x<sick[,"Myocardial infarction"])})
pop_norm_pre_disease=pre_disease/colSums(E)
pre_disease_enrichment=pre_disease/colSums(sick<51)
pre_disease_df=data.frame(disease=names(pre_disease),count=pre_disease,
                              pop_norm=pop_norm_pre_disease,
                              enrichment=pre_disease_enrichment)

## for age matched population, count number of people with a given disease not conditional on having MI
## 
## age_matched_matrix=matrix(NA,nrow=nrow(sick),ncol=ncol(sick))
## for(i in 1:nrow(sick)){  
## sick_age=sick[i,"Myocardial infarction"]
## 