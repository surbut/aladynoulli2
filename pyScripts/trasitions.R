## transitions
## 
## 

all_patient_diseases = data.frame(read.csv("~/aladynoulli2/pyScripts/all_patient_diseases.csv", fill = T))
sig_refs = read.csv("~/aladynoulli2/pyScripts/reference_thetas.csv", header = T)
E=readRDS("E_full_tensor.rds")
colnames(E)=all_patient_diseases$X0


## identify patients who had myocardial infarction before age 51
sick=E[E[,"Myocardial infarction"]<51,]

## column by column, idenfity number of people with diseases that occurred before myocardial infarction
pre_disease=apply(sick,2,function(x){sum(x<sick[,"Myocardial infarction"])})
pre_disease_enrichment=pre_disease/colSums(sick<51)
