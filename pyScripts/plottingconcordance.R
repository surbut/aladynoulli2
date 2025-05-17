library(data.table)
setwd("~/aladynoulli2/pyScripts/")


cox_aladyn=fread("model_comparison_results_cox_aladyn.csv")
#static_auc=fread("model_comparison_results_bootstatic_auc.csv")
static_c=fread("model_comparison_results_cindex.csv")
dynamic=fread("model_comparison_results_dynami.csv")

m=merge(cox_aladyn,static_auc[,c("Disease","Aladynoulli_AUC")],by="Disease")
m=merge(m,static_c[,c("Disease","Cox_Concordance")],by="Disease")
m=merge(m,dynamic[,c("Disease","Aladynoulli_AUC")],by="Disease")

names(m)[5]="Aladynoulli_static"
names(m)[7]="Aladynoulli_dynamic"

