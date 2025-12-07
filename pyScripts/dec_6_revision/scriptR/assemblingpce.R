## assemble_all_scores
## 
qrisk=read.table("~/Downloads/ukb_qrisk3.txt",header=T)
pce=read.csv("~/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv")
gail=read.csv("~/Library/CloudStorage/Dropbox-Personal/gail_dat_ordered.csv")

library(dplyr)

p2 <- pce[,c("eid","age","Sex","SexNumeric","pce","pce_goff","pce_goff_fuull","prevent_base_ascvd_risk")] %>%
  left_join(qrisk[,c("eid","score")], by="eid")


names(p2)=c("eid","age","Sex'","SexNumeric","pce_jemma","pce_goff_nas","pce_goff_imputed",
            "prevent_base_ascvd_risk","qrisk3")

p2$prevent_impute=p2$prevent_base_ascvd_risk
p2$prevent_impute[is.na(p2$prevent_base_ascvd_risk)]=mean(na.omit(p2$prevent_base_ascvd_risk))

p2=merge(p2,gail[,c("identifier","sex","T1","Gail_absRisk")],all.x = TRUE,by.x ="eid",by.y="identifier",
         sort=FALSE)

all.equal(p2$eid,pce$eid)

ifelse(is.na(p2$qrisk3),mean(na.omit(p2$qrisk3)),p2$qrisk3)->p2$qrisk3
write.csv(p2,"~/Library/CloudStorage/Dropbox-Personal/ukb_pce_prevent_gail_qrisk3_combined.csv",row.names=FALSE)