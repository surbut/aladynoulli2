
#####################################################
# load packages
#####################################################

#install.packages("devtools")
#It may be necessary to install required as not all package dependencies are installed by devtools:
#install.packages(c("dplyr","tidyr","ggplot2","MASS","meta","ggrepel","DT"))
#devtools::install_github("PheWAS/PheWAS")
library(PheWAS)
library(data.table)
library(stringr)
library(stringi)
library(lubridate)
library(ATM)
library(dplyr)
library(tidyr)


#####################################################
# load UK Biobank date of death and loss-to-follow-up
# data to define follow-up time
message("###load UK Biobank censor and HES data up to Spring 2020###")
#####################################################


dfukb_baseline_pheno=data.frame(readRDS("~/Library/CloudStorage/Dropbox-Personal/pheno_dir/output/dfukb_baseline_pheno.rds"))

england<-c("10003","11001","11002","11007","11008","11009","11010","11011","11012","11013","11014","11016","11017","11018","11019","11020","11021")

scotland<-c("11004","11005")

wales<-c("11003","11022", "11006","11023")
# from https://biobank.ndph.ox.ac.uk/ukb/exinfo.cgi?src=Data_providers_and_dates

dfukb_baseline_pheno[dfukb_baseline_pheno$f.54.0.0 %in% england,"censordateHES"]<-as.Date("2022-10-31")

dfukb_baseline_pheno[dfukb_baseline_pheno$f.54.0.0 %in% scotland,"censordateHES"]<-as.Date("2022-08-31")

dfukb_baseline_pheno[dfukb_baseline_pheno$f.54.0.0 %in% wales,"censordateHES"]<-as.Date("2022-05-31")

dfukb_baseline_pheno$death_censor_date=as.Date("2022-11-30")


bd.censor=dfukb_baseline_pheno %>%
  dplyr::rename(f.eid = identifier) %>%
  dplyr::select(f.eid, f.53.0.0, censordateHES, death_censor_date,reference_date)
colnames(bd.censor)=c("f.eid","enroll_date","phenotype_censor_date","death_censor_date","birthdate")
bd.censor$phenotype_censor_date <- ymd(bd.censor$phenotype_censor_date)
bd.censor$enroll_date <- ymd(bd.censor$enroll_date)


bd_diag <- fread("~/Documents/hesin_May2023//hesin_diag.txt.gz")
bd_diag <- bd_diag %>%
  rename(f.eid = eid)
bd_diag[bd_diag == ""] <- NA



bd_hes <- fread("~/Documents/hesin_May2023/hesin.txt.gz")
bd_hes[bd_hes == ""] <- NA
bd_hes <- bd_hes %>%
  dplyr::select(eid, ins_index, epistart, admidate) %>%
  mutate(event_start = case_when(!is.na(epistart) ~ epistart,
                                 is.na(epistart) ~ admidate)) %>%
  dplyr::select(-epistart, -admidate) %>%
  rename(f.eid = eid)
bd_hes$event_start <- dmy(bd_hes$event_start)
bd_hes <- bd_hes %>%
  filter(!is.na(event_start)) %>%
  filter(event_start <= as.Date("2022-10-31")) ## use dates here https://biobank.ndph.ox.ac.uk/ukb/exinfo.cgi?src=Data_providers_and_dates
bd_hes <- bd_hes %>%
  left_join(bd_diag, by = c("f.eid", "ins_index")) %>%
  mutate(vocabulary_id = case_when(!is.na(diag_icd10) ~ "ICD10CM",
                                   is.na(diag_icd10) &
                                     !is.na(diag_icd10_nb) ~ "ICD10CM",
                                   is.na(diag_icd10) & is.na(diag_icd10_nb) &
                                     !is.na(diag_icd9) ~ "ICD9CM")) %>%
  mutate(code = case_when(!is.na(diag_icd10) ~ as.character(diag_icd10),
                          is.na(diag_icd10) &
                            !is.na(diag_icd10_nb) ~ as.character(diag_icd10_nb),
                          is.na(diag_icd10) & is.na(diag_icd10_nb) &
                            !is.na(diag_icd9) ~
                            as.character(diag_icd9)))

# subset and annotate ICD10 data
icd10 <- bd_hes %>%
  filter(vocabulary_id == "ICD10CM") %>%
  dplyr::select(f.eid, event_start, ins_index, arr_index, level,
                vocabulary_id, code) %>%
  filter(!str_detect(code, "^S")) %>% # remove XIX Injury, poisoning and certain other consequences of external causes
  filter(!str_detect(code, "^T")) %>% # remove XIX Injury, poisoning and certain other consequences of external causes
  filter(!str_detect(code, "^V")) %>% # remove Chapter XX External causes of morbidity and mortality
  filter(!str_detect(code, "^W")) %>% # remove Chapter XX External causes of morbidity and mortality
  filter(!str_detect(code, "^X")) %>% # remove Chapter XX External causes of morbidity and mortality
  filter(!str_detect(code, "^Y")) %>% # remove Chapter XX External causes of morbidity and mortality
  filter(!str_detect(code, "^Z")) %>% # remove Chapter XXI Factors influencing health status and contact with health services
  filter(!str_detect(code, "^U")) %>% # remove Chapter XXII Codes for special purposes
  mutate(code_char = nchar(code))


#
## to use with atm
###
###
library(ATM)
## we don't want the period
d=dplyr::right_join(bd.censor[,c("f.eid","birthdate")],icd10,by="f.eid")
d$age=round(as.numeric(difftime(d$event_start,d$birthdate,units="days")/365.25),1)
hes_for_atm=d[,c("f.eid","code","age")]
names(hes_for_atm)=names(HES_icd10_example)
i=icd2phecode(hes_for_atm)
#dim(i)
## 3989749       3

#length(unique(i$diag_icd10))
# there are 1225 unique icd10 codes
#saveRDS(i,file="~/Library/CloudStorage/Dropbox-Personal/icd10phe_lab.rds")


i=readRDS("~/Library/CloudStorage/Dropbox-Personal/icd10phe_lab.rds")

a=i%>%left_join(disease_info_phecode_icd10, by = c("diag_icd10"="phecode" ))
##make sure these all occurred 

setdiff(a$ICD10,union(bd_diag[bd_diag$f.eid%in%5760433,"diag_icd10_nb"],bd_diag[bd_diag$f.eid%in%5760433,"diag_icd10"]))

df_good=a[,c("eid","age_diag","phenotype")]

df_grouped <- df_good %>%
  group_by(eid, phenotype) %>%
  summarize(first_age_diag = min(age_diag, na.rm = TRUE))

df_wide <- df_grouped %>%
  pivot_wider(names_from = phenotype, values_from = first_age_diag)
dim(df_wide)


col=df_wide[,(colSums(!is.na(df_wide))>1000)]
row=col[(rowSums(!is.na(col))>2),]
colrow=row[,(colSums(!is.na(row))>1000)]

saveRDS(colrow,"~/Library/CloudStorage/Dropbox-Personal/phecode/icd_subset.rds")

dfw=colrow[,-c(1)]
dfw[!is.na(dfw)]=1
dfw[is.na(dfw)]=0
X=as.matrix(dfw)

o=order(rowSums(X),decreasing = T)
o=order(colSums(X),decreasing = T)


## test spline vs AR model

l=apply(colrow[head(o),],1,function(x){x[which(!is.na(x))]})
length(l[[1]])
unique(bd_diag[bd_diag$f.eid%in%l[[1]]["eid"]][,"diag_icd10"])


## you can see that using the disease_info_phecode_icd10 map is equivalnet to merging the phecode_icd10 and phecode_icd10cm  mappings
f <- fread("~/Library/CloudStorage/Dropbox-Personal/medpop:phewas.catalog/phewas.catalog/phecode_icd10.csv") 
g <- fread("~/Library/CloudStorage/Dropbox-Personal/medpop:phewas.catalog/phewas.catalog/Phecode_map_v1_2_icd10cm_beta.csv") 
#show that these files are essentially the same as the disease_info_phecode
#length(union(f$Phenotype,g$phecode_str)) = 1760
##setdiff(unique(disease_info_phecode_icd10$phenotype),(union(f$Phenotype,g$phecode_str)))

## but what are the ICD10 codes in disease_info_phecode_icd10 if there are many icd10 codes that map to a phecode

# character(0)
#length(setdiff((union(f$Phenotype,g$phecode_str)),unique(disease_info_phecode_icd10$phenotype)))

dim(unique(icd10[icd10$f.eid%in%5760433,"code"][,1]))

### phecode_icd10cm maps between ICD-10-CM to Phecode;
### phecode_icd10 maps ICD-10 to Phecode; 
### disease_info_phecode_icd10 saves the disease names of 1755 Phecodes
dim(merge(disease_info_phecode_icd10,unique(icd10[icd10$f.eid%in%5760433,"code"]),by.x="ICD10",by.y="code"))

a1=merge(unique(icd10[icd10$f.eid%in%5760433,"code"]),phecode_icd10,by.x="code",by.y="ICD10")
a2=merge(unique(icd10[icd10$f.eid%in%5760433,"code"]),phecode_icd10cm,by.x="code",by.y="ICD10")
length(union(a1$PheCode,a2$phecode))

sam=icd10[icd10$f.eid%in%5760433,]
sam2=merge(sam,bd.censor[,c("f.eid","birthdate")],by.x="f.eid",by.y="f.eid",all.x=TRUE)
sam2$age=as.numeric(difftime(sam2$event_start,sam2$birthdate,units="days")/365.25)

rec_data=sam2[,c("f.eid","age","code")]
names(rec_data)=c("eid","age_diag","diag_icd10")

###
rec_data=readRDS("~/Library/CloudStorage/Dropbox-Personal/rec_data.rds")
new_data <- rec_data %>% select(eid, diag_icd10, age_diag) %>% filter(stringr::str_detect(diag_icd10, "^[A-N]")) %>% 
left_join(phecode_icd10cm, by = c(diag_icd10 = "ICD10")) %>% 
mutate(diag_icd10 = substring(diag_icd10, 1, 4)) %>% 
left_join(phecode_icd10, by = c(diag_icd10 = "ICD10")) %>% 
left_join(short_icd10, by = c(diag_icd10 = "ICD10"))


new_data <- new_data %>% mutate(phecode = if_else(is.na(phecode), 
parent_phecode, phecode)) %>% mutate(PheCode = if_else(is.na(PheCode),phecode, PheCode)) %>% filter(!is.na(PheCode)) %>% select(eid,PheCode, age_diag) %>% 
  rename(diag_icd10 = PheCode) %>% 
  group_by(eid, diag_icd10) %>% filter(n() == 1 | age_diag == 
                                         min(age_diag)) %>% slice(1) %>% dplyr::ungroup()


### to go backward: what i don't understand is if i wanted to see which icd10 codes generated the phecode (even if it isn't 1:1 mapping), 
##i'd like to take a sample individual with many (122 diagnoses) 5760433
## i can look at him/her in 
new_data
### I could then merge with phecode definitions 
a=new_data%>%left_join(disease_info_phecode_icd10, by = c("diag_icd10"="phecode" ))
## where here diag_icd10 = phecode means that the diag_icd10 is actually phecode name (not diagnostic icd10 code)
## however, not all of the ICD10 codes are found in his initial icd10 feedin

x=as.vector(rec_data[rec_data$eid%in%5760433,"diag_icd10"])
y=as.vector(data.frame(a$ICD10))
setdiff(unique(x[[1]]),y$a.ICD10) ## but this is ok because we expect some subsetting
## however, the smaller file (ie the result of merging with phecodes) should have all captured ... 
setdiff(y$a.ICD10,x[[1]])
length(setdiff(y$a.ICD10,x[[1]]))


## there are 82 ICD10 codes created by fold mapping that were not present in original data, but that can't be right because this should have created subsets ... 

## However, remember the 1:e matching issue: so let's try grabbing the icd10 codes that match with the raw phe-icd10 mapping, and then seeing how many in the 
## final set of phecodes map with those. 
## unique icd10 and icd10m codes from raw (icd10) file and file from phewas website codes  
n=left_join(unique(icd10[icd10$f.eid%in%l[[1]]["eid"]][,"code"]),phecode_icd10,by=c(code="ICD10"))
m=left_join(unique(icd10[icd10$f.eid%in%l[[1]]["eid"]][,"code"]),phecode_icd10cm,by=c(code="ICD10"))

#mapping of unique phecodes that correspond with those icd10 codes grabbed from original file
u=union(m$phecode,n$PheCode)

#looking at intersection of final phecodes (a$diag_icd10) with union of original phecodes
setdiff(a$diag_icd10,u)
setdiff(u,a$diag_icd10)

## threa are just two different between final and 

unique(merge(unique(bd_diag[bd_diag$f.eid%in%l[[1]]["eid"]][,"diag_icd10"]),
             disease_info_phecode_icd10,by.x="diag_icd10",by.y="ICD10")$phenotype)

i=intersect(names(l[[1]]),unique(merge(unique(bd_diag[bd_diag$f.eid%in%l[[1]]["eid"]][,"diag_icd10"]),
disease_info_phecode_icd10,by.x="diag_icd10",by.y="ICD10")$phenotype))

setdiff(names(l[[1]]),unique(merge(unique(bd_diag[bd_diag$f.eid%in%l[[1]]["eid"]][,"diag_icd10"]),
                                             disease_info_phecode_icd10,by.x="diag_icd10",by.y="ICD10")$phenotype))
