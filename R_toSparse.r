#install.packages("devtools")
#It may be necessary to install required as not all package dependencies are installed by devtools:
#install.packages(c("dplyr","tidyr","ggplot2","MASS","meta","ggrepel","DT"))
#devtools::install_github("PheWAS/PheWAS")
library(PheWAS)
library(data.table)
library(stringr)
library(stringi)
library(lubridate)
library(AgeTopicModels)
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

## we don't want the period
d=dplyr::right_join(bd.censor[,c("f.eid","birthdate")],icd10,by="f.eid")
d$age=round(as.numeric(difftime(d$event_start,d$birthdate,units="days")/365.25),1)
hes_for_atm=d[,c("f.eid","code","age")]
names(hes_for_atm)=names(HES_icd10_example)
i=icd2phecode(hes_for_atm)


icdlab=readRDS("~/Dropbox-Personal/icd10phe_lab.rds")
icdlab$age_diag=round(icdlab$age_diag)
dim(icdlab)
icdlab=icdlab[icdlab$diag_icd10%in%AgeTopicModels::UKB_349_disease$diag_icd10,]
length(unique(icdlab$diag_icd10))
dim(icdlab)
head(icdlab)
prs=readRDS("~/Library/CloudStorage//Dropbox-Personal//pheno_dir/prs_subset.rds")
# Intersect ICD data with PRS first




person_ids=intersect(prs$Identifier,icdlab$eid)
# Store the actual IDs and ICD codes in order



prs=prs[prs$Identifier%in%person_ids,]
all.equal(as.character(rownames(prs)),as.character(person_ids))
saveRDS(prs,"~/Dropbox (Personal)/prs_subset_forsparse.rds")
G=prs[,-37]
saveRDS(G,"~/Dropbox (Personal)/G_subset_forsparse.rds")


# Create mapping dictionaries with names preserved
id_map <- setNames(seq_len(length(person_ids)), person_ids)
disease_map <- setNames(seq_len(length(disease_codes)), disease_codes)
icdlab=icdlab[icdlab$eid%in%person_ids,]


# Create sparse matrix with same logic as before
sparse_data <- icdlab %>%
  mutate(
    id_idx = id_map[as.character(eid)],
    disease_idx = disease_map[as.character(diag_icd10)],
    age_idx = age_diag - 29
  ) %>%
  filter(age_idx >= 1, age_idx <= 52)


# Create sparse array using sparseMatrix, now with dimnames
sparse_arrays <- vector("list", 52)
for(t in 1:52) {
  temp_data <- sparse_data %>% filter(age_idx == t)
  sparse_arrays[[t]] <- sparseMatrix(
    i = temp_data$id_idx,
    j = temp_data$disease_idx,
    x = 1,
    dims = c(length(id_map), length(disease_map)),
    dimnames = list(person_ids, disease_codes)
  )
}
all.equal(rownames(sparse_arrays[[1]]),rownames(prs))
saveRDS(sparse_arrays,"~/Dropbox (Personal)/sparse_array.rds")


disease_list=ATM::disease_info_phecode_icd10[ATM::disease_info_phecode_icd10$phecode%in%disease_codes,"phenotype"]
saveRDS(disease_list,"~/Dropbox (Personal)/disease_list_forsparse.rds")
[1] TRUE
write.csv(atm_table_Ukb,"~/aladynoulli2/ukb_disease_names_phecode.csv",row.names = FALSE,quote=FALSE))
