d=readRDS("~/Dropbox (Personal)/desktopmess/disease_array_incidence.rds") ## 10000

prs=readRDS("~/Dropbox (Personal)/pheno_dir/prs_subset.rds")
prs$Identifier <- as.character(prs$Identifier)

# Filter the disease array to include only individuals in prs
d <- d[rownames(d) %in% prs$Identifier, , ]

# Ensure prs is in the same order as d
prs <- prs[match(rownames(d), prs$Identifier), ]
rownames(prs)=as.character(prs$Identifier)
#
all.equal(rownames(d),as.character(prs$Identifier))
prs=prs[,c(1:36)]


bpmed=readRDS("~/Dropbox (Personal)/dfukb_chol_bp_smoke.rds")
baseline=readRDS("~/Dropbox (Personal)/pheno_dir/output/dfukb_baseline_pheno.rds")
bpmed=merge(bpmed,baseline,by="identifier")
# Ensure prs is in the same order as d
bpmed <- bpmed[match(rownames(d), bpmed$identifier), ]
all.equal(rownames(d),as.character(bpmed$identifier))

saveRDS(b,file="~/Dropbox (Personal)/metadata10k.rds")
ages=30:80
T=dim(d)[3]
event_indices = array(0, dim = c(dim(d)[1], dim(d)[2]))
for (i in 1:dim(d)[1]) {
  for (disease in 1:dim(d)[2]) {
    if (sum(d[i, disease, ]) != 0) {
      # Find the index when the event occurred
      event_indices[i, disease] = which(d[i, disease, ] == 1)[1] - 1  # Subtract 1 for 0-based indexing
    }
    else {
      # For censored events, use the last index
      event_indices[i, disease] = dim(d)[3]  # keep at T
      #event_indices[i, disease] = dim(d)[3]-1  # 0 based indexing
    }
  }
}

saveRDS(event_indices,"~/tensornoulli_ehr/data/event.rds")
#saveRDS(event_indices,"~/tensornoulli_ehr/data/event_for_aladynoulli.rds")
saveRDS(d,"~/tensornoulli_ehr/data/Y.rds")
saveRDS(prs,"~/tensornoulli_ehr/data/prs.rds")

