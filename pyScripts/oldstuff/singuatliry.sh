#!/bin/bash
#$ -cwd
#$ -j y
#$ -o regenie_step1_$TASK_ID_resub.log
#$ -pe smp 16
#$ -l h_vmem=24G
#$ -l h_rt=72:00:00
#$ -N regenie_$TASK_ID

# Load REGENIE
source /broad/software/scripts/useuse
WALLTIME=72:00:00
OUT_DIR="./results"
mkdir -p ${OUT_DIR}

sif=/medpop/esp2/projects/software/singularity/regenie/v3.4.2/regenie_3.4.2.sif

chr=$SGE_TASK_ID

singularity exec --bind /medpop/:/medpop/,/broad/hptmp/:/broad/hptmp,/broad/ukbb/genotype:/broad/ukbb/genotype,/medpop/esp2/SarahUrbut/:/medpop/esp2/SarahUrbut/,/broad/ukbb/imputed_v3:/broad/ukbb/imputed_v3,/medpop/esp2/pradeep/UKBiobank/v3data/:/medpop/esp2/pradeep/UKBiobank/v3data/ $sif \
regenie \
  --step 1 \
  --bed /medpop/esp2/SarahUrbut/regenie_GWAS_aladynoulli/ukb_links/ukb_chr${chr}_v2 \
  --extract good_snps_chr${chr}.snplist \
  --covarFile ukbb_covariates_400k.txt \
  --phenoFile signature_auc_phenotypes.txt \
  --bsize 10000 \
  --threads 16 \
  --out ${OUT_DIR}/ukb_step1_chr${chr} \
  --lowmem \
  --lowmem-prefix tmp_rg_chr${chr} \
  --verbose \
  --gz \
  --force-step1



#########

qsub ... -t 1-22 <>


