rm -f list_beds.txt
for chr in {2..22}; do echo "/broad/ukbb/genotype/ukb_cal_chr${chr}_v2.bed /broad/ukbb/genotype/ukb_snp_chr${chr}_v2.bim /medpop/esp2/pradeep/UKBiobank/v2data/fam/ukb708_cal_chr20_v2_s488374.fam" >> list_beds.txt; done

/medpop/esp2/btruong/Tools/plink \
  --bed /broad/ukbb/genotype/ukb_cal_chr1_v2.bed \
  --bim /broad/ukbb/genotype/ukb_snp_chr1_v2.bim \
  --fam /medpop/esp2/pradeep/UKBiobank/v2data/fam/ukb708_cal_chr1_v2_s488374.fam \
  --merge-list list_beds.txt \
  --make-bed --out ukb_cal_allChrs
