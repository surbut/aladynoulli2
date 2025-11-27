#!/bin/bash
# Upload first 10 batches (0-100000) model files to S3 for comparison

cd ~/aladyn_project/output/retrospective_pooled

echo "Creating tar archive of model files (batches 0-9)..."
tar -czf aws_first_10_batches_models.tar.gz \
    model_enroll_fixedphi_sex_0_10000.pt \
    model_enroll_fixedphi_sex_10000_20000.pt \
    model_enroll_fixedphi_sex_20000_30000.pt \
    model_enroll_fixedphi_sex_30000_40000.pt \
    model_enroll_fixedphi_sex_40000_50000.pt \
    model_enroll_fixedphi_sex_50000_60000.pt \
    model_enroll_fixedphi_sex_60000_70000.pt \
    model_enroll_fixedphi_sex_70000_80000.pt \
    model_enroll_fixedphi_sex_80000_90000.pt \
    model_enroll_fixedphi_sex_90000_100000.pt

echo "Checking tar file size..."
ls -lh aws_first_10_batches_models.tar.gz

echo "Uploading to S3..."
aws s3 cp aws_first_10_batches_models.tar.gz s3://sarah-research-aladynoulli/data_for_running/aws_first_10_batches_models.tar.gz

echo "Done! File uploaded to: s3://sarah-research-aladynoulli/data_for_running/aws_first_10_batches_models.tar.gz"




