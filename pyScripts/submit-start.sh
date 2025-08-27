# pyScripts/submit-start.sh  (fixed)
#!/bin/bash
set -euo pipefail

JOB_QUEUE="launcher-test-spot-MM-Batch-JobQueue"
JOB_DEF="aladyn-model-job:15"
START=0
END=10000

for AGE in $(seq 71 79); do
  NAME="aladyn-${START}-${END}-age-${AGE}"
  aws batch submit-job \
    --job-name "$NAME" \
    --job-queue "$JOB_QUEUE" \
    --job-definition "$JOB_DEF" \
    --parameters start_index="${START}",end_index="${END}",age="${AGE}" \
    --tags slice="0-10k",target_age="${AGE}"
done
