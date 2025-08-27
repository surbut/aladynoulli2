#!/bin/bash

JOB_QUEUE="launcher-test-spot-MM-Batch-JobQueue"
JOB_DEF="aladyn-model-job:8"  # or whatever revision is latest

for ((i=0; i<40000; i+=10000)); do
  START=$i
  END=$((i + 10000))
  NAME="aladyn-${START}-${END}"
  
  aws batch submit-job \
    --job-name "$NAME" \
    --job-queue "$JOB_QUEUE" \
    --job-definition "$JOB_DEF" \
    --parameters start_index="$START",end_index="$END"
done
