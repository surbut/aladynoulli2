# Batch Training Pipeline: EFS → Docker → AWS Batch

This document describes the end‑to‑end workflow for running the survival model training at scale using Amazon Elastic File System (EFS), a Docker image stored in Amazon ECR, and AWS Batch. It also clarifies what must be updated when you publish new code versions or want to change parameters.

---

## 1) Data on EFS (Inputs & Outputs)

**Purpose**  
EFS provides a shared, POSIX‑compatible filesystem that all Batch jobs can mount. It allows large input tensors and reference artifacts to be read by multiple jobs, and it centralizes outputs.

**Mount Layout (inside the container)**  
- Input data is read from a path under `/data/input`.  
- Outputs are written to `/data/output`.  
- Each job writes to an age‑specific subfolder (for example, `age_40`, `age_41`, …) to avoid file overwrites. Filenames also embed the age for clarity.

**Best Practices**  
- Keep all model artifacts (tensors, essentials, references, checkpoints) under the input directory, mirroring the paths referenced by the training script.  
- Ensure your EFS throughput mode can handle concurrent reads/writes. Consider Provisioned Throughput or Max I/O performance mode if many jobs run in parallel.  
- Use tags or directory naming conventions for each batch run (e.g., include date, slice, and age range) so results stay organized and discoverable.  
- Monitor EFS metrics (throughput, IOPS, burst credits) when scaling up concurrency.

---

## 2) Docker Image (Build & Push)

**Purpose**  
The container bundles your Python environment and training code so jobs run consistently across instances and instance families.

**Key Points**  
- The working directory inside the image is `/app`.  
- The training scripts reside under `/app/pyScripts`.  
- `PYTHONPATH` includes both `/app` and `/app/pyScripts` so custom modules can be imported directly.  
- No fixed entrypoint or command is set in the image; the AWS Batch job definition provides the runtime command and parameters.  
- After building locally or in CI, push the image to your Amazon ECR repository. The job definition uses the ECR URI tagged as `latest` (or another tag you choose).

**Operational Tips**  
- Pin dependency versions in `requirements.txt` to keep behaviors stable across rebuilds.  
- Rebuild the image whenever your code or dependencies change, and push it under a new tag if you want a safe rollback option.  
- Keep images small (slim base images, no unused tools) for faster provisioning.

---

## 3) AWS Batch Job Definition (Registration)

**Purpose**  
The job definition specifies the container image, vCPUs, memory, EFS mount, and the command the container runs at job start.

**What It Contains**  
- The ECR image URI for the container to run.  
- vCPU and memory sizing for scheduling and bin packing; adjust these based on actual CPU and RAM usage to improve concurrency.  
- A mount for EFS at `/data`.  
- The container command that invokes the training script and passes parameters, including the selected age.  
- Default parameters for `start_index`, `end_index`, and a default `age`, which can be overridden per submission.  
- Retry strategy settings that determine how failures are retried.

**Capacity & Scheduling Considerations**  
- Ensure your Compute Environment instance types and quotas support the chosen vCPU size so multiple jobs can run in parallel.  
- Prefer capacity‑optimized strategies for Spot.  
- Include multiple instance sizes and all Availability Zones to improve placement success.  
- Right‑size vCPUs so instances can pack multiple jobs without stranding capacity.

---

## 4) Job Submission (Ages & Slices)

**Purpose**  
Submissions create individual AWS Batch jobs for each target age so outputs don’t overwrite and jobs run in parallel as capacity allows.

**What Happens at Submission Time**  
- Each job gets a unique name that includes the index slice and age.  
- Parameters such as `start_index`, `end_index`, and `age` are passed to the job definition and flow into the training script.  
- The training script resolves the `age` to an internal offset and writes outputs into an age‑specific subfolder under `/data/output` with age‑stamped filenames.

**Operational Tips**  
- Submit ages in a loop for the full range needed (for example, 40–79).  
- Use tags for tracing (e.g., age, slice, run‑id).  
- Monitor queue and job states in the AWS console or via CloudWatch dashboards.  
- If throughput is limited by storage, consider staging inputs to instance local storage during the job and syncing results back to EFS afterward.

---

## Updating Versions & Changing Parameters

**New Code Version**  
- Rebuild the image with your changes and push to ECR.  
- If you want zero risk of pulling an older cached tag, publish under a new immutable tag and update the job definition to point at that tag.  
- If the job command or parameters changed (for example, new flags), update the job definition’s command and default parameters accordingly.  
- Register a new job definition revision after updates; reference the new revision when submitting jobs.

**New Dependencies**  
- Update `requirements.txt` and rebuild the image.  
- Validate that serialized artifacts (e.g., PyTorch checkpoints) remain compatible with your chosen library versions.  
- If a framework changed default behaviors (e.g., PyTorch `torch.load` safety defaults), ensure your code handles those explicitly.

**New Input Data or Artifacts**  
- Place new tensors and reference files in the EFS input directory, preserving the expected filenames and layout used by the training script.  
- Version your EFS directories by date or run ID to keep lineage and enable rollbacks.

**Changing Runtime Parameters**  
- To change the age range: alter the submission loop to use the desired ages and ensure the script’s argument validation allows the range.  
- To change the slice (start or end index): adjust the parameters at submission time; outputs will embed indices and age so files remain distinct.  
- To change CPU/Memory: update the job definition resources with sizes that reflect current job profiles, then register a new revision and submit with that revision.  
- To increase throughput: adjust the Compute Environment’s max vCPU, diversify instance types, and ensure EC2 vCPU quotas are sufficient. If storage is the bottleneck, provision additional EFS throughput or stage to local storage within jobs.

**Roll‑Forward/Roll‑Back**  
- Keep an immutable image tag history in ECR and maintain multiple job definition revisions.  
- Document which tag and revision produced a given result set via job tags and output directory naming.

---

## Monitoring & Troubleshooting

- Watch CloudWatch metrics for CPU, memory, disk, and EFS throughput to detect under‑utilization or I/O saturation.  
- Inspect per‑job logs for training and data‑loading times to identify bottlenecks.  
- If jobs are stuck in `RUNNABLE`, investigate Batch capacity, CE instance types, fair‑share policies, and EC2 service quotas.  
- Keep retry attempts low while debugging to avoid queue congestion.

---

## Quick Checklist

- EFS mounted and healthy; input data in place under the expected paths.  
- Docker image built and pushed to ECR; PYTHONPATH set to include app and scripts.  
- Job definition registered with correct image tag, resources, EFS mount, and command.  
- Submission loop passes the correct parameters (age, indices) and uses the intended job definition revision.  
- Outputs written to age‑specific subfolders with age‑stamped filenames to avoid collisions.  
- Capacity plan validated (instance families/sizes, quotas, CE max vCPU) for the desired parallelism.

---

## Commands: Create Job Definition & Submit Jobs

### Register (or Update) the Job Definition
Use the AWS CLI to register the job definition JSON. This creates a **new revision** each time you run it.

```bash
# Register (or update) the job definition from a local JSON file
aws batch register-job-definition   --cli-input-json file://pyScripts/reg.json
```

**Tips**
- After pushing a new Docker image tag to ECR, update the `image` field in `reg.json` to that tag and re-register.
- Grab the latest revision number if needed:
```bash
aws batch describe-job-definitions   --job-definition-name aladyn-model-job   --status ACTIVE   --query 'jobDefinitions | sort_by(@,&revision)[-1].revision'
```

### Submit Jobs (single job)
```bash
# Submit one job (example age=55)
aws batch submit-job   --job-name "aladyn-0-10000-age-55"   --job-queue "launcher-test-spot-MM-Batch-JobQueue"   --job-definition "aladyn-model-job:<REVISION>"   --parameters start_index="0",end_index="10000",age="55"   --tags slice="0-10k",target_age="55"
```

### Submit Jobs (age sweep)
```bash
# Submit jobs for ages 40..79
JOB_QUEUE="launcher-test-spot-MM-Batch-JobQueue"
JOB_DEF="aladyn-model-job:<REVISION>"
START=0
END=10000

for AGE in $(seq 40 79); do
  NAME="aladyn-${START}-${END}-age-${AGE}"
  aws batch submit-job     --job-name "$NAME"     --job-queue "$JOB_QUEUE"     --job-definition "$JOB_DEF"     --parameters start_index="${START}",end_index="${END}",age="${AGE}"     --tags slice="0-10k",target_age="${AGE}"
done
```

**Notes**
- Replace `<REVISION>` with the latest number returned by `describe-job-definitions`, e.g., `aladyn-model-job:13`.
- The script writes outputs into `/data/output/age_<AGE>` with filenames that include the age, preventing collisions.
- Ensure the Compute Environment has sufficient vCPU and the instance types pack your chosen `vcpus` per job cleanly.
