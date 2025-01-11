import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Suppress Python warnings globally
os.environ["PYTHONWARNINGS"] = "ignore"

def run_script(start_index, end_index, work_dir, log_dir):
    """Run the script with given start and end indices and log outputs."""
    log_file = os.path.join(log_dir, f"batch_{start_index}_{end_index}.log")
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    command = [
        "python", "script_with_init_local.py", 
        "--start_index", str(start_index), 
        "--end_index", str(end_index), 
        "--work_dir", work_dir
    ]
    
    print(f"Starting process: {command}")
    
    with open(log_file, "w") as log:
        # Combine stdout and stderr into a single log file
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    
    if result.returncode == 0:
        print(f"Batch {start_index}-{end_index} completed successfully.")
    else:
        print(f"Batch {start_index}-{end_index} failed. See {log_file} for details.")
    return result.returncode

def main():
    # Define starting and ending ranges
    start_range = 0  # Starting index
    end_range = 200000    # Ending index
    step_size = 10000     # Batch size

    # Generate batch ranges
    batch_ranges = [
        (i, i + step_size) for i in range(start_range, end_range, step_size)
    ]

    work_dir = "./results"  # Directory to store outputs
    log_dir = "./logs"      # Directory to store logs
    
    # Create output directories
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Use a thread pool to run up to 2 processes concurrently
    max_concurrent_batches = 2
    with ThreadPoolExecutor(max_workers=max_concurrent_batches) as executor:
        futures = [
            executor.submit(run_script, start, end, work_dir, log_dir) 
            for start, end in batch_ranges
        ]

        # Wait for all processes to complete
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()