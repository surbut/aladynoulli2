#!/bin/bash

# Script to run all batches for age 70 filtered predictions
# This will process patients with max_censor > 70 in batches of 25K

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/run_offset_age70_filtered.py"

# Configuration (use $HOME instead of ~ for parameter expansion)
DATA_DIR="${DATA_DIR:-$HOME/aladyn_project/data_for_running/}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/aladyn_project/output/age70_filtered/}"
VENV_PATH="${VENV_PATH:-$HOME/aladyn_project/aladyn_env}"
BATCH_SIZE=25000
MIN_CENSOR_AGE=70.0
MAX_BATCHES=10  # Run up to 10 batches (250K patients max)

# Activate virtual environment
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Activated virtual environment: $VENV_PATH"
    echo "Python path: $(which python)"
else
    echo "Warning: Virtual environment not found at $VENV_PATH, using system python"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Log file for overall progress
MAIN_LOG="$OUTPUT_DIR/logs/batch_run_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================" | tee -a "$MAIN_LOG"
echo "Age 70 Filtered Predictions - Batch Runner" | tee -a "$MAIN_LOG"
echo "Started: $(date)" | tee -a "$MAIN_LOG"
echo "==========================================" | tee -a "$MAIN_LOG"
echo "Data directory: $DATA_DIR" | tee -a "$MAIN_LOG"
echo "Output directory: $OUTPUT_DIR" | tee -a "$MAIN_LOG"
echo "Batch size: $BATCH_SIZE" | tee -a "$MAIN_LOG"
echo "Min censor age: $MIN_CENSOR_AGE" | tee -a "$MAIN_LOG"
echo "Virtual environment: $VENV_PATH" | tee -a "$MAIN_LOG"
echo "Python: $(which python)" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Track successful and failed batches
SUCCESSFUL_BATCHES=()
FAILED_BATCHES=()

# Run batches
for BATCH_NUM in $(seq 0 $((MAX_BATCHES - 1))); do
    START_INDEX=$((BATCH_NUM * BATCH_SIZE))
    END_INDEX=$((START_INDEX + BATCH_SIZE))
    
    BATCH_LOG="$OUTPUT_DIR/logs/batch_${BATCH_NUM}_$(date +%Y%m%d_%H%M%S).log"
    
    # Check if output file already exists (skip if it does)
    # Output file pattern: pi_fixedphi_age_40_offset_0_filtered_censor70.0_batch_START_END.pt
    EXPECTED_OUTPUT_PATTERN="pi_fixedphi_age_40_offset_0_filtered_censor${MIN_CENSOR_AGE}_batch_${START_INDEX}_*.pt"
    EXISTING_OUTPUT=$(find "$OUTPUT_DIR" -name "$EXPECTED_OUTPUT_PATTERN" 2>/dev/null | head -1)
    
    if [ -n "$EXISTING_OUTPUT" ] && [ -f "$EXISTING_OUTPUT" ]; then
        echo "----------------------------------------" | tee -a "$MAIN_LOG"
        echo "SKIPPING batch $BATCH_NUM (output already exists)" | tee -a "$MAIN_LOG"
        echo "  Existing file: $(basename "$EXISTING_OUTPUT")" | tee -a "$MAIN_LOG"
        echo "  Started: $(date)" | tee -a "$MAIN_LOG"
        SUCCESSFUL_BATCHES+=($BATCH_NUM)
        echo "✓ Batch $BATCH_NUM skipped (already completed)" | tee -a "$MAIN_LOG"
        echo "Completed: $(date)" | tee -a "$MAIN_LOG"
        echo "" | tee -a "$MAIN_LOG"
        continue
    fi
    
    echo "----------------------------------------" | tee -a "$MAIN_LOG"
    echo "Starting batch $BATCH_NUM (indices $START_INDEX to $END_INDEX)" | tee -a "$MAIN_LOG"
    echo "Started: $(date)" | tee -a "$MAIN_LOG"
    echo "Log file: $BATCH_LOG" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
    
    # Run the Python script (from script's directory)
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        echo "✗ Error: Python script not found at $PYTHON_SCRIPT" | tee -a "$MAIN_LOG"
        echo "Stopping batch processing." | tee -a "$MAIN_LOG"
        break
    fi
    
    python "$PYTHON_SCRIPT" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --start_index "$START_INDEX" \
        --end_index "$END_INDEX" \
        --min_censor_age "$MIN_CENSOR_AGE" \
        > "$BATCH_LOG" 2>&1
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Batch $BATCH_NUM completed successfully" | tee -a "$MAIN_LOG"
        SUCCESSFUL_BATCHES+=($BATCH_NUM)
        
        # Check if we've processed all available patients
        # If the actual_end is less than END_INDEX, we're done
        # Pattern: "After subsetting [0:25000]:" - we want the second number (after the colon)
        if grep -q "After subsetting" "$BATCH_LOG"; then
            # Extract the end index (second number after colon in brackets)
            ACTUAL_END=$(grep "After subsetting" "$BATCH_LOG" | grep -oP '\[\d+:\K\d+' | head -1)
            if [ -n "$ACTUAL_END" ] && [ "$ACTUAL_END" -lt "$END_INDEX" ]; then
                echo "Reached end of filtered data at index $ACTUAL_END" | tee -a "$MAIN_LOG"
                echo "Stopping batch processing." | tee -a "$MAIN_LOG"
                break
            fi
        fi
    else
        echo "✗ Batch $BATCH_NUM failed with exit code $EXIT_CODE" | tee -a "$MAIN_LOG"
        FAILED_BATCHES+=($BATCH_NUM)
        echo "Check log file: $BATCH_LOG" | tee -a "$MAIN_LOG"
        
        # Check if failure is due to running out of data (not a real error)
        if grep -qi "index.*out of range\|end_index.*exceeds" "$BATCH_LOG"; then
            echo "Failure appears to be due to end of data. Stopping." | tee -a "$MAIN_LOG"
            break
        fi
    fi
    
    echo "Completed: $(date)" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
    
    # Small delay between batches
    sleep 2
done

# Summary
echo "==========================================" | tee -a "$MAIN_LOG"
echo "Batch Processing Summary" | tee -a "$MAIN_LOG"
echo "Completed: $(date)" | tee -a "$MAIN_LOG"
echo "==========================================" | tee -a "$MAIN_LOG"
echo "Successful batches: ${#SUCCESSFUL_BATCHES[@]}" | tee -a "$MAIN_LOG"
if [ ${#SUCCESSFUL_BATCHES[@]} -gt 0 ]; then
    echo "  ${SUCCESSFUL_BATCHES[*]}" | tee -a "$MAIN_LOG"
fi
echo "Failed batches: ${#FAILED_BATCHES[@]}" | tee -a "$MAIN_LOG"
if [ ${#FAILED_BATCHES[@]} -gt 0 ]; then
    echo "  ${FAILED_BATCHES[*]}" | tee -a "$MAIN_LOG"
fi
echo "" | tee -a "$MAIN_LOG"
echo "Output files saved to: $OUTPUT_DIR" | tee -a "$MAIN_LOG"
echo "Log files saved to: $OUTPUT_DIR/logs/" | tee -a "$MAIN_LOG"
echo "==========================================" | tee -a "$MAIN_LOG"

