#!/bin/bash

# Multi-GPU Distributed Inference Script for SoliReward
# This script splits input JSON across all GPUs and runs single-GPU inference on each
# Usage: bash solireward_infer.sh

#================================================================
# ROOT DIRECTORY CONFIGURATION
#================================================================
# Explicitly specify the root directory for the project
ROOT_DIR="/path/to/SoliReward"
SCRIPT_DIR="${ROOT_DIR}/scripts"
RESULT_DIR="${ROOT_DIR}/result-inference"

# Network interface for distributed inference (adjust based on your cluster)
# Common values: eth0, ib0, bond0, ens, etc.
NETWORK_INTERFACE='eth0'

#================================================================
# SETTINGS SECTION (CONFIGURE THIS)
#================================================================

# Input JSON file
INPUT_JSON='/path/to/input.json'

# Model checkpoint path
CHECKPOINT_PATH='/path/to/checkpoint'

# Reward model task type: "text_alignment" or "phy_deform"
REWARD_MODEL_TASK_TYPE='phy_deform'

# Output directory (relative to RESULT_DIR)
OUTPUT_DIR='OOD'

# Inference settings
BATCH_SIZE=1
DTYPE="bf16"
DEVICE="cuda"
NUM_GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
max_samples=${max_samples:--1}  # Default to -1 (process all samples) if not set

# Polling interval for checking completion (in seconds)
POLL_INTERVAL=30

#================================================================
# OPTIONAL: WAIT BEFORE STARTING
#================================================================

# total_seconds=$((2 * 60 * 60))
total_seconds=0
end_ts=$(( $(date +%s) + total_seconds ))

while :; do
    now_ts=$(date +%s)
    remaining=$(( end_ts - now_ts ))
    if (( remaining <= 0 )); then
        break
    fi
    hrs=$(( remaining / 3600 ))
    mins=$(( (remaining % 3600) / 60 ))
    printf "Time remaining: %02dh %02dm at %s\n" "$hrs" "$mins" "$(date)"
    sleep $(( remaining > 60 ? 60 : remaining ))
done

echo "Resuming at: $(date)"

#================================================================
# INITIALIZATION
#================================================================

# Time string for unique identification
datetime_str=$(date +%Y%m%d%H%M)

# Change work dir to SoliReward root (using ROOT_DIR defined at the top)
WORK_DIR="${ROOT_DIR}"
cd $WORK_DIR || { echo "Error: Cannot change to work directory $WORK_DIR"; exit 1; }
echo "Working directory: $WORK_DIR"
echo "Script directory: $SCRIPT_DIR"
echo "Result directory: $RESULT_DIR"

#================================================================
# ENVIRONMENT SETUP
#================================================================

echo "=== Environment Setup ==="

# Clear proxy settings
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# Environment configuration
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Get local IP address
localhost_ip=$(ip addr show ${NETWORK_INTERFACE} | grep "inet " | awk '{print $2}' | cut -d/ -f1)
if [ -z "$localhost_ip" ]; then
    echo "Warning: Could not detect IP from ${NETWORK_INTERFACE}, using hostname"
    localhost_ip=$(hostname -I | awk '{print $1}')
fi

#================================================================
# CLUSTER SETUP
#================================================================

echo "=== Cluster Configuration ==="

# Hostfile configuration
# Option 1: Use existing hostfile (recommended for multi-node inference)
hostfile_path="${HOSTFILE_PATH:-./hostfile}"

# Option 2: Auto-generate hostfile for single-node inference
if [ ! -f "$hostfile_path" ] || [ ! -s "$hostfile_path" ]; then
    echo "Host file not found, creating single-node hostfile..."
    echo "localhost" > "$hostfile_path"
    echo "Created single-node hostfile at $hostfile_path"
else
    echo "Using existing host file $hostfile_path"
fi

# Validate hosts file
if [ ! -f "$hostfile_path" ] || [ ! -s "$hostfile_path" ]; then
    echo "Error: Host file $hostfile_path not found or empty after generation"
    exit 1
fi

# Read hosts and calculate total GPUs
echo "Reading hostfile: $hostfile_path"
declare -a NODE_IPS
while IFS= read -r host || [[ -n "$host" ]]; do
    # Skip empty lines and comments
    if [[ -z "$host" || "$host" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # Clean the host name (remove any extra spaces)
    host=$(echo "$host" | xargs)
    
    # Skip if host is still empty after cleaning
    if [[ -z "$host" ]]; then
        continue
    fi
    
    NODE_IPS+=("$host")
done < "$hostfile_path"

NUM_NODES=${#NODE_IPS[@]}
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))

echo "Cluster configuration:"
echo "  Number of nodes: $NUM_NODES"
echo "  GPUs per node: $NUM_GPUS_PER_NODE"
echo "  Total GPUs: $TOTAL_GPUS"
echo "  Node IPs:"
for i in "${!NODE_IPS[@]}"; do
    echo "    Node $i: ${NODE_IPS[$i]}"
done

#================================================================
# ENVIRONMENT CONFIGURATION
#================================================================

echo "=== Environment Configuration ==="

env_name='solireward'
conda_path='/path/to/conda/bin/activate'

echo "Environment: $env_name"

# Activate conda environment
echo "Activating conda environment: $env_name"
source $conda_path $env_name
echo "Conda environment activated successfully"

#=OUTPUT DIRECTORY CONFIGURATION
#================================================================

echo "=== Output Directory Configuration ==="

# Update output directory to use absolute path based on RESULT_DIR
OUTPUT_DIR="${RESULT_DIR}/${OUTPUT_DIR}"

echo "Input JSON: $INPUT_JSON"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"

#================================================================
# ===============================================================
# SPLIT INPUT JSON
#================================================================

echo "=== Splitting Input JSON ==="

# Create"${SCRIPT_DIR}/split_json_for_inference.py"
mkdir -p "$OUTPUT_DIR"
cp $0 "$OUTPUT_DIR/"  # Copy this script to output dir for reference
SPLIT_DIR="${OUTPUT_DIR}/splits"
DONE_DIR="${OUTPUT_DIR}/done_markers"

# Create done markers directory
mkdir -p "$DONE_DIR"

# Clean up any existing done markers
rm -f "${DONE_DIR}"/gpu_*.done 2>/dev/null

# Split JSON using Python script (located in scripts/)
python3 "${SCRIPT_DIR}/split_json_for_inference.py" \
    --input_file "$INPUT_JSON" \
    --output_dir "$SPLIT_DIR" \
    --num_splits $TOTAL_GPUS \
    --max_samples $max_samples

if [ $? -ne 0 ]; then
    echo "Error: Failed to split JSON file"
    exit 1
fi

#================================================================
# LAUNCH INFERENCE ON ALL GPUS
#================================================================

echo "=== Launching Inference on All GPUs ==="

gpu_global_id=0
declare -a ACTIVE_GPUS=()

for node_id in "${!NODE_IPS[@]}"; do
    node_ip="${NODE_IPS[$node_id]}"
    
    echo "Configuring node $node_id (${node_ip})..."
    
    for local_gpu_id in $(seq 0 $((NUM_GPUS_PER_NODE - 1))); do
        input_file="${SPLIT_DIR}/input_gpu_${gpu_global_id}.json"
        output_file="${OUTPUT_DIR}/output_gpu_${gpu_global_id}.json"
        log_file="${OUTPUT_DIR}/inference_gpu_${gpu_global_id}.log"
        log_file=$(realpath "$log_file")
        done_marker="${DONE_DIR}/gpu_${gpu_global_id}.done"
        
        # Check if input file exists and has data
        if [ ! -f "$input_file" ]; then
            echo "Warning: Input file not found for GPU $gpu_global_id: $input_file"
            # Create an empty done marker for skipped GPUs
            touch "$done_marker"
            gpu_global_id=$((gpu_global_id + 1))
            continue
        fi
        
        # Check if input file is empty
        sample_count=$(python3 -c "import json; data=json.load(open('$input_file')); print(len(data))")
        if [ "$sample_count" -eq 0 ]; then
            echo "Skipping GPU $gpu_global_id (no samples assigned)"
            # Create an empty done marker for skipped GPUs
            touch "$done_marker"
            gpu_global_id=$((gpu_global_id + 1))
            continue
        fi
        
        echo "  GPU $gpu_global_id (Node $node_id, Local GPU $local_gpu_id): $sample_count samples"
        
        # Track active GPUs
        ACTIVE_GPUS+=($gpu_global_id)
        
        # Build command for this GPU (includes creating done marker after completion)
        CMD="cd $WORK_DIR && \
source $conda_path $env_name && \
export CUDA_VISIBLE_DEVICES=$local_gpu_id && \
export PYTHONUNBUFFERED=1 && \
python ${SCRIPT_DIR}/solireward_infer.py \\
    --model_name_or_path $CHECKPOINT_PATH \
    --input_file $input_file \
    --output_file $output_file \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --dtype $DTYPE \
    --reward_model_task_type $REWARD_MODEL_TASK_TYPE \
    --max_samples -1 && \
touch $done_marker"
        
        # Launch on remote node (or local if it's current node)
        if [ "$node_ip" == "$localhost_ip" ]; then
            # Launch locally
            echo "Launching inference on local GPU $local_gpu_id (Global ID: $gpu_global_id)..."
            nohup bash -c "$CMD" > "$log_file" 2>&1 &
        else
            # Launch remotely via ssh
            echo "Launching inference on remote node $node_ip, GPU $local_gpu_id (Global ID: $gpu_global_id)..."
            ssh -o StrictHostKeyChecking=no "$node_ip" "nohup bash -c '$CMD' > $log_file 2>&1 &" &
        fi
        
        gpu_global_id=$((gpu_global_id + 1))
        
        # Small delay to avoid overwhelming the system
        sleep 0.5
    done
done

echo "All inference jobs launched!"
echo ""
echo "=== Summary ==="
echo "Total GPUs used: $TOTAL_GPUS"
echo "Active GPUs: ${#ACTIVE_GPUS[@]}"
echo "Input splits directory: $SPLIT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Done markers directory: $DONE_DIR"
echo "Log files: ${OUTPUT_DIR}/inference_gpu_*.log"
echo "Output files: ${OUTPUT_DIR}/output_gpu_*.json"

#================================================================
# WAIT FOR ALL GPUS TO COMPLETE
#================================================================

echo ""
echo "=== Waiting for all GPUs to complete inference ==="

while true; do
    completed_count=0
    
    # Count completed GPUs
    for gpu_id in $(seq 0 $((TOTAL_GPUS - 1))); do
        if [ -f "${DONE_DIR}/gpu_${gpu_id}.done" ]; then
            completed_count=$((completed_count + 1))
        fi
    done
    
    # Print progress
    current_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$current_time] Progress: $completed_count / $TOTAL_GPUS GPUs completed"
    
    # Check if all completed
    if [ $completed_count -eq $TOTAL_GPUS ]; then
        echo ""
        echo "=== All GPUs completed inference! ==="
        break
    fi
    
    # Wait before next check
    sleep $POLL_INTERVAL
done

#================================================================
# MERGE RESULTS
#================================================================

echo ""
echo "=== Merging inference results ==="

python3 "${SCRIPT_DIR}/merge_inference_results.py" --input_dir "${OUTPUT_DIR}"

if [ $? -ne 0 ]; then
    echo "Error: Failed to merge inference results"
    exit 1
fi

echo ""
echo "=== Script Completed Successfully ==="
echo "Final merged results are in: ${OUTPUT_DIR}"
