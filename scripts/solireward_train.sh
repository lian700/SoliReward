#!/bin/bash

# SoliReward Training Script with DeepSpeed
# Usage: bash solireward_train.sh (launch in only one node, distributed training will be auto-configured)

#================================================================
#                    CONFIGURATION SECTION
#           (All configurable variables are here)
#================================================================

#----------------------------------------------------------------
# Directory Configuration
#----------------------------------------------------------------
ROOT_DIR="/path/to/SoliReward"
SCRIPT_DIR="${ROOT_DIR}/scripts"
RESULT_DIR="${ROOT_DIR}/result"

#----------------------------------------------------------------
# Environment Configuration
#----------------------------------------------------------------
env_name='solireward'
conda_path='/path/to/conda/bin/activate'
CUTLASS_PATH='/path/to/cutlass'
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

# Network interface for distributed training (adjust based on your cluster)
# Common values: eth0, ib0, bond0, ens, etc.
NETWORK_INTERFACE='eth0'

#----------------------------------------------------------------
# Model Configuration
#----------------------------------------------------------------
MODEL_TYPE='InternVL3'
MODEL_PATH='/path/to/InternVL3-1B'

#----------------------------------------------------------------
# DeepSpeed Configuration
#----------------------------------------------------------------
# Select ZeRO stage based on model size
# 0: disabled, 1: optimizer, 2: optimizer+gradient, 3: optimizer+gradient+parameter
ZERO_STAGE=0

#----------------------------------------------------------------
# HPQA Configuration
#----------------------------------------------------------------
reduce_sequence='progressive_hierarchical_attention'
bt_label_smoothing=0.0
bce_loss_coeff=0.0
bce_label_smoothing=0.0
hierarchical_query_attn_layers='6 12 18 24' # 1B
# hierarchical_query_attn_layers='7 14 21 28' # 8B
# hierarchical_query_attn_layers='8 16 24 32 40 48' # 14B

#----------------------------------------------------------------
# Data Path Configuration
#----------------------------------------------------------------
EVAL_DATA_PATH='/path/to/eval_data.json'
TRAIN_DATA_PATH='/path/to/train_data.json'
OUTPUT_DIR_NAME="result-1002/physics-deformity-win_win"
enable_btt_loss=1

#----------------------------------------------------------------
# Training Hyperparameters
#----------------------------------------------------------------
NUM_EPOCHS=${NUM_EPOCHS:-3}
TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=1
LEARNING_RATE=1e-6
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.05
MAX_GRAD_NORM=1.0

# Effective batch size configuration
# GRADIENT_ACCUMULATION_STEPS will be auto-calculated based on:
# EFFECTIVE_BATCH_SIZE = TRAIN_BATCH_SIZE * NUM_GPUS * NUM_NODES * GRADIENT_ACCUMULATION_STEPS
EFFECTIVE_BATCH_SIZE=${EFFECTIVE_BATCH_SIZE:-320}

#----------------------------------------------------------------
# Loss Configuration
#----------------------------------------------------------------
reward_margin=3.0
bt_loss_coeff=${bt_loss_coeff:-1.0}
btt_loss_coeff=${btt_loss_coeff:-1.0}
bce_loss_coeff=${bce_loss_coeff:-0.0}
bt_label_smoothing=${bt_label_smoothing:-0.0}
bce_label_smoothing=${bce_label_smoothing:-0.0}

#----------------------------------------------------------------
# Vision Processing Parameters (for InternVL models)
#----------------------------------------------------------------
INPUT_SIZE=448
MAX_NUM=1
NUM_SEGMENTS=8
CENTER_CROP_VIDEO=${CENTER_CROP_VIDEO:-false}

#----------------------------------------------------------------
# Logging and Saving Configuration
#----------------------------------------------------------------
LOGGING_STEPS=10
SAVE_STEPS=100
EVAL_STEPS=50
SAVE_TOTAL_LIMIT=1000

#----------------------------------------------------------------
# Distributed Training Configuration
#----------------------------------------------------------------
MASTER_PORT=29501

#----------------------------------------------------------------
# Optional: Wait Before Starting (in seconds)
#----------------------------------------------------------------
# total_seconds=$((2 * 60 * 60))
total_seconds=0

#================================================================
#                    END OF CONFIGURATION
#================================================================

#================================================================
# TIME STRING GENERATION
#================================================================
datetime_str=$(date +%Y%m%d%H%M%S)

# Build OUTPUT_DIR with datetime
OUTPUT_DIR="${OUTPUT_DIR_NAME}/${reduce_sequence}-${MODEL_TYPE}-bce_${bce_loss_coeff}-${datetime_str}"

# Apply default values
ZERO_STAGE=${ZERO_STAGE:-0}
hierarchical_query_attn_layers=${hierarchical_query_attn_layers:-'6 12 18 24'}

#================================================================
# OPTIONAL: WAIT BEFORE STARTING
#================================================================
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

# NCCL Configuration for Distributed Training
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export GLOO_SOCKET_CONNECT_TIMEOUT_MSEC=300000
export DEEPSPEED_LOG_LEVEL=DEBUG

# Basic NCCL settings
export NCCL_TIMEOUT=3600
export NCCL_SOCKET_IFNAME=${NETWORK_INTERFACE}
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# Advanced NCCL settings (uncomment and adjust based on your cluster configuration)
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_TIMEOUT=240
# export NCCL_IB_HCA=mlx5_0,mlx5_1  # Adjust based on your InfiniBand devices
# export UCX_NET_DEVICES=${NETWORK_INTERFACE}
# export NCCL_NET_GDR_LEVEL=2
# export NCCL_IB_QPS_PER_CONNECTION=4

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
NUM_GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS_PER_NODE GPUs per node"

# Hostfile configuration
# Option 1: Use existing hostfile (recommended for multi-node training)
hostfile_path="${HOSTFILE_PATH:-./hostfile}"

# Option 2: Auto-generate hostfile for single-node training
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

# Dynamically create hostfile with slots information
hostfile_with_slots_path="${WORK_DIR}/hostfile_with_slots_$(date +%s).tmp"
echo "Creating hostfile with GPU slots information..."

# Ensure the hostfile ends with a newline to avoid missing the last line
if [[ -s "$hostfile_path" ]] && [[ $(tail -c1 "$hostfile_path" | wc -l) -eq 0 ]]; then
    echo "" >> "$hostfile_path"
    echo "Added missing newline to hostfile: $hostfile_path"
fi

# Create new hostfile with slots (each node has NUM_GPUS_PER_NODE GPUs)
> "$hostfile_with_slots_path"
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
    
    # Use the predefined NUM_GPUS_PER_NODE value
    echo "Adding host: $host with $NUM_GPUS_PER_NODE GPUs"
    
    # Add to new hostfile
    echo "$host slots=$NUM_GPUS_PER_NODE" >> "$hostfile_with_slots_path"
done < "$hostfile_path"

echo "Created new hostfile with slots: $hostfile_with_slots_path"
echo "Contents of new hostfile:"
cat "$hostfile_with_slots_path"

# Auto-configure cluster from hostfiles
# MASTER_ADDR=$(cat "$hostfile_path" | grep -v '^#' | grep -v '^[[:space:]]*$' | sort | head -1)
MASTER_ADDR=$localhost_ip
NUM_NODES=$(cat "$hostfile_with_slots_path" | grep -v '^#' | grep -v '^[[:space:]]*$' | wc -l)

echo "Cluster configuration:"
echo "  Master address: $MASTER_ADDR"
echo "  Number of nodes: $NUM_NODES"
echo "  Original hosts:"
cat "$hostfile_path" | sort | nl
echo "  New hosts (with slots):"
cat "$hostfile_with_slots_path" | nl

#================================================================
# ENVIRONMENT CONFIGURATION
#================================================================

echo "=== Environment Configuration ==="

echo "Environment: $env_name"
echo "Result directory: $RESULT_DIR"

# Activate conda environment
echo "Activating conda environment: $env_name"
source $conda_path $env_name
echo "Conda environment activated successfully"

#================================================================
# DISTRIBUTED TRAINING SETUP
#================================================================

echo "=== Distributed Training Setup ==="

export CUTLASS_PATH="$CUTLASS_PATH"

#================================================================
# OUTPUT DIRECTORY CONFIGURATION
#================================================================

echo "=== Output Directory Configuration ==="

# Update output directory to use absolute path based on RESULT_DIR
OUTPUT_DIR="${RESULT_DIR}/${OUTPUT_DIR}"

echo "Model: $MODEL_PATH"
echo "Train Dataset: $TRAIN_DATA_PATH"
echo "Validation Dataset: $EVAL_DATA_PATH"
echo "Output directory: $OUTPUT_DIR"

#================================================================
# AUTO-CALCULATE GRADIENT ACCUMULATION STEPS
#================================================================

echo "=== Calculating Gradient Accumulation Steps ==="

# Calculate total number of GPUs
TOTAL_GPUS=$((NUM_GPUS_PER_NODE * NUM_NODES))
echo "Total GPUs: $TOTAL_GPUS (${NUM_GPUS_PER_NODE} GPUs/node × ${NUM_NODES} nodes)"

# Calculate gradient accumulation steps
# EFFECTIVE_BATCH_SIZE = TRAIN_BATCH_SIZE * TOTAL_GPUS * GRADIENT_ACCUMULATION_STEPS
# Therefore: GRADIENT_ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE / (TRAIN_BATCH_SIZE * TOTAL_GPUS)
GRADIENT_ACCUMULATION_STEPS=$((EFFECTIVE_BATCH_SIZE / (TRAIN_BATCH_SIZE * TOTAL_GPUS)))

# Ensure at least 1 accumulation step
if [ $GRADIENT_ACCUMULATION_STEPS -lt 1 ]; then
    GRADIENT_ACCUMULATION_STEPS=1
    echo "Warning: Calculated accumulation steps < 1, setting to 1"
fi

# Calculate actual effective batch size (may differ due to integer division)
ACTUAL_EFFECTIVE_BATCH_SIZE=$((TRAIN_BATCH_SIZE * TOTAL_GPUS * GRADIENT_ACCUMULATION_STEPS))

echo "Effective batch size (target): $EFFECTIVE_BATCH_SIZE"
echo "Effective batch size (actual): $ACTUAL_EFFECTIVE_BATCH_SIZE"
echo "Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"


# Auto-select DeepSpeed config based on ZeRO stage
case $ZERO_STAGE in
    0)
        DEEPSPEED_CONFIG="configs/deepspeed_config_stage0.json"
        ;;
    1)
        DEEPSPEED_CONFIG="configs/deepspeed_config_stage1.json"
        ;;
    2)
        DEEPSPEED_CONFIG="configs/deepspeed_config_stage2.json"
        ;;
    3)
        DEEPSPEED_CONFIG="configs/deepspeed_config_stage3.json"
        ;;
    *)
        echo "Error: Invalid ZERO_STAGE: $ZERO_STAGE"
        exit 1
        ;;
esac

echo "DeepSpeed configuration: $DEEPSPEED_CONFIG (ZeRO Stage $ZERO_STAGE)"

# flash attention and gradient checkpointing settings
echo "Using default configurations for model type: $MODEL_TYPE"
GRADIENT_CHECKPOINTING="--gradient_checkpointing"
ATTN_IMPLEMENTATION="flash_attention_2"

# Additional arguments
PRECISION="--bf16"  # Use --fp16 for older GPUs that don't support bf16

#================================================================
# PRE-TRAINING CLEANUP
#================================================================

echo "=== Pre-training Cleanup ==="

# Kill existing training scripts
pssh -i -t 0 -h $hostfile_path "pkill -9 -f solireward_train.py" 2>/dev/null || true

# Create output directory
mkdir -p $OUTPUT_DIR


# Copy script for tracking
script_name=$(basename "$0")
cp "$0" "$OUTPUT_DIR/$script_name"
echo "Script copied to: $OUTPUT_DIR/$script_name"

#================================================================
# TRAINING EXECUTION
#================================================================

echo "=== Starting Training ==="
echo "Training configuration summary:"
echo "  Model: $MODEL_PATH"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Batch size per device: $TRAIN_BATCH_SIZE"
echo "  Max segments: $NUM_SEGMENTS"
echo "  ZeRO Stage: $ZERO_STAGE"
echo "  DeepSpeed Config: $DEEPSPEED_CONFIG"
echo "  Hostfile: $hostfile_with_slots_path"

# Export environment variables for distributed training
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"

# Build the command
CMD="deepspeed \
    --hostfile $hostfile_with_slots_path \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    scripts/solireward_train.py \
    --model_name_or_path $MODEL_PATH \
    --train_data_path $TRAIN_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --max_grad_norm $MAX_GRAD_NORM \
    --input_size $INPUT_SIZE \
    --max_num $MAX_NUM \
    --num_segments $NUM_SEGMENTS \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --dataloader_num_workers $NUM_GPUS_PER_NODE \
    --seed 42 \
    --report_to tensorboard \
    --run_name reward_model_${datetime_str} \
    --eval_data_path $EVAL_DATA_PATH \
    --eval_steps $EVAL_STEPS \
    --eval_strategy steps \
    --do_eval \
    $GRADIENT_CHECKPOINTING \
    $PRECISION \
    --save_only_model \
    --model_type $MODEL_TYPE \
    --reward_margin $reward_margin \
    --bt_loss_coeff $bt_loss_coeff \
    --btt_loss_coeff $btt_loss_coeff \
    --bce_loss_coeff $bce_loss_coeff \
    --attn_implementation $ATTN_IMPLEMENTATION \
    --reduce_sequence $reduce_sequence \
    --hierarchical_query_attn_layers $hierarchical_query_attn_layers \
    --bt_label_smoothing $bt_label_smoothing \
    --bce_label_smoothing $bce_label_smoothing \
    --enable_btt_loss ${enable_btt_loss:-0} \
    --deepspeed $DEEPSPEED_CONFIG"

if [[ "$CENTER_CROP_VIDEO" == "true" || "$CENTER_CROP_VIDEO" == "1" ]]; then
    CMD="$CMD \
    --center_crop_video"
fi



nohup $CMD > $OUTPUT_DIR/$localhost_ip.log 2>&1 &

echo "Training started in background"
echo "Log file: $OUTPUT_DIR/$localhost_ip.log"

echo "=== Training Script Completed ==="
echo "Check log file for training progress: $OUTPUT_DIR/$localhost_ip.log"
