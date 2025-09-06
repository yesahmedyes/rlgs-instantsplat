#!/bin/bash

# Change the absolute path first!

# K, N_lr, reshuffle_interval, reward_set_len, entropy_coef
# Define hyperparameter configurations as a bash array
declare -a EXP_CONFIGS=(
    "20,3,100,2,0.01"
    "20,3,100,4,0.01"
    "20,3,50,2,0.01"
    "20,3,50,4,0.01"
    "10,3,100,2,0.01"
    "10,3,100,4,0.01"
    "10,3,50,2,0.01"
    "10,3,50,4,0.01"
    "5,3,100,2,0.01"
    "5,3,100,4,0.01"
    "5,3,50,2,0.01"
    "5,3,50,4,0.01"
)

OUTPUT_DIR="output_infer_hyp"

DATA_ROOT_DIR="/home/ubuntu/ahmed-etri"

DATASETS=(tanks_templates)

N_VIEWS=(
    24
)

SCENES=(
horse
ballroom
barn
church
family
francis
ignatius
museum
)

gs_train_iter=1000

# Return all GPU indices
get_all_gpus() {
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' '
}

# GPUs with low memory usage (for info only)
get_all_available_gpus() {
    local mem_threshold=500
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
    $2 < threshold { print $1 }
    '
}

# Atomically acquire a free GPU by creating its lock dir
get_available_gpu() {
    local mem_threshold=500
    for gpu in $(get_all_gpus); do
        local mem_used
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id="$gpu" | tr -d ' ')
        if [ "$mem_used" -lt "$mem_threshold" ]; then
            if mkdir "$GPU_LOCK_DIR/gpu_${gpu}.lock" 2>/dev/null; then
                echo "$gpu"
                return 0
            fi
        fi
    done
    return 1
}

# Function: Run task on specified GPU with hyperparameters
run_on_gpu() {
    local GPU_ID=$1
    local DATASET=$2
    local SCENE=$3
    local N_VIEW=$4
    local gs_train_iter=$5
    local K=$6
    local N_lr=$7
    local reshuffle_interval=$8
    local reward_set_len=$9
    local entropy_coef=${10}
    
    # Create unique model path for this hyperparameter configuration
    local hyp_id="K${K}_Nlr${N_lr}_ri${reshuffle_interval}_rsl${reward_set_len}_ec${entropy_coef}"
    
    SOURCE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/
    GT_POSE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/
    IMAGE_PATH=${SOURCE_PATH}images
    MODEL_PATH=./${OUTPUT_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views/${hyp_id}

    # Create necessary directories
    mkdir -p ${MODEL_PATH}

    echo "======================================================="
    echo "Starting process: ${DATASET}/${SCENE} (${N_VIEW} views/${gs_train_iter} iters) with hyperparams ${hyp_id} on GPU ${GPU_ID}"
    echo "======================================================="

    # (1) Co-visible Global Geometry Initialization
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./init_geo.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    --n_views ${N_VIEW} \
    --focal_avg \
    --co_vis_dsp \
    --conf_aware_ranking \
    --infer_video \
    > ${MODEL_PATH}/01_init_geo.log 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Co-visible Global Geometry Initialization completed. Log saved in ${MODEL_PATH}/01_init_geo.log"
 
    # (2) Train: jointly optimize pose with hyperparameters
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training with hyperparams: K=${K}, N_lr=${N_lr}, reshuffle_interval=${reshuffle_interval}, reward_set_len=${reward_set_len}, entropy_coef=${entropy_coef}..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train_rlgs.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    -r 1 \
    --n_views ${N_VIEW} \
    --iterations ${gs_train_iter} \
    --pp_optimizer \
    --optim_pose \
    --rlgs_enabled \
    --rlgs_K ${K} \
    --rlgs_N_lr ${N_lr} \
    --rlgs_reshuffle_interval ${reshuffle_interval} \
    --rlgs_reward_set_len ${reward_set_len} \
    --rlgs_entropy_coef ${entropy_coef} \
    > ${MODEL_PATH}/02_train.log 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed. Log saved in ${MODEL_PATH}/02_train.log"

    # (3) Render-Video
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./render.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    -r 1 \
    --n_views ${N_VIEW} \
    --iterations ${gs_train_iter} \
    --infer_video \
    > ${MODEL_PATH}/03_render.log 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering completed. Log saved in ${MODEL_PATH}/03_render.log"

    echo "======================================================="
    echo "Task completed: ${DATASET}/${SCENE} (${N_VIEW} views/${gs_train_iter} iters) with hyperparams ${hyp_id} on GPU ${GPU_ID}"
    echo "======================================================="
    
    # Mark GPU as available when task completes
    rmdir "$GPU_LOCK_DIR/gpu_${GPU_ID}.lock" 2>/dev/null || true
}

# Initialize GPU tracking - mark all GPUs as available initially
echo "Initializing GPU tracking..."
all_gpus=$(get_all_available_gpus)
for gpu in $all_gpus; do
    # This part is no longer needed as we use a fixed lock directory
    # GPU_IN_USE[$gpu]=0
    echo "GPU $gpu is available (initial check)"
done

# After gs_train_iter=1000
GPU_LOCK_DIR="/tmp/rlgs_gpu_locks"
mkdir -p "$GPU_LOCK_DIR"

# Main loop - now includes hyperparameter configurations
total_tasks=$((${#DATASETS[@]} * ${#SCENES[@]} * ${#N_VIEWS[@]} * ${#EXP_CONFIGS[@]}))
current_task=0

echo "======================================================="
echo "Starting hyperparameter experiment with ${#EXP_CONFIGS[@]} configurations on ${#SCENES[@]} scenes"
echo "Total tasks: ${total_tasks}"
echo "Available GPUs: $(get_all_available_gpus | tr '\n' ' ')"
echo "======================================================="

for DATASET in "${DATASETS[@]}"; do
    for SCENE in "${SCENES[@]}"; do
        for N_VIEW in "${N_VIEWS[@]}"; do
            for config in "${EXP_CONFIGS[@]}"; do
                current_task=$((current_task + 1))
                
                # Parse hyperparameter configuration
                IFS=',' read -r K N_lr reshuffle_interval reward_set_len entropy_coef <<< "$config"
                
                echo "Processing task $current_task / $total_tasks: ${DATASET}/${SCENE} with config K=${K}, N_lr=${N_lr}, reshuffle_interval=${reshuffle_interval}, reward_set_len=${reward_set_len}, entropy_coef=${entropy_coef}"

                # Get available GPU
                GPU_ID=$(get_available_gpu)

                # If no GPU is available, wait for a while and retry
                while [[ -z "$GPU_ID" ]]; do
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No GPU available, waiting 60 seconds before retrying..."
                    sleep 60
                    GPU_ID=$(get_available_gpu)
                done

                # Run the task in the background
                (run_on_gpu $GPU_ID "$DATASET" "$SCENE" "$N_VIEW" "$gs_train_iter" "$K" "$N_lr" "$reshuffle_interval" "$reward_set_len" "$entropy_coef") &

                # Small delay to ensure GPU marking takes effect
                sleep 2
            done
        done
    done
done

# Wait for any remaining background tasks to complete
wait

echo "======================================================="
echo "All tasks completed! Processed $total_tasks tasks in total."
echo "======================================================="

# Collect and generate results table
echo "======================================================="
echo "Collecting results and generating markdown table..."
echo "======================================================="

# Change to the project root directory (assuming script is in scripts/ subdirectory)
cd "$(dirname "$0")/.."

# Run the metrics collection script
python collect_metrics.py --outputs_dir ${OUTPUT_DIR} --output_file hyperparameter_results.md

echo "======================================================="
echo "Hyperparameter experiment results saved to: hyperparameter_results.md"
echo "======================================================="
