#!/bin/bash

# Change the absolute path first!

OUTPUT_DIR="output_infer"
DATA_ROOT_DIR="/home/ubuntu/ahmed-etri"

# Single GPU ID to use (change this if you want to use a different GPU)
GPU_ID=0

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

# Function: Run task sequentially on single GPU
run_scene() {
    local DATASET=$1
    local SCENE=$2
    local N_VIEW=$3
    local gs_train_iter=$4
    
    SOURCE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/
    GT_POSE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/
    IMAGE_PATH=${SOURCE_PATH}images
    MODEL_PATH=./${OUTPUT_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views

    # Create necessary directories
    mkdir -p ${MODEL_PATH}

    echo "======================================================="
    echo "Starting process: ${DATASET}/${SCENE} (${N_VIEW} views/${gs_train_iter} iters) on GPU ${GPU_ID}"
    echo "======================================================="

    # (1) Co-visible Global Geometry Initialization
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Co-visible Global Geometry Initialization..."
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

    # (2) Train: jointly optimize pose with RLGS and weight saving
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training with GRU weight persistence..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train_rlgs.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    -r 1 \
    --n_views ${N_VIEW} \
    --iterations ${gs_train_iter} \
    --pp_optimizer \
    --optim_pose \
    --rlgs_enabled \
    --rlgs_K 20 \
    --rlgs_N_lr 3 \
    --rlgs_reshuffle_interval 100 \
    --rlgs_reward_set_len 2 \
    --rlgs_entropy_coef 0.01 \
    --save-weights \
    > ${MODEL_PATH}/02_train.log 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed. Log saved in ${MODEL_PATH}/02_train.log"
    
    # Check if GRU weights were saved
    if [ -f "rlgs/gru_priors.pth" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ GRU priors saved - next scene will use these weights"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  No GRU priors found - check weight saving"
    fi

    # (3) Render-Video
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting rendering training views..."
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
    echo "Task completed: ${DATASET}/${SCENE} (${N_VIEW} views/${gs_train_iter} iters) on GPU ${GPU_ID}"
    echo "======================================================="
}

# Main execution - sequential processing
echo "🚀 Starting sequential training on GPU ${GPU_ID} with GRU weight persistence"
echo "💾 Each scene will save GRU weights for the next scene to use as priors"

# Calculate total tasks
total_tasks=$((${#DATASETS[@]} * ${#SCENES[@]} * ${#N_VIEWS[@]}))
current_task=0

# Sequential execution loop
for DATASET in "${DATASETS[@]}"; do
    for SCENE in "${SCENES[@]}"; do
        for N_VIEW in "${N_VIEWS[@]}"; do
            current_task=$((current_task + 1))
            echo ""
            echo "🔄 Processing task $current_task / $total_tasks: ${DATASET}/${SCENE}"
            
            # Run scene and wait for completion before next scene
            run_scene "$DATASET" "$SCENE" "$N_VIEW" "$gs_train_iter"
            
            # Add small delay between scenes
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting 5 seconds before next scene..."
            sleep 5
        done
    done
done

echo ""
echo "======================================================="
echo "✅ All tasks completed sequentially! Processed $total_tasks scenes in total."
echo "🧠 GRU priors have been progressively learned across all scenes."
echo "======================================================="
