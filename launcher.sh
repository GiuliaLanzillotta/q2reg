#!/bin/bash

# --- 1. Activate Environment ---
source ../clenv/bin/activate

# === Configuration ===
CONFIG="config_2D_classification.yaml"
EXP_NAME="study_q2approx_v1"

# Define available GPUs
GPUS=(1 2) 

# Hyperparams
SEEDS=(13 11 33)
REG_TYPES=("taylor-block" "taylor-diag" "taylor-full")
ALPHAS=(1.0)
MODES=("regularized" "sequential" "replay")
GRAD=("True" "False")

# =====================

# 2. Create a directory for logs so they don't clutter the root
LOG_DIR="logs/${EXP_NAME}"
mkdir -p "$LOG_DIR"

NUM_GPUS=${#GPUS[@]}
counter=0

echo "Active Environment: $VIRTUAL_ENV"
echo "Launching jobs on ${NUM_GPUS} GPUs. Logs are in: $LOG_DIR"

for seed in "${SEEDS[@]}"; do
    for reg in "${REG_TYPES[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            for mode in "${MODES[@]}"; do
                
                gpu_id=${GPUS[$((counter % NUM_GPUS))]}
                
                # Define a unique log filename for this run
                LOG_FILE="${LOG_DIR}/s${seed}_${reg}_${mode}.log"

                # 3. Add '-u' to python for unbuffered output
                cmd="python -u run_single_experiment.py \
                    --config $CONFIG \
                    --exp_name $EXP_NAME \
                    --seed $seed \
                    --reg_type $reg \
                    --alpha $alpha \
                    --mode $mode \
                    --accumulate"

                echo "[GPU $gpu_id] Running: Seed=$seed Reg=$reg Mode=$mode -> Logs: $LOG_FILE"
                
                # 4. Redirect output (>) and errors (2>&1) to the log file
                CUDA_VISIBLE_DEVICES=$gpu_id $cmd > "$LOG_FILE" 2>&1 &
                
                ((counter++))

                # Wait loop
                while [[ $(jobs -r -p | wc -l) -ge $NUM_GPUS ]]; do
                    wait -n 2>/dev/null || wait
                done
                
            done
        done
    done
done

wait
echo "All experiments finished."