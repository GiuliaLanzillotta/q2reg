#!/bin/bash

# --- 1. Activate Environment ---
source ../clenv/bin/activate

# === Configuration ===
CONFIG="config_2D_classification.yaml"
EXP_NAME="sgd_hyperparam_sweep_v1" # Updated directory name
LOG_DIR="logs/${EXP_NAME}"
mkdir -p "$LOG_DIR"

# Define available GPUs
GPUS=(2 2 3 3) 

# --- Fixed Settings ---
SEEDS=(11 21) # Reduced seeds for faster sweeping initially
REG_TYPES=("taylor-block" "taylor-diag")
CURVATURES=("true_fisher")    
ACC_OPTS=("true")              
MODES=("regularized")

# --- Sweep Settings (Hyperparameter Search) ---
# We are searching for the best combination of Alpha and SGD Learning Rate
ALPHAS=(0.1 0.5 1.0)
LRS=(0.0008 0.001 0.005 0.01) # Common SGD starting rates
BATCH_SIZES=(16 32)
OPTIMIZER="sgd"
SCHEDULER="one_cycle"    # Using one cycle decay
MOMENTUMS=(0.9)
# =====================

NUM_GPUS=${#GPUS[@]}
counter=0
PIDS=()

cleanup() {
    echo -e "\n[!] Killing background jobs..."
    for pid in "${PIDS[@]}"; do kill -9 $pid 2>/dev/null; done
    wait
    exit 1
}
trap cleanup SIGINT SIGTERM SIGHUP

echo "Launching SGD Sweep. Logs: $LOG_DIR"

for seed in "${SEEDS[@]}"; do
    for lr in "${LRS[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            for reg in "${REG_TYPES[@]}"; do
                for curv in "${CURVATURES[@]}"; do
                    for acc in "${ACC_OPTS[@]}"; do
                        for bs in "${BATCH_SIZES[@]}"; do
                            for mode in "${MODES[@]}"; do
                                for mom in "${MOMENTUMS[@]}"; do
                                    gpu_id=${GPUS[$((counter % NUM_GPUS))]}
                                    
                                    # Accumulation logic
                                    ACC_FLAG=""
                                    [ "$acc" = "true" ] && ACC_FLAG="--accumulate"
                                    # 1. Create a unique Run Name for this configuration
                                    RUN_NAME="s${seed}_lr${lr}_alpha${alpha}_bs${bs}_m${mom}_reg${reg}"
                                    LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
                                    # Pass this as the directory for metrics.pt and config.yaml
                                    RESULTS_SUBDIR="results/${EXP_NAME}/${RUN_NAME}"

                                    # Build command with SGD specific flags
                                    # Note: Ensure your run_single_experiment.py accepts --optimizer, --lr, and --scheduler
                                    cmd="python -u run_sweep.py \
                                        --config $CONFIG \
                                        --exp_name $EXP_NAME \
                                        --seed $seed \
                                        --reg_type $reg \
                                        --curvature $curv \
                                        --alpha $alpha \
                                        --mode $mode \
                                        --optimizer $OPTIMIZER \
                                        --lr $lr \
                                        --scheduler $SCHEDULER \
                                        --batch_size $bs \
                                        --mode $mode \
                                        --run_id $RUN_NAME \
                                        $ACC_FLAG"

                                    echo "[GPU $gpu_id] LR=$lr Alpha=$alpha -> $LOG_FILE"
                                    
                                    CUDA_VISIBLE_DEVICES=$gpu_id $cmd > "$LOG_FILE" 2>&1 &
                                    PIDS+=($!) 
                                    
                                    ((counter++))
                                    while [[ $(jobs -r -p | wc -l) -ge $NUM_GPUS ]]; do sleep 1; done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

wait
echo "Sweep finished."