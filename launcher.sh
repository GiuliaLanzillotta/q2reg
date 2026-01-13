#!/bin/bash

# --- 1. Activate Environment ---
source ../clenv/bin/activate

# === Configuration ===
CONFIG="config_toy_FullReg.yaml"
EXP_NAME="study_q2approx_v2"

# Define available GPUs (Using GPU 2 twice means 2 concurrent jobs on GPU 2)
GPUS=(1 1) 

# Hyperparams
SEEDS=(28 33 13)
REG_TYPES=("taylor-full")
CURVATURES=("fisher" "true_fisher")    
ACC_OPTS=("true" "false")              # (true means flag is present)
ALPHAS=(0.05 0.5 5.0)
MODES=("regularized")

# =====================

# 2. Create a directory for logs
LOG_DIR="logs/${EXP_NAME}"
mkdir -p "$LOG_DIR"

NUM_GPUS=${#GPUS[@]}
counter=0

# Track PIDs of background jobs
PIDS=()

# Cleanup function to kill all background processes
cleanup() {
    echo -e "\n[!] Termination signal received. Killing all background jobs..."
    for pid in "${PIDS[@]}"; do
        if ps -p $pid > /dev/null; then
            kill -9 $pid 2>/dev/null
        fi
    done
    wait
    echo "[!] All processes cleaned up. Exiting."
    exit 1
}

# Trap SIGINT (Ctrl+C), SIGTERM, and SIGHUP (terminal close)
trap cleanup SIGINT SIGTERM SIGHUP

echo "Active Environment: $VIRTUAL_ENV"
echo "Launching jobs on ${NUM_GPUS} GPUs. Logs are in: $LOG_DIR"

for seed in "${SEEDS[@]}"; do
    for reg in "${REG_TYPES[@]}"; do
        for curv in "${CURVATURES[@]}"; do
            for acc in "${ACC_OPTS[@]}"; do
                for alpha in "${ALPHAS[@]}"; do
                    for mode in "${MODES[@]}"; do
                        
                        gpu_id=${GPUS[$((counter % NUM_GPUS))]}
                        
                        # Handle the --accumulate flag and log naming
                        ACC_FLAG=""
                        ACC_LABEL="noacc"
                        if [ "$acc" = "true" ]; then
                            ACC_FLAG="--accumulate"
                            ACC_LABEL="acc"
                        fi

                        # Unique log filename including curvature and accumulation state
                        LOG_FILE="${LOG_DIR}/s${seed}_${reg}_${curv}_${mode}_${ACC_LABEL}_alpha${alpha}.log"

                        # Build command
                        cmd="python -u run_single_experiment.py \
                            --config $CONFIG \
                            --exp_name $EXP_NAME \
                            --seed $seed \
                            --reg_type $reg \
                            --curvature $curv \
                            --alpha $alpha \
                            --mode $mode \
                            $ACC_FLAG"

                        echo "[GPU $gpu_id] Seed=$seed Reg=$reg Curv=$curv Acc=$acc -> $LOG_FILE"
                        
                        # Launch and capture PID
                        CUDA_VISIBLE_DEVICES=$gpu_id $cmd > "$LOG_FILE" 2>&1 &
                        PIDS+=($!) # Store the PID of the process just started
                        
                        ((counter++))

                        while [[ $(jobs -r -p | wc -l) -ge $NUM_GPUS ]]; do
                            sleep 1
                        done

                    done  
                done
            done
        done
    done
done

wait
echo "All experiments finished."