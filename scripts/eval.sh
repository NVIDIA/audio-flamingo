#!/bin/bash

# Environment variables
export NCCL_DEBUG=WARN
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export VILA_DATASETS=audio_test

# Configs
DEFAULT_RUN_NAME="eval_all_audio"
YAML_FILE="llava/eval/registry_audio.yaml"
THINK_MODE=false  # or true if you want to enable it

# Function to parse task names from YAML
get_tasks_from_yaml() {
    grep '^[^[:space:]]' "$YAML_FILE" | sed 's/://'
}

# Loop through each task and submit a job
for TASK in $(get_tasks_from_yaml); do
    echo "Submitting job for task: $TASK"
    sh scripts/eval/eval_audio.sh nvidia/audio-flamingo-3 auto "$TASK" "$THINK_MODE"
done
