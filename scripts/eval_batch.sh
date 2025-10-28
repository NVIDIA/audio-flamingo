#!/bin/bash
set -e

# ------------------- Environment -------------------
export NCCL_DEBUG=WARN
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export VILA_DATASETS=audio_test

# ------------------- Configs -------------------
MODEL_PATH=$1
INFER_JSON=$2
YAML_FILE="llava/eval/registry_audio.yaml"
BATCH_SIZE=8
THINK_MODE=true  # set to true if you want to enable think mode

# ------------------- Helper -------------------
get_tasks_from_yaml() {
    grep '^[^[:space:]]' "$YAML_FILE" | sed 's/://'
}

# ------------------- Main -------------------
if [[ -n "$INFER_JSON" && -f "$INFER_JSON" ]]; then
    echo "Running single evaluation with infer_json: $INFER_JSON"
    TASK=$(basename "$INFER_JSON" .json)
    sh scripts/eval/eval_audio_batch.sh "$MODEL_PATH" auto "$TASK" "$THINK_MODE" "$INFER_JSON" "$BATCH_SIZE"
else
    echo "No infer_json provided. Running all tasks from: $YAML_FILE"
    for TASK in $(get_tasks_from_yaml); do
        echo "Submitting job for task: $TASK"
        sh scripts/eval/eval_audio_batch.sh "$MODEL_PATH" auto "$TASK" "$THINK_MODE"
    done
fi
