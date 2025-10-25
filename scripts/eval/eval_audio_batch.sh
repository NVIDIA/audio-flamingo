#!/bin/bash
export VILA_DATASETS=audio_test

MODEL_PATH=$1
CONV_MODE=$2
TASK=$3
THINK_MODE=$4
INFER_JSON=$5
BATCH_SIZE=$6
MODEL_NAME="AF3"
OUTPUT_DIR=${OUTPUT_DIR:-"runs/eval/$MODEL_NAME/audio"}


NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
GENERATION_CONFIG='{"max_new_tokens": 128}'

torchrun --nproc-per-node=$NPROC_PER_NODE \
    llava/eval/eval_audio_bench_batch.py \
    --model-base $MODEL_PATH \
    --infer-json $INFER_JSON \
    --conv-mode $CONV_MODE \
    --task $TASK \
    --think-mode $THINK_MODE \
    --generation-config "$GENERATION_CONFIG" \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE
