#!/bin/bash
export VILA_DATASETS=audio_test

MODEL_PATH=$1
CONV_MODE=$2
TASK=$3
THINK_MODE=$4
MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR=${OUTPUT_DIR:-"runs/eval/$MODEL_NAME/audio"}


NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
GENERATION_CONFIG='{"max_new_tokens": 128}'

torchrun --nproc-per-node=$NPROC_PER_NODE \
    llava/eval/eval_audio_bench.py \
    --model-base $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --task $TASK \
    --think-mode $THINK_MODE \
    --generation-config "$GENERATION_CONFIG" \
    --output-dir $OUTPUT_DIR
