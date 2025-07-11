#!/bin/bash
export NCCL_DEBUG=WARN
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

DEFAULT_RUN_NAME="stage2_af3"

STAGE_PATH=${1:-"runs/train/stage1_af3/checkpoint-xxxx"}

# data_mixture_1 is the entry of the dataset in llava/data/datasets_mixture.py.
DATA_MIXTURE=${2:-"data_mixture_1+data_mixture_2"}


if [ "$NNODES" = "1" ] || [ "$NNODES" = "2" ]; then
    echo "Detected on single machine. Automatically set batch size to 1 for debugging purpose."
    PER_DEVICE_TRAIN_BATCH_SIZE=1
fi
    
    
torchrun --nnodes \$NUM_NODES --nproc_per_node \$SUBMIT_GPUS --master_addr \$MASTER_ADDR --master_port \$MASTER_PORT --node_rank \$NODE_RANK \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3_gradient_clipping.json \
    --model_name_or_path $STAGE_PATH \
    --chat_template qwen2 \
    --data_mixture $DATA_MIXTURE \
    --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
    --dynamic_s2 True \
    --s2_scales "448,896,1344" \
    --s2_max_split_size 448 \
    --s2_resize_output_to_scale_idx -1 \
    --speech_tower openai/whisper-large-v2 \
    --sound_tower /path/to/AF-Whisper \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --speech_mm_projector mlp \
    --sound_mm_projector mlp \
    --tune_vision_tower False \
    --tune_speech_tower False \
    --tune_sound_tower True \
    --tune_mm_projector False \
    --tune_speech_mm_projector False \
    --tune_sound_mm_projector True \
    --tune_language_model False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio dynamic_s2 \
    --bf16 True \
    --audio_frames 1 \
    --output_dir runs/train/stage2_af3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 1.5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to wandb

