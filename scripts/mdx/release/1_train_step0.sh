#!/bin/bash

source venv/bin/activate

bs=4
acc_step=8

BASE_MODEL_PATH="llm-jp/llm-jp-3-13b-instruct"
OUTPUT="llm-jp-3-13b-instruct_siglip_mlp2xgelu_step-0"

MNAME=$(echo $BASE_MODEL_PATH | rev | cut -d "/" -f 1 | rev)

deepspeed --num_gpus=8 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $BASE_MODEL_PATH \
    --version plain \
    --data_mixture llava_1_5_mm_align_en+llm_jp_mm_pair_step0_558k \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp2x_gelu \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model False \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir ./checkpoints/$OUTPUT \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $acc_step \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb
