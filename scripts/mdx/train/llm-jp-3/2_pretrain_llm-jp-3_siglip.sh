#!/bin/bash

#SBATCH --job-name=0043_vila_step1       # name
#SBATCH --nodes=8                        # nodes
#SBATCH --ntasks-per-node=1              # crucial - only 1 task per dist per node!
#SBATCH --gres=gpu:8                     # number of gpus
#SBATCH --time 96:00:00                  # maximum execution time (HH:MM:SS)
#SBATCH --output=step_1_20240928.out     # output file name
#SBATCH --error=step_1_20240928.err
#SBATCH --partition=gpu

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901


# export CURRENT_RANK=${SLURM_PROCID:-"0"}
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')

echo "MASTER_ADDR="$MASTER_ADDR
echo "JobID: $SLURM_JOB_ID | Full list: $worker_list"

# OUTPUT of stage 1 script
export STAGE1_PATH="./checkpoints/llm-jp-3-13b-instruct_siglip_mlp2xgelu_step-0_20240927"
# for example, llava-v1.5-7b-mm-align
OUTPUT="llm-jp-3-13b-instruct_siglip_mlp2xgelu_step-1_20240928"
export OUTPUT_PATH=/model/experiments/0043_vila_step1/checkpoints/$OUTPUT

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
#export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
#export CUDA_LAUNCH_BLOCKING=1

n_node=$SLURM_NNODES
export acc_step=8
export bs=$((128 / n_node / acc_step))
echo "number of nodes:" $n_node
echo "gradient accumulation steps:" $acc_step
echo "per device batch size:" $bs
# echo "node rank:" $SLURM_PROCID

source venv/bin/activate

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $STAGE1_PATH \
    --version llmjp_v3 \
    --data_mixture coyo_6m+mmc4core_6m+llm_jp_mm_pair_step1_6m+llm_jp_mm_interleaved_step1_6m \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp2x_gelu \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $acc_step \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb'
