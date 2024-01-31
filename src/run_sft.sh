#! /bin/bash

#export CUDA_VISIBLE_DEVICES='0,1'
export WANDB_PROJECT=EmoLLMs_train
export WANDB_RUN_ID=31012024_1
export WANDB_RESUME=allow
export ABS_PATH="local_path"

model_name_or_path=meta-llama/Llama-2-7b-chat-hf # model_path

train_file=data/train.json
validation_file=data/dev.json
output_dir="$ABS_PATH/saved_models/${WANDB_PROJECT}_${WANDB_RUN_ID}"
mkdir -p ${output_dir}

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=2048

#FT
torchrun --nproc_per_node 2 src/sft_train.py \
     --ddp_timeout 36000 \
     --model_name_or_path ${model_name_or_path} \
     --deepspeed configs/deepspeed_config_stage3.json \
     --train_file ${train_file} \
     --validation_file ${validation_file} \
     --per_device_train_batch_size 128 \
     --per_device_eval_batch_size 128 \
     --gradient_accumulation_steps 1 \
     --num_train_epochs 3 \
     --model_max_length ${cutoff_len} \
     --save_strategy "steps" \
     --save_total_limit 3 \
     --learning_rate 1e-6 \
     --weight_decay 0.00001 \
     --warmup_ratio 0.03 \
     --lr_scheduler_type "cosine" \
     --logging_steps 5 \
     --evaluation_strategy "steps" \
     --torch_dtype "bfloat16" \
     --bf16 \
     --seed 123 \
     --gradient_checkpointing \
     --cache_dir ${cache_dir} \
     --output_dir ${output_dir} \
     --llama \
    # --use_flash_attention
    # --resume_from_checkpoint ...