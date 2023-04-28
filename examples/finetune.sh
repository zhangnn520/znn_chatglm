#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python /root/autodl-tmp/chatglm_efficient_tune/src/finetune.py \
    --model_name_or_path /root/autodl-fs/chatglm-6b_model\
    --do_train \
    --do_eval \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --metric_for_best_model "eval_loss" \
    --dataset dev_examples,train_examples \
    --dataset_dir /root/autodl-tmp/chatglm_efficient_tune/data/my_data \
    --finetuning_type p_tuning \
    --prefix_projection \
    --output_dir ../output_finetune \
    --overwrite_cache \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 6 \
    --lr_scheduler_type cosine \
    --logging_steps 200 \
    --save_steps 200 \
    --max_train_samples 2000 \
    --learning_rate 2e-5 \
    --num_train_epochs 6.0 \
    --fp16


