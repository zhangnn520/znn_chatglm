#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup python /root/autodl-tmp/chatglm_efficient_tune/src/finetune.py \
  --do_predict true \
  --model_name_or_path /root/autodl-fs/chatglm-6b_model\
  --dataset dev_examples \
  --dataset_dir /root/autodl-tmp/chatglm_efficient_tune/data/my_data \
  --checkpoint_dir /root/autodl-tmp/chatglm_efficient_tune/output_finetune/checkpoint-2000 \
  --output_dir /root/autodl-tmp/chatglm_efficient_tune/output_finetune/checkpoint-2000/eval \
  --per_device_eval_batch_size 8 \
  --max_eval_samples 4000 \
  --predict_with_generate