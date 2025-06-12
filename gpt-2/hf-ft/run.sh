#!/bin/bash

python run_clm.py \
    --model_name_or_path ../../gpt2-medium \
    --dataset_name ../datasets/wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --save_steps 2000 \
    --output_dir ./tmp/test-clm-medium

# torchrun --nproc_per_node=4 run_clm.py \
#     --model_name_or_path ../../gpt2-medium \
#     --dataset_name ../datasets/wikitext-2-raw-v1 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 3 \
#     --do_train \
#     --do_eval \
#     --num_train_epochs 3 \
#     --save_steps 2000 \
#     --output_dir ./tmp/test-clm-xwwx-medium-60