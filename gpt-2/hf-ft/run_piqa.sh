#!/bin/bash

python run_piqa.py \
  --model_name_or_path ../../gpt2 \
  --dataset_name ../datasets/piqa \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --save_steps 1000 \
  --output_dir ./tmp/piqa-default 