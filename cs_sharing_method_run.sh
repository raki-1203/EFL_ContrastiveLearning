#!/bin/bash

for method in "efl" "efl_scl" "std" "std_scl" "efl_rdrop"
do
  python3 train.py \
  --is_train \
  --task sentiment \
  --use_amp \
  --write_summary \
  --device 1 \
  --method ${method} \
  --output_path ./model/saved_model/${method} \
  --model_name_or_path ./model/checkpoint-2000000 \
  --vocab_path ./tokenizer/version_1.9 \
  --path_to_train_data ./data/preprocessed/sentiment_train_ver6.csv \
  --path_to_valid_data ./data/preprocessed/sentiment_valid_ver6.csv \
  --max_len 256 \
  --batch_size 100 \
  --lr 1e-5 \
  --weight_decay 0.1 \
  --cl_weight 0.6 \
  --epochs 10 \
  --pooler_option cls \
  --eval_steps 100 \
  --tensorboard_dir tensorboard_logs/${method} \
  --warmup_ratio 0.05 \
  --temperature 0.25 \
  --trial 0 \
  --seed 42 \
  --lr_scheduler_type ReduceLROnPlateau
done