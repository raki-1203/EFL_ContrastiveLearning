#!/bin/bash

# temperature=0.05
# cl_weight=0

# efl + scl
for lr in 0.00001 0.00005
do
for cl_weight in 0.6 0.3 0.1
do
for temperature in 0.25 0.5
do
python3 train.py \
  --is_train \
  --task sentiment \
  --use_amp \
  --write_summary \
  --device 1 \
  --method efl_scl \
  --output_path ./model/saved_model/sentiment_model_ver5 \
  --model_name_or_path ./model/checkpoint-2000000 \
  --vocab_path ./tokenizer/version_1.9 \
  --path_to_train_data ./data/preprocessed/sentiment_train_ver5_hate_speech.csv \
  --path_to_valid_data ./data/preprocessed/sentiment_valid_ver5.csv \
  --max_len 256 \
  --batch_size 128 \
  --lr ${lr} \
  --weight_decay 0.1 \
  --cl_weight ${cl_weight} \
  --epochs 10 \
  --pooler_option cls \
  --eval_steps 100 \
  --tensorboard_dir tensorboard_logs/sentiment_model_ver5 \
  --warmup_ratio 0.05 \
  --temperature ${temperature} \
  --trial 0 \
  --seed 42
done
done
done


# efl + rdrop
#python3 train.py \
#  --is_train \
#  --task sentiment \
#  --use_amp \
#  --write_summary \
#  --device 1 \
#  --method efl_rdrop \
#  --output_path ./model/saved_model/sentiment_model_ver5 \
#  --model_name_or_path ./model/checkpoint-2000000 \
#  --vocab_path ./tokenizer/version_1.9 \
#  --path_to_train_data ./data/preprocessed/sentiment_train_ver5.csv \
#  --path_to_valid_data ./data/preprocessed/sentiment_valid_ver5.csv \
#  --max_len 256 \
#  --batch_size 32 \
#  --lr 1e-5 \
#  --weight_decay 0.1 \
#  --epochs 10 \
#  --pooler_option cls \
#  --eval_steps 100 \
#  --tensorboard_dir tensorboard_logs/sentiment_model_ver5 \
#  --warmup_ratio 0.05 \
#  --rdrop_coef 3 \
#  --trial 0 \
#  --seed 42
