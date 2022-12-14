#!/bin/bash

# temperature=0.05
# cl_weight=0

for lr in 0.00001 0.00005
do
for cl_weight in 0.6 0.3 0.1
do
for temperature in 0.25 0.5
do
python3 train.py \
  --is_train \
  --task category \
  --use_amp \
  --write_summary \
  --device 1 \
  --method efl_scl \
  --output_path ./model/saved_model/category_model_ver5_ai_hub \
  --model_name_or_path ./model/checkpoint-2000000 \
  --vocab_path ./tokenizer/version_1.9 \
  --saved_model_state_path ./model/saved_model/sentiment_model_ver6/STEP_900_efl_scl_TASKsentiment_LR1e-05_WD0.1_LAMBDA0.6_POOLERcls_TEMP0.5_ACC0.8575 \
  --path_to_train_data ./data/cs_sharing/train_ver5_ai_hub \
  --path_to_valid_data ./data/cs_sharing/valid_ver5 \
  --max_len 256 \
  --batch_size 16 \
  --lr ${lr} \
  --weight_decay 0.1 \
  --cl_weight ${cl_weight} \
  --epochs 10 \
  --pooler_option cls \
  --eval_steps 100 \
  --tensorboard_dir tensorboard_logs/category_model_ver5_ai_hub \
  --warmup_ratio 0.05 \
  --temperature ${temperature} \
  --trial 0 \
  --seed 42
done
done
done

#for lr in 0.00001 0.00005
#do
#for cl_weight in 0.6 0.3 0.1
#do
#for temperature in 0.25 0.5
#do
#python3 train.py \
#  --is_train \
#  --task category \
#  --use_amp \
#  --write_summary \
#  --device 1 \
#  --method efl_scl \
#  --output_path ./model/saved_model/all_category_model_ver4 \
#  --model_name_or_path ./model/checkpoint-2000000 \
#  --vocab_path ./tokenizer/version_1.9 \
#  --path_to_train_data ./data/cs_sharing/train_ver4 \
#  --path_to_valid_data ./data/cs_sharing/valid_ver4 \
#  --max_len 256 \
#  --batch_size 32 \
#  --lr ${lr} \
#  --weight_decay 0.1 \
#  --cl_weight ${cl_weight} \
#  --epochs 10 \
#  --pooler_option cls \
#  --eval_steps 100 \
#  --tensorboard_dir tensorboard_logs/all_category_model_ver4 \
#  --warmup_ratio 0.05 \
#  --temperature ${temperature} \
#  --trial 0 \
#  --seed 42
#done
#done
#done


#for lr in 0.00001 0.00005
#do
#for cl_weight in 0.9 0.6 0.3 0.1
#do
#for temperature in 0.05 0.10 0.15 0.20 0.25 0.5
#do
#python3 train.py \
#  --is_train \
#  --task category \
#  --use_amp \
#  --write_summary \
#  --device 1 \
#  --method efl_scl \
#  --output_path ./model/saved_model/all_category_model_ver2 \
#  --model_name_or_path ./model/checkpoint-2000000 \
#  --vocab_path ./tokenizer/version_1.9 \
#  --path_to_train_data ./data/cs_sharing/train_ver2 \
#  --path_to_valid_data ./data/cs_sharing/valid_ver2 \
#  --max_len 256 \
#  --batch_size 32 \
#  --lr ${lr} \
#  --weight_decay 0.1 \
#  --cl_weight ${cl_weight} \
#  --epochs 10 \
#  --pooler_option cls \
#  --eval_steps 100 \
#  --tensorboard_dir tensorboard_logs/all_category_model_ver2 \
#  --warmup_ratio 0.05 \
#  --temperature ${temperature} \
#  --trial 0 \
#  --seed 42
#done
#done
#done


#for lr in 0.00001 0.00005
#do
#for cl_weight in 0.9 0.6 0.3 0.1
#do
#for temperature in 0.05 0.10 0.15 0.20 0.25 0.5
#do
#python3 train.py \
#  --is_train \
#  --task category \
#  --use_amp \
#  --write_summary \
#  --device 1 \
#  --method efl_scl \
#  --output_path ./model/saved_model/all_category_model_ver3 \
#  --model_name_or_path ./model/checkpoint-2000000 \
#  --vocab_path ./tokenizer/version_1.9 \
#  --path_to_train_data ./data/cs_sharing/train_ver3 \
#  --path_to_valid_data ./data/cs_sharing/valid_ver3 \
#  --max_len 256 \
#  --batch_size 32 \
#  --lr ${lr} \
#  --weight_decay 0.1 \
#  --cl_weight ${cl_weight} \
#  --epochs 10 \
#  --pooler_option cls \
#  --eval_steps 100 \
#  --tensorboard_dir tensorboard_logs/all_category_model_ver3 \
#  --warmup_ratio 0.05 \
#  --temperature ${temperature} \
#  --trial 0 \
#  --seed 42
#done
#done
#done
