#!/bin/bash

# 定义模型名称数组、数据集名称数组和种子数组
model_names=("roberta-base" "bert-base-uncased" "allenai/scibert_scivocab_uncased")
dataset_names=("restaurant_sup" "acl_sup" "agnews_sup")
seeds=(347913 594729 162850 95842 43288)

# 外层循环：遍历模型名称
for model_name in "${model_names[@]}"; do
  # 中层循环：遍历数据集名称
  for dataset_name in "${dataset_names[@]}"; do
    # 内层循环：遍历种子值
    for seed in "${seeds[@]}"; do
      # 设置输出目录
      output_dir="./output/${model_name}-${dataset_name}-${seed}"
      
      # 检查输出目录是否存在，若存在则跳过当前任务
      if [ -d "$output_dir" ]; then
        echo "Skipping completed task for model: $model_name, dataset: $dataset_name, seed: $seed"
        continue
      fi

      # 运行训练脚本
      python train.py \
        --model_name_or_path="$model_name" \
        --dataset_name="$dataset_name" \
        --max_seq_length=128 \
        --output_dir="./output" \
        --do_train=true \
        --do_eval=true \
        --evaluation_strategy=epoch \
        --learning_rate=2e-5 \
        --per_device_train_batch_size=128 \
        --per_device_eval_batch_size=128 \
        --num_train_epochs=3 \
        --weight_decay=0.01 \
        --report_to=wandb \
        --seed="$seed"
      
      # 删除包含 "checkpoint" 的文件夹和所有 ".safetensors" 文件
      find "$output_dir" -type d -name "*checkpoint*" -exec rm -rf {} +
      find "$output_dir" -type f -name "*.safetensors" -exec rm -f {} +
      
      echo "Completed task for model: $model_name, dataset: $dataset_name, seed: $seed"
    done
  done
done
