#!/bin/bash
python train.py \
  --mask_ratio 0.6 \
  --num_epoch_pretrain 300 \
  --num_epoch 300 \
  --lr 0.001 \
  --load_pretrained_model 1 \
  --save_path exp/ONT/test \
  --data_path dataset/ont/ \
  --device cuda:2 \
  --model_id 48 