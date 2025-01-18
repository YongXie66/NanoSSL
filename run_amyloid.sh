#!/bin/bash
for i in 0
do
python test.py \
    --save_path exp/amyloid/test \
    --data_path data/amyloid/non_norm/ \
    --dataset amyloid \
    --mutation S26PO4 \
    --mask_ratio 0.6 \
    --wave_length 12 \
    --layers 8 \
    --d_model 64 \
    --num_epoch_pretrain 300 \
    --num_epoch 300 \
    --lr 0.001 \
    --dropout 0.2 \
    --train_batch_size 128 \
    --load_pretrained_model 1 \
    --cv_split $i \
    --train_ratio 1 \
    --device cuda:1 \
    --model_id 76
done