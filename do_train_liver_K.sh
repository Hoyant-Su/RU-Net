#!/bin/sh

#redirect to your own workspace
export PYTHONPATH='/RU-net'

### RU-net train
python train_concat.py -c exp_configs.yaml \
--data_dir datasets/Data/Image \
--batch-size 32 \
--model uniformer_base_IL \
--lr 1e-4 --warmup-epochs 20 --epochs 700 \
--img_size 20 160 160 \
--crop_size 15 150 150 \
--initial-checkpoint ./weights/uniformer_base_k400_8x8_partial.pth \

