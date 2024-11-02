#!/bin/sh

#redirect to your own workspace
export PYTHONPATH='/RU-net'

### RU-net test
python validate_concat.py -c exp_configs.yaml \
--data_dir datasets/Data/Image \
--batch-size 2 \
--model uniformer_base_IL \
--img_size 20 160 160 \
--crop_size 15 150 150 \

