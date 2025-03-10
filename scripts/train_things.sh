#!/usr/bin/env bash

set -e

source ~/.bashrc

WORK_DIR="$( cd "$( dirname "$BASH_SOURCE[0]" )" && pwd )"
cd $WORK_DIR
cd ..
echo $PWD

################################################################################
## 1. GPEN ARGS

if [ ! $TRIAL_NAME ]; then
TRIAL_NAME=sunshangkun_test
fi

if [ ! $GPU_NUM ]; then
GPU_NUM=1
fi

DATA_PATH=/dataset/OpticalFlow
CODE_PATH=/usr/sunshangkun/StreamFlow2/
BASE_DIR=/usr/sunshangkun/StreamFlowResults/lr_6e-4_skflow_mf3_T3_b32_pretrained
MODEL=SKFlow_MF8


mkdir -p $BASE_DIR

CHAIRS_ROOT=${DATA_PATH}/FlyingChairs_release/data
THINGS_ROOT=${DATA_PATH}/flyingthings3d
SINTEL_ROOT=${DATA_PATH}/sintel
KITTI_ROOT=${DATA_PATH}/KITTI
HD1K_ROOT=${DATA_PATH}/HD1K
CHAIRS_SAVE_DIR=${BASE_DIR}/chairs
THINGS_SAVE_DIR=${BASE_DIR}/things 
SINTEL_SAVE_DIR=${BASE_DIR}/sintel 
KITTI_SAVE_DIR=${BASE_DIR}/kitti

echo "============================================================"
echo "CODE_PATH : " $CODE_PATH
echo "DATA_PATH : " $DATA_PATH


################################################################################
## 2. prepare pretrained models

# mkdir -p $BASE_DIR/weights/
# cp $CODE_PATH/pretrained_model/* $BASE_DIR/weights/
# mkdir -p /root/.cache/torch/hub/checkpoints/
# cp $CODE_PATH/pretrained_model/alexnet-owt-4df8aa71.pth \
# /root/.cache/torch/hub/checkpoints/alexnet-owt-4df8aa71.pth
# cp $CODE_PATH/pretrained_model/alexnet-owt-7be5be79.pth \
# /root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth


################################################################################
## 3. gogogo

python train_mf.py \
--name things \
--stage things \
--validation sintel \
--mixed_precision \
--num_steps 300000 \
--image_size 400 720 \
--lr 0.0006 \
--output ${THINGS_SAVE_DIR} \
--wdecay 0.00001 \
--gpus 0 1 2 3 \
--batch_size 16 \
--T 3 \
--val_freq 5000 \
--print_freq 100 \
--model_name ${MODEL} \
--use_gma \
--k_conv 1 15 \
--Encoder Twins_CSC \
--MotionEncoder SKMotionEncoder6_Deep_nopool_res \
--UpdateBlock SKFlowDecoder \
--chairs_root ${CHAIRS_ROOT} \
--things_root ${THINGS_ROOT} \
--sintel_root ${SINTEL_ROOT} \
--kitti_root ${KITTI_ROOT} \
--hd1k_root ${HD1K_ROOT} \
--restore_ckpt pretrained/twins-skflow-things.pth \

