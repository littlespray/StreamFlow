#!/usr/bin/env bash

TIME=$(date "+%Y%m%d-%H%M%S")
DATA_PATH=/dataset/OpticalFlow
BASE_DIR=/StreamFlowResults/SKUpdateBlock_TAM_v3_T4_lr175_18w_8p_nomix
MODEL=SKFlow_MF8


mkdir -p $BASE_DIR

CHAIRS_ROOT=${DATA_PATH}/FlyingChairs_release/data
THINGS_ROOT=${DATA_PATH}/flyingthings3d
SINTEL_ROOT=${DATA_PATH}/sintel
KITTI_ROOT=${DATA_PATH}/KITTI
HD1K_ROOT=${DATA_PATH}/HD1K
MULTI_ROOT=${DATA_PATH}/multi-kitti
CHAIRS_SAVE_DIR=${BASE_DIR}/chairs
THINGS_SAVE_DIR=${BASE_DIR}/things 
SINTEL_SAVE_DIR=${BASE_DIR}/sintel 
KITTI_SAVE_DIR=${BASE_DIR}/kitti

echo "============================================================"
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

# pack code
if [ ! -d ${THINGS_SAVE_DIR} ]; then
    mkdir -p ${THINGS_SAVE_DIR}
fi



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_mf.py \
--name sintel \
--stage sintel \
--validation kitti \
--num_steps 180000 \
--image_size 432 960 \
--lr 0.000175 \
--output ${SINTEL_SAVE_DIR} \
--wdecay 0.00001 \
--gamma 0.85 \
--gpus 0 1 2 3 4 5 6 7 \
--batch_size 8 \
--T 4 \
--val_freq 5000 \
--print_freq 100 \
--model_name ${MODEL} \
--use_gma \
--k_conv 1 15 \
--Encoder Twins_CSC \
--MotionEncoder SKMotionEncoder6_Deep_nopool_res \
--UpdateBlock SKUpdateBlock_TAM_v3 \
--chairs_root ${CHAIRS_ROOT} \
--things_root ${THINGS_ROOT} \
--sintel_root ${SINTEL_ROOT} \
--kitti_root ${KITTI_ROOT} \
--hd1k_root ${HD1K_ROOT} \
--multi_root ${MULTI_ROOT} \
--nofreeze_untemporal \
--iters 12 \
--seed 3407 \
--restore_ckpt ${THINGS_CKPT} \

