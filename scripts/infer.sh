#!/usr/bin/env bash
CKPT_PATH='ckpts/streamflow-sintel.pth'
PYTHON=python

DATAROOT=/shangkunsun
CHAIRS_ROOT=${DATAROOT}/FlyingChairs_release/data
SINTEL_ROOT=${DATAROOT}/sintel
KITTI_ROOT=${DATAROOT}/KITTI
MULTI_ROOT=${DATAROOT}/multi-kitti
SPRING_ROOT=${DATAROOT}/spring

MODEL=SKFlow_MF8

CUDA_VISIBLE_DEVICES=0 \
${PYTHON} evaluate_mf.py \
--dataset sintel \
--iters 15 \
--model_name ${MODEL} \
--sintel_root ${SINTEL_ROOT} \
--model ${CKPT_PATH} \
--use_gma \
--k_conv 1 15 \
--Encoder Twins_CSC \
--MotionEncoder SKMotionEncoder6_Deep_nopool_res \
--UpdateBlock SKUpdateBlock_TAM_v3 \
--T 4


