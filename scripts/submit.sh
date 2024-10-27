#!/usr/bin/env bash
CHAIRS_SAVE_DIR=results/chairs/debug
THINGS_SAVE_DIR=results/things/debug
SINTEL_SAVE_DIR=results/sintel/debug
KITTI_SAVE_DIR=results/kitti/debug
PYTHON=python

DATAROOT=/mnt/apdcephfs_sh3/share_301074934/shangkunsun/OpticalFlowDataset
CHAIRS_ROOT=${DATAROOT}/FlyingChairs_release/data
SINTEL_ROOT=${DATAROOT}/sintel
KITTI_ROOT=${DATAROOT}/KITTI
MULTI_ROOT=${DATAROOT}/multi-kitti

MODEL=SKFlow_MF8
CKPT=/shangkunsun/StreamFlowResults/highres-ft-t4-8p-mix/sintel/50000_sintel.pth
# CKPT=/StreamFlowResults/SKUpdateBlock_TAM_v3_T3_oldckpt3407_lr175_18w/sintel/167500_sintel.pth
# CKPT=/StreamFlowResults/SKUpdateBlock_TAM_v3_T3_newckpt_lr175_15w/sintel/150000_sintel.pth
# CKPT=/share/sunshangkun/twins-skflow-sintel.pth

# CKPT=/share/sunshangkun/twins-skii-things.pth
# CKPT=/share/sunshangkun/ablations/gma_sc_145w.pth
# CKPT=/share/sunshangkun/ablations/gma_sc_145w.pth
# CKPT=/share/sunshangkun/gma-things.pth
# CKPT=/share/sunshangkun/ablation_gma.pth
# CKPT=/share/sunshangkun/skflow-things.pth
# CKPT=/share/sunshangkun/twins-skii-180000_gma-sintel.pth
# CKPT=/share/sunshangkun/twins-skii-kitti.pth
# CKPT=/share/sunshangkun/
# WORSTCASE_ROOT=/share/sunshangkun/worst_case
# CKPT=/share/sunshangkun/ft_0418_things.pth
# CKPT=/share/sunshangkun/ft0418_fromthings_sintelsubmit.pth
# CKPT=


CUDA_VISIBLE_DEVICES=0 \
${PYTHON} submit_mf.py \
--dataset all \
--iters 20 \
--model_name ${MODEL} \
--chairs_root ${CHAIRS_ROOT} \
--sintel_root ${SINTEL_ROOT} \
--kitti_root ${KITTI_ROOT} \
--model ${CKPT} \
--use_gma \
--k_conv 1 15 \
--Encoder Twins_CSC \
--MotionEncoder SKMotionEncoder6_Deep_nopool_res \
--UpdateBlock SKUpdateBlock_TAM_v3 \
--T 4 \
--multi_root ${MULTI_ROOT} \
--output_path T4_ft_iters20 \

# --UpdateBlock SKFlow_TMM \


# --use_temporal_decoder \

# --UpdateBlock MFSKIIUpdateBlock \
# --Encoder TWINS_3D \ MFBasicEncoder

