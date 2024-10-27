#!/usr/bin/env bash
SPRING_SAVE_DIR="results/spring_all-nomix-lr4-18w"
PYTHON=/userhome/py22/bin/python

DATAROOT=/userhome
CHAIRS_ROOT=${DATAROOT}/FlyingChairs_release/data
SINTEL_ROOT=${DATAROOT}/sintel
KITTI_ROOT=${DATAROOT}/KITTI
MULTI_ROOT=${DATAROOT}/multi-kitti
SPRING_ROOT=${DATAROOT}/spring

MODEL=SKFlow_MF8

for x in {180000..180000..5000}
do
    echo ${x} | tee -a ${SPRING_SAVE_DIR}/infer_log.txt
    CUDA_VISIBLE_DEVICES=0 \
    ${PYTHON} evaluate_mf.py \
    --dataset spring \
    --iters 15 \
    --model_name ${MODEL} \
    --spring_root ${SPRING_ROOT} \
    --model ${SPRING_SAVE_DIR}/${x}_spring.pth \
    --use_gma \
    --k_conv 1 15 \
    --Encoder Twins_CSC \
    --MotionEncoder SKMotionEncoder6_Deep_nopool_res \
    --UpdateBlock SKUpdateBlock_TAM_v3 \
    --T 4 \
    | tee -a ${SPRING_SAVE_DIR}/infer_log.txt
done

# --use_temporal_decoder \
# --use_temporal_decoder \

# --UpdateBlock MFSKIIUpdateBlock \
# --Encoder TWINS_3D \ MFBasicEncoder

