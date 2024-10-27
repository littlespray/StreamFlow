

TIME=$(date "+%Y%m%d-%H%M%S")
DATA_PATH=/userhome
SPRING_ROOT=${DATA_PATH}/spring
MULTI_ROOT=${DATA_PATH}/multi_kitti
BASE_DIR="results/spring_all-nomix-lr4-18w"
MODEL=SKFlow_MF8

mkdir -p $BASE_DIR

SPRING_SAVE_DIR=${BASE_DIR}

# 1080 * 1920
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_mf.py \
--name spring \
--stage spring \
--validation kitti \
--num_steps 180000 \
--image_size 432 768 \
--lr 0.0004 \
--output ${SPRING_SAVE_DIR} \
--wdecay 0.00001 \
--gamma 0.85 \
--gpus 0 1 2 3 \
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
--spring_root ${SPRING_ROOT} \
--multi_root ${MULTI_ROOT} \
--nofreeze_untemporal \
--iters 12 \
--seed 3407 \
--restore_ckpt streamflow_t4-sintel.pth