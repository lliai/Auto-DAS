#!/bin/bash
CURRENT_TIME=`date "+%Y_%m_%d"`
SAVE_DIR=./output/rank/kds

mkdir -p ${SAVE_DIR}
echo "Current time: $CURRENT_TIME"

# kd, nst, cc, ickd, rkd, at, fitnet, sp
# c100

# kd
CUDA_VISIBLE_DEVICES=1 python exps/rank/vit/eval_rank_kds.py --dataset 'c100' --api_path './data/diswotv2_autoformer.pth' --data_path './data' --kd_loss kd > ${SAVE_DIR}/eval_c100_rank_kd_autoformer_${CURRENT_TIME}.txt 2>&1

# nst
CUDA_VISIBLE_DEVICES=1 python exps/rank/vit/eval_rank_kds.py --dataset 'c100' --api_path './data/diswotv2_autoformer.pth' --data_path './data' --kd_loss nst > ${SAVE_DIR}/eval_c100_rank_nst_autoformer_${CURRENT_TIME}.txt 2>&1

# cc
CUDA_VISIBLE_DEVICES=1 python exps/rank/vit/eval_rank_kds.py --dataset 'c100' --api_path './data/diswotv2_autoformer.pth' --data_path './data' --kd_loss cc > ${SAVE_DIR}/eval_c100_rank_cc_autoformer_${CURRENT_TIME}.txt 2>&1

# ickd
CUDA_VISIBLE_DEVICES=1 python exps/rank/vit/eval_rank_kds.py --dataset 'c100' --api_path './data/diswotv2_autoformer.pth' --data_path './data' --kd_loss ickd > ${SAVE_DIR}/eval_c100_rank_ickd_autoformer_${CURRENT_TIME}.txt 2>&1

# rkd
CUDA_VISIBLE_DEVICES=1 python exps/rank/vit/eval_rank_kds.py --dataset 'c100' --api_path './data/diswotv2_autoformer.pth' --data_path './data' --kd_loss rkd > ${SAVE_DIR}/eval_c100_rank_rkd_autoformer_${CURRENT_TIME}.txt 2>&1

# at
CUDA_VISIBLE_DEVICES=1 python exps/rank/vit/eval_rank_kds.py --dataset 'c100' --api_path './data/diswotv2_autoformer.pth' --data_path './data' --kd_loss at > ${SAVE_DIR}/eval_c100_rank_at_autoformer_${CURRENT_TIME}.txt 2>&1

# fitnet
CUDA_VISIBLE_DEVICES=1 python exps/rank/vit/eval_rank_kds.py --dataset 'c100' --api_path './data/diswotv2_autoformer.pth' --data_path './data' --kd_loss fitnet > ${SAVE_DIR}/eval_c100_rank_fitnet_autoformer_${CURRENT_TIME}.txt 2>&1

# sp
CUDA_VISIBLE_DEVICES=1 python exps/rank/vit/eval_rank_kds.py --dataset 'c100' --api_path './data/diswotv2_autoformer.pth' --data_path './data' --kd_loss
