#!/bin/bash
CURRENT_TIME=`date "+%Y_%m_%d"`
SAVE_DIR=./output/rank/diswot_vit

mkdir -p ${SAVE_DIR}
echo "Current time: $CURRENT_TIME"

# c100, flowers, chaoyang
# autoformer, pit

# c100, autoformer
CUDA_VISIBLE_DEVICES=2 python exps/rank/vit/eval_rank_diswot.py --dataset 'c100' --api_path './data/diswotv2_autoformer.pth' --data_path './data' > ${SAVE_DIR}/eval_c100_rank_diswot_autoformer_${CURRENT_TIME}.txt 2>&1

# c100, pit
CUDA_VISIBLE_DEVICES=2 python exps/rank/vit/eval_rank_diswot.py --dataset 'c100' --api_path './data/diswotv2_pit.pth' --data_path './data' > ${SAVE_DIR}/eval_c100_rank_diswot_pit_${CURRENT_TIME}.txt 2>&1

# flowers, autoformer
CUDA_VISIBLE_DEVICES=2 python exps/rank/vit/eval_rank_diswot.py --dataset 'flowers' --api_path './data/diswotv2_autoformer.pth' --data_path './data/flowers' > ${SAVE_DIR}/eval_flowers_rank_diswot_autoformer_${CURRENT_TIME}.txt 2>&1

# flowers, pit
CUDA_VISIBLE_DEVICES=2 python exps/rank/vit/eval_rank_diswot.py --dataset 'flowers' --api_path './data/diswotv2_pit.pth' --data_path './data/flowers' > ${SAVE_DIR}/eval_flowers_rank_diswot_pit_${CURRENT_TIME}.txt 2>&1

# chaoyang, autoformer
CUDA_VISIBLE_DEVICES=2 python exps/rank/vit/eval_rank_diswot.py --dataset 'chaoyang' --api_path './data/diswotv2_autoformer.pth' --data_path './data/chaoyang' > ${SAVE_DIR}/eval_chaoyang_rank_diswot_autoformer_${CURRENT_TIME}.txt 2>&1


# chaoyang, pit
CUDA_VISIBLE_DEVICES=2 python exps/rank/vit/eval_rank_diswot.py --dataset 'chaoyang' --api_path './data/diswotv2_pit.pth' --data_path './data/chaoyang' > ${SAVE_DIR}/eval_chaoyang_rank_diswot_pit_${CURRENT_TIME}.txt 2>&1
