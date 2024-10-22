#!/bin/bash

CURRENT_TIME=`date "+%Y_%m_%d"`
SAVE_DIR=./output/rank/diswotv2_jointly

mkdir -p ${SAVE_DIR}
echo "Current time: $CURRENT_TIME"


#### NB101-KD ####

# SEED=111
# CUDA_VISIBLE_DEVICES=1 python exps/rank/nb101/eval_rank_jointly.py --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_jointly_${SEED}_${CURRENT_TIME}.txt 2>&1

# SEED=222
# CUDA_VISIBLE_DEVICES=1 python exps/rank/nb101/eval_rank_jointly.py --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_jointly_${SEED}_${CURRENT_TIME}.txt 2>&1

# SEED=333
# CUDA_VISIBLE_DEVICES=1 python exps/rank/nb101/eval_rank_jointly.py --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_jointly_${SEED}_${CURRENT_TIME}.txt 2>&1

# SEED=444
# CUDA_VISIBLE_DEVICES=1 python exps/rank/nb101/eval_rank_jointly.py --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_jointly_${SEED}_${CURRENT_TIME}.txt 2>&1

# #### NB201-KD ####
# SEED=111
# CUDA_VISIBLE_DEVICES=2 python exps/rank/nb201/eval_rank_jointly.py --seed ${SEED}  > ${SAVE_DIR}/eval_NB201_rank_jointly_${SEED}_${CURRENT_TIME}.txt 2>&1

# SEED=222
# CUDA_VISIBLE_DEVICES=2 python exps/rank/nb201/eval_rank_jointly.py --seed ${SEED}  > ${SAVE_DIR}/eval_NB201_rank_jointly_${SEED}_${CURRENT_TIME}.txt 2>&1

# SEED=333
# CUDA_VISIBLE_DEVICES=2 python exps/rank/nb201/eval_rank_jointly.py --seed ${SEED}  > ${SAVE_DIR}/eval_NB201_rank_jointly_${SEED}_${CURRENT_TIME}.txt 2>&1

# SEED=444
# CUDA_VISIBLE_DEVICES=2 python exps/rank/nb201/eval_rank_jointly.py --seed ${SEED}  > ${SAVE_DIR}/eval_NB201_rank_jointly_${SEED}_${CURRENT_TIME}.txt 2>&1


# c100, flowers, chaoyang
# autoformer, pit

# c100, autoformer
CUDA_VISIBLE_DEVICES=2 python exps/rank/vit/eval_rank_jointly.py --dataset 'c100' --api_path './data/diswotv2_autoformer.pth' --data_path './data' > ${SAVE_DIR}/eval_c100_rank_diswotv2_autoformer_${CURRENT_TIME}.txt 2>&1

# c100, pit
CUDA_VISIBLE_DEVICES=2 python exps/rank/vit/eval_rank_jointly.py --dataset 'c100' --api_path './data/diswotv2_pit.pth' --data_path './data' > ${SAVE_DIR}/eval_c100_rank_diswotv2_pit_${CURRENT_TIME}.txt 2>&1

# flowers, autoformer
CUDA_VISIBLE_DEVICES=2 python exps/rank/vit/eval_rank_jointly.py --dataset 'flowers' --api_path './data/diswotv2_autoformer.pth' --data_path './data/flowers' > ${SAVE_DIR}/eval_flowers_rank_diswotv2_autoformer_${CURRENT_TIME}.txt 2>&1

# flowers, pit
CUDA_VISIBLE_DEVICES=2 python exps/rank/vit/eval_rank_jointly.py --dataset 'flowers' --api_path './data/diswotv2_pit.pth' --data_path './data/flowers' > ${SAVE_DIR}/eval_flowers_rank_diswotv2_pit_${CURRENT_TIME}.txt 2>&1

# chaoyang, autoformer
CUDA_VISIBLE_DEVICES=2 python exps/rank/vit/eval_rank_jointly.py --dataset 'chaoyang' --api_path './data/diswotv2_autoformer.pth' --data_path './data/chaoyang' > ${SAVE_DIR}/eval_chaoyang_rank_diswotv2_autoformer_${CURRENT_TIME}.txt 2>&1

# chaoyang, pit
CUDA_VISIBLE_DEVICES=2 python exps/rank/vit/eval_rank_jointly.py --dataset 'chaoyang' --api_path './data/diswotv2_pit.pth' --data_path './data/chaoyang' > ${SAVE_DIR}/eval_chaoyang_rank_diswotv2_pit_${CURRENT_TIME}.txt 2>&1
