#!/bin/bash
# Conduct series experiments for different search space structure

CURRENT_TIME=`date "+%Y_%m_%d"`
DATA_PATH='./data'
SAVE_DIR=./output/search_model/s0
SEED="555"

mkdir -p ${SAVE_DIR}
echo "Current time: $CURRENT_TIME"

# unconstrainted
PARAM_CONSTRAIN=100
CUDA_VISIBLE_DEVICES=1 python exps/search_model/s0/rnd_search_model.py --param_constraint ${PARAM_CONSTRAIN} 2>&1 | tee -a ${SAVE_DIR}/s0_rnd_search_jointly_${PARAM_CONSTRAIN}_${SEED}_${CURRENT_TIME}.log

# 0.5 or 1 or 2M
PARAM_CONSTRAIN=0.5
CUDA_VISIBLE_DEVICES=0 python exps/search_model/s0/rnd_search_model.py --param_constraint ${PARAM_CONSTRAIN} 2>&1 | tee -a ${SAVE_DIR}/s0_rnd_search_jointly_${PARAM_CONSTRAIN}_${SEED}_${CURRENT_TIME}.log

PARAM_CONSTRAIN=1
CUDA_VISIBLE_DEVICES=1 python exps/search_model/s0/rnd_search_model.py --param_constraint ${PARAM_CONSTRAIN} 2>&1 | tee -a ${SAVE_DIR}/s0_rnd_search_jointly_${PARAM_CONSTRAIN}_${SEED}_${CURRENT_TIME}.log

PARAM_CONSTRAIN=2
CUDA_VISIBLE_DEVICES=2 python exps/search_model/s0/rnd_search_model.py --param_constraint ${PARAM_CONSTRAIN} 2>&1 | tee -a ${SAVE_DIR}/s0_rnd_search_jointly_${PARAM_CONSTRAIN}_${SEED}_${CURRENT_TIME}.log


# cost time
echo "Cost time: $SECONDS seconds"
