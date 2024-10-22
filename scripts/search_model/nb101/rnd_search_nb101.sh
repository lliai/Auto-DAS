#!/bin/bash
# Conduct series experiments for different search space structure

CURRENT_TIME=`date "+%Y_%m_%d"`
DATA_PATH='./data'
SAVE_DIR=./output/search_model/nb101
ITERATIONS=1000
SEEDs="555"

mkdir -p ${SAVE_DIR}
echo "Current time: $CURRENT_TIME"

for SEED in ${SEEDs}
    do
        # modify DEVICE ID here for different GPU
        CUDA_VISIBLE_DEVICES=1 python exps/search_model/nb101/rnd_search_model.py 2>&1 | tee -a ${SAVE_DIR}/nb101_rnd_search_jointly_${SEED}_${CURRENT_TIME}.log
    done

# cost time
echo "Cost time: $SECONDS seconds"
